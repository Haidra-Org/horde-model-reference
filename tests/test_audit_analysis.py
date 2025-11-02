"""Unit tests for the audit analysis module."""

from __future__ import annotations

import pytest

from horde_model_reference.analytics.audit_analysis import (
    DeletionRiskFlags,
    UsageTrend,
    analyze_models_for_audit,
    calculate_audit_summary,
    get_deletion_flags,
)
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class TestDeletionRiskFlags:
    """Tests for DeletionRiskFlags model."""

    def test_any_flags_all_false(self) -> None:
        """Test any_flags returns False when no flags are set."""
        flags = DeletionRiskFlags()

        assert not flags.any_flags()

    def test_any_flags_one_true(self) -> None:
        """Test any_flags returns True when at least one flag is set."""
        flags = DeletionRiskFlags(no_download_urls=True)

        assert flags.any_flags()

    def test_flag_count_zero(self) -> None:
        """Test flag_count returns 0 when no flags are set."""
        flags = DeletionRiskFlags()

        assert flags.flag_count() == 0

    def test_flag_count_multiple(self) -> None:
        """Test flag_count returns correct count for multiple flags."""
        flags = DeletionRiskFlags(
            no_download_urls=True,
            no_active_workers=True,
            low_usage=True,
        )

        assert flags.flag_count() == 3


class TestGetDeletionFlags:
    """Tests for get_deletion_flags function."""

    def test_missing_downloads_empty_list(self) -> None:
        """Test detection of missing downloads with empty list."""
        model_data = {
            "name": "test_model",
            "config": {"download": []},
        }

        flags = get_deletion_flags(model_data, MODEL_REFERENCE_CATEGORY.image_generation)

        assert flags.no_download_urls

    def test_missing_downloads_no_config(self) -> None:
        """Test detection of missing downloads with no config."""
        model_data = {"name": "test_model"}

        flags = get_deletion_flags(model_data, MODEL_REFERENCE_CATEGORY.image_generation)

        assert flags.no_download_urls

    def test_valid_download_huggingface(self) -> None:
        """Test valid download from preferred host (huggingface)."""
        model_data = {
            "name": "test_model",
            "config": {
                "download": [
                    {
                        "file_name": "model.safetensors",
                        "file_url": "https://huggingface.co/model.safetensors",
                        "sha256sum": "abc123",
                    }
                ]
            },
        }

        flags = get_deletion_flags(model_data, MODEL_REFERENCE_CATEGORY.image_generation)

        assert not flags.no_download_urls
        assert not flags.has_unknown_host
        assert not flags.has_non_preferred_host

    def test_non_preferred_host(self) -> None:
        """Test detection of non-preferred host."""
        model_data = {
            "name": "test_model",
            "config": {
                "download": [
                    {
                        "file_name": "model.safetensors",
                        "file_url": "https://example.com/model.safetensors",
                        "sha256sum": "abc123",
                    }
                ]
            },
        }

        flags = get_deletion_flags(model_data, MODEL_REFERENCE_CATEGORY.image_generation)

        assert not flags.no_download_urls
        assert not flags.has_unknown_host
        assert flags.has_non_preferred_host

    def test_invalid_download_url(self) -> None:
        """Test detection of invalid download URLs."""
        model_data = {
            "name": "test_model",
            "config": {
                "download": [
                    {
                        "file_name": "model.safetensors",
                        "file_url": "not_a_valid_url",
                        "sha256sum": "abc123",
                    }
                ]
            },
        }

        flags = get_deletion_flags(model_data, MODEL_REFERENCE_CATEGORY.image_generation)

        # Invalid URL with no valid scheme/netloc results in both flags
        assert flags.no_download_urls  # No valid parseable URLs
        assert not flags.has_unknown_host  # URL just lacks scheme, not a parse exception

    def test_missing_description(self) -> None:
        """Test detection of missing description."""
        model_data = {"name": "test_model"}

        flags = get_deletion_flags(model_data, MODEL_REFERENCE_CATEGORY.image_generation)

        assert flags.missing_description

    def test_empty_description(self) -> None:
        """Test detection of empty description."""
        model_data = {"name": "test_model", "description": "   "}

        flags = get_deletion_flags(model_data, MODEL_REFERENCE_CATEGORY.image_generation)

        assert flags.missing_description

    def test_valid_description(self) -> None:
        """Test valid description doesn't trigger flag."""
        model_data = {"name": "test_model", "description": "A valid model description"}

        flags = get_deletion_flags(model_data, MODEL_REFERENCE_CATEGORY.image_generation)

        assert not flags.missing_description

    def test_missing_baseline_image_generation(self) -> None:
        """Test detection of missing baseline for image_generation."""
        model_data = {"name": "test_model"}

        flags = get_deletion_flags(model_data, MODEL_REFERENCE_CATEGORY.image_generation)

        assert flags.missing_baseline

    def test_valid_baseline_image_generation(self) -> None:
        """Test valid baseline doesn't trigger flag."""
        model_data = {"name": "test_model", "baseline": "stable_diffusion_xl"}

        flags = get_deletion_flags(model_data, MODEL_REFERENCE_CATEGORY.image_generation)

        assert not flags.missing_baseline

    def test_baseline_not_checked_for_clip(self) -> None:
        """Test baseline not checked for categories like CLIP."""
        model_data = {"name": "test_model"}

        flags = get_deletion_flags(model_data, MODEL_REFERENCE_CATEGORY.clip)

        assert not flags.missing_baseline

    def test_zero_workers(self) -> None:
        """Test detection of zero workers from enriched data."""
        model_data = {"name": "test_model"}
        enriched_data = {"worker_count": 0}

        flags = get_deletion_flags(
            model_data,
            MODEL_REFERENCE_CATEGORY.image_generation,
            enriched_data,
        )

        assert flags.no_active_workers

    def test_nonzero_workers(self) -> None:
        """Test non-zero workers doesn't trigger flag."""
        model_data = {"name": "test_model"}
        enriched_data = {"worker_count": 5}

        flags = get_deletion_flags(
            model_data,
            MODEL_REFERENCE_CATEGORY.image_generation,
            enriched_data,
        )

        assert not flags.no_active_workers

    def test_no_recent_usage(self) -> None:
        """Test detection of no recent usage."""
        model_data = {"name": "test_model"}
        enriched_data = {"usage_stats": {"day": 0, "month": 0, "total": 100}}

        flags = get_deletion_flags(
            model_data,
            MODEL_REFERENCE_CATEGORY.image_generation,
            enriched_data,
        )

        assert flags.zero_usage_day
        assert flags.zero_usage_month
        assert not flags.zero_usage_total

    def test_recent_usage(self) -> None:
        """Test recent usage doesn't trigger flag."""
        model_data = {"name": "test_model"}
        enriched_data = {"usage_stats": {"day": 5, "month": 50, "total": 150}}

        flags = get_deletion_flags(
            model_data,
            MODEL_REFERENCE_CATEGORY.image_generation,
            enriched_data,
        )

        assert not flags.zero_usage_day
        assert not flags.zero_usage_month
        assert not flags.zero_usage_total

    def test_low_usage(self) -> None:
        """Test detection of low usage (< 0.1% of category total)."""
        model_data = {"name": "test_model"}
        enriched_data = {"usage_stats": {"month": 5, "total": 100}}

        # Category total is 10000, so 5/10000 = 0.05% < 0.1%
        flags = get_deletion_flags(
            model_data,
            MODEL_REFERENCE_CATEGORY.image_generation,
            enriched_data,
            category_total_usage=10000,
        )

        assert flags.low_usage

    def test_not_low_usage(self) -> None:
        """Test adequate usage doesn't trigger flag."""
        model_data = {"name": "test_model"}
        enriched_data = {"usage_stats": {"month": 50, "total": 200}}

        # Category total is 10000, so 50/10000 = 0.5% > 0.1%
        flags = get_deletion_flags(
            model_data,
            MODEL_REFERENCE_CATEGORY.image_generation,
            enriched_data,
            category_total_usage=10000,
        )

        assert not flags.low_usage


class TestAnalyzeModelsForAudit:
    """Tests for analyze_models_for_audit function."""

    def test_analyze_empty_dict(self) -> None:
        """Test analyzing empty model dictionary."""
        audit_models = analyze_models_for_audit({}, 0, MODEL_REFERENCE_CATEGORY.image_generation)

        assert len(audit_models) == 0

    def test_analyze_single_model(self) -> None:
        """Test analyzing a single model."""
        enriched_models = {
            "test_model": {
                "name": "test_model",
                "baseline": "stable_diffusion_xl",
                "nsfw": False,
                "description": "A test model",
                "config": {
                    "download": [
                        {
                            "file_name": "model.safetensors",
                            "file_url": "https://huggingface.co/model.safetensors",
                            "sha256sum": "abc123",
                        }
                    ]
                },
                "worker_count": 5,
                "usage_stats": {"day": 10, "month": 100, "total": 500},
            }
        }

        audit_models = analyze_models_for_audit(
            enriched_models,
            1000,
            MODEL_REFERENCE_CATEGORY.image_generation,
        )

        assert len(audit_models) == 1
        model = audit_models[0]
        assert model.name == "test_model"
        assert model.category == MODEL_REFERENCE_CATEGORY.image_generation
        assert not model.at_risk
        assert model.risk_score == 0
        assert model.worker_count == 5
        assert model.usage_day == 10
        assert model.usage_month == 100
        assert model.usage_total == 500
        assert model.baseline == "stable_diffusion_xl"
        assert model.nsfw is False
        assert model.has_description
        assert model.download_count == 1
        assert "huggingface.co" in model.download_hosts

    def test_analyze_model_at_risk(self) -> None:
        """Test analyzing a model with deletion risks."""
        enriched_models = {
            "risky_model": {
                "name": "risky_model",
                "config": {"download": []},
                "worker_count": 0,
                "usage_stats": {"day": 0, "month": 0, "total": 10},
            }
        }

        audit_models = analyze_models_for_audit(
            enriched_models,
            10000,
            MODEL_REFERENCE_CATEGORY.image_generation,
        )

        assert len(audit_models) == 1
        model = audit_models[0]
        assert model.at_risk
        assert model.risk_score > 0
        assert model.deletion_risk_flags.no_download_urls
        assert model.deletion_risk_flags.no_active_workers
        assert model.deletion_risk_flags.zero_usage_month

    def test_analyze_sorts_by_usage(self) -> None:
        """Test that models are sorted by usage (descending)."""
        enriched_models = {
            "low_usage": {"name": "low_usage", "usage_stats": {"day": 1, "month": 10, "total": 50}},
            "high_usage": {"name": "high_usage", "usage_stats": {"day": 5, "month": 100, "total": 500}},
            "medium_usage": {"name": "medium_usage", "usage_stats": {"day": 3, "month": 50, "total": 200}},
        }

        audit_models = analyze_models_for_audit(
            enriched_models,
            160,
            MODEL_REFERENCE_CATEGORY.image_generation,
        )

        assert len(audit_models) == 3
        assert audit_models[0].name == "high_usage"
        assert audit_models[1].name == "medium_usage"
        assert audit_models[2].name == "low_usage"


class TestCalculateAuditSummary:
    """Tests for calculate_audit_summary function."""

    def test_summary_empty_list(self) -> None:
        """Test summary calculation with empty list."""
        summary = calculate_audit_summary([])

        assert summary.total_models == 0
        assert summary.models_at_risk == 0
        assert summary.average_risk_score == 0.0

    def test_summary_no_risks(self) -> None:
        """Test summary calculation with models having no risks."""
        from horde_model_reference.analytics.audit_analysis import ModelAuditInfo

        audit_models = [
            ModelAuditInfo(
                name="model1",
                category=MODEL_REFERENCE_CATEGORY.image_generation,
                deletion_risk_flags=DeletionRiskFlags(),
                at_risk=False,
                risk_score=0,
                usage_trend=UsageTrend(),
            ),
            ModelAuditInfo(
                name="model2",
                category=MODEL_REFERENCE_CATEGORY.image_generation,
                deletion_risk_flags=DeletionRiskFlags(),
                at_risk=False,
                risk_score=0,
                usage_trend=UsageTrend(),
            ),
        ]

        summary = calculate_audit_summary(audit_models)

        assert summary.total_models == 2
        assert summary.models_at_risk == 0
        assert summary.average_risk_score == 0.0

    def test_summary_with_risks(self) -> None:
        """Test summary calculation with models having risks."""
        from horde_model_reference.analytics.audit_analysis import ModelAuditInfo

        audit_models = [
            ModelAuditInfo(
                name="risky_model",
                category=MODEL_REFERENCE_CATEGORY.image_generation,
                deletion_risk_flags=DeletionRiskFlags(no_download_urls=True, no_active_workers=True),
                at_risk=True,
                risk_score=2,
                usage_trend=UsageTrend(),
            ),
            ModelAuditInfo(
                name="safe_model",
                category=MODEL_REFERENCE_CATEGORY.image_generation,
                deletion_risk_flags=DeletionRiskFlags(),
                at_risk=False,
                risk_score=0,
                usage_trend=UsageTrend(),
            ),
        ]

        summary = calculate_audit_summary(audit_models)

        assert summary.total_models == 2
        assert summary.models_at_risk == 1
        assert summary.models_with_no_downloads == 1
        assert summary.models_with_no_active_workers == 1
        assert summary.average_risk_score == 1.0

    def test_summary_counts_specific_flags(self) -> None:
        """Test summary correctly counts specific flag types."""
        from horde_model_reference.analytics.audit_analysis import ModelAuditInfo

        audit_models = [
            ModelAuditInfo(
                name="model1",
                category=MODEL_REFERENCE_CATEGORY.image_generation,
                deletion_risk_flags=DeletionRiskFlags(no_download_urls=True),
                at_risk=True,
                risk_score=1,
                usage_trend=UsageTrend(),
            ),
            ModelAuditInfo(
                name="model2",
                category=MODEL_REFERENCE_CATEGORY.image_generation,
                deletion_risk_flags=DeletionRiskFlags(has_non_preferred_host=True),
                at_risk=True,
                risk_score=1,
                usage_trend=UsageTrend(),
            ),
            ModelAuditInfo(
                name="model3",
                category=MODEL_REFERENCE_CATEGORY.image_generation,
                deletion_risk_flags=DeletionRiskFlags(no_active_workers=True, low_usage=True),
                at_risk=True,
                risk_score=2,
                usage_trend=UsageTrend(),
            ),
        ]

        summary = calculate_audit_summary(audit_models)

        assert summary.total_models == 3
        assert summary.models_at_risk == 3
        assert summary.models_with_no_downloads == 1
        assert summary.models_with_non_preferred_hosts == 1
        assert summary.models_with_no_active_workers == 1
        assert summary.models_with_low_usage == 1
        assert summary.average_risk_score == pytest.approx(1.33, abs=0.01)
