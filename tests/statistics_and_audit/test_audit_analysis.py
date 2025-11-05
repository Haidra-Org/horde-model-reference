"""Unit tests for the audit analysis module."""

from __future__ import annotations

import pytest

from horde_model_reference import KNOWN_IMAGE_GENERATION_BASELINE, MODEL_DOMAIN, MODEL_PURPOSE, ModelClassification
from horde_model_reference.analytics.audit_analysis import (
    CategoryAuditSummary,
    DeletionRiskFlags,
    DeletionRiskFlagsFactory,
    DeletionRiskFlagsHandler,
    GenericDeletionRiskFlagsHandler,
    GenericModelAuditHandler,
    ImageGenerationDeletionRiskFlagsHandler,
    ImageGenerationModelAuditHandler,
    ModelAuditInfo,
    ModelAuditInfoFactory,
    ModelAuditInfoHandler,
    TextGenerationDeletionRiskFlagsHandler,
    UsageTrend,
)
from horde_model_reference.integrations.data_merger import CombinedModelStatistics, UsageStats, WorkerSummary
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecord,
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
)


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


class TestAnalyzeModelsForAudit:
    """Tests for ModelAuditInfoFactory.analyze_models method."""

    def test_analyze_empty_dict(self) -> None:
        """Test analyzing empty model dictionary."""
        factory = ModelAuditInfoFactory.create_default()
        audit_models = factory.analyze_models({}, {}, 0, MODEL_REFERENCE_CATEGORY.image_generation)

        assert len(audit_models) == 0

    def test_analyze_single_model(self) -> None:
        """Test analyzing a single model."""
        # Create a typed model record
        model_record = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="test_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            nsfw=False,
            description="A test model",
            config=GenericModelRecordConfig(
                download=[
                    DownloadRecord(
                        file_name="model.safetensors",
                        file_url="https://huggingface.co/model.safetensors",
                        sha256sum="abc123",
                    )
                ]
            ),
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
        )

        # Create statistics
        statistics = CombinedModelStatistics(
            usage_stats=UsageStats(day=10, month=100, total=500),
            worker_summaries={
                "worker1": WorkerSummary(
                    id="1", name="worker1", performance="1.0", online=True, trusted=True, uptime=100
                ),
                "worker2": WorkerSummary(
                    id="2", name="worker2", performance="1.0", online=True, trusted=True, uptime=100
                ),
                "worker3": WorkerSummary(
                    id="3", name="worker3", performance="1.0", online=True, trusted=True, uptime=100
                ),
                "worker4": WorkerSummary(
                    id="4", name="worker4", performance="1.0", online=True, trusted=True, uptime=100
                ),
                "worker5": WorkerSummary(
                    id="5", name="worker5", performance="1.0", online=True, trusted=True, uptime=100
                ),
            },
        )

        model_records: dict[str, ImageGenerationModelRecord] = {"test_model": model_record}
        model_statistics: dict[str, CombinedModelStatistics] = {"test_model": statistics}

        factory = ModelAuditInfoFactory.create_default()
        audit_models = factory.analyze_models(
            model_records,
            model_statistics,
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
        # Create a model record with risks
        model_record = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="risky_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            config=GenericModelRecordConfig(download=[]),  # No downloads - risk
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
        )

        # Create statistics with risks
        statistics = CombinedModelStatistics(
            usage_stats=UsageStats(day=0, month=0, total=10),
            worker_summaries=None,  # No workers - risk
        )

        model_records = {"risky_model": model_record}
        model_statistics = {"risky_model": statistics}

        factory = ModelAuditInfoFactory.create_default()
        audit_models = factory.analyze_models(
            model_records,
            model_statistics,
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
        # Create model records
        low_usage_model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="low_usage",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        )
        high_usage_model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="high_usage",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        )
        medium_usage_model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="medium_usage",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        )

        model_records = {
            "low_usage": low_usage_model,
            "high_usage": high_usage_model,
            "medium_usage": medium_usage_model,
        }

        # Create statistics with different usage levels
        model_statistics = {
            "low_usage": CombinedModelStatistics(usage_stats=UsageStats(day=1, month=10, total=50)),
            "high_usage": CombinedModelStatistics(usage_stats=UsageStats(day=5, month=100, total=500)),
            "medium_usage": CombinedModelStatistics(usage_stats=UsageStats(day=3, month=50, total=200)),
        }

        factory = ModelAuditInfoFactory.create_default()
        audit_models = factory.analyze_models(
            model_records,
            model_statistics,
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
        summary = CategoryAuditSummary.from_audit_models([])

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

        summary = CategoryAuditSummary.from_audit_models(audit_models)

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

        summary = CategoryAuditSummary.from_audit_models(audit_models)

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

        summary = CategoryAuditSummary.from_audit_models(audit_models)

        assert summary.total_models == 3
        assert summary.models_at_risk == 3
        assert summary.models_with_no_downloads == 1
        assert summary.models_with_non_preferred_hosts == 1
        assert summary.models_with_no_active_workers == 1
        assert summary.models_with_low_usage == 1
        assert summary.average_risk_score == pytest.approx(1.33, abs=0.01)


class TestModelAuditInfoFactory:
    """Tests for ModelAuditInfoFactory and handler system."""

    def test_create_default_factory(self) -> None:
        """Test creating factory with default handlers."""
        factory = ModelAuditInfoFactory.create_default()

        assert factory is not None
        assert len(factory._handlers) == 3  # Image, Text, Generic

    def test_factory_with_image_generation_model(self) -> None:
        """Test factory creates correct audit info for image generation model."""
        factory = ModelAuditInfoFactory.create_default()

        model_record = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="test_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            nsfw=False,
            description="A test model",
            config=GenericModelRecordConfig(
                download=[
                    DownloadRecord(
                        file_name="model.safetensors",
                        file_url="https://huggingface.co/model.safetensors",
                        sha256sum="abc123",
                    )
                ]
            ),
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
        )

        statistics = CombinedModelStatistics(
            usage_stats=UsageStats(day=10, month=100, total=500),
            worker_summaries={},
        )

        audit_info = factory.create_audit_info(
            model_name="test_model",
            model_record=model_record,
            statistics=statistics,
            category_total_usage=10000,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        assert audit_info.name == "test_model"
        assert audit_info.category == MODEL_REFERENCE_CATEGORY.image_generation
        assert audit_info.usage_month == 100
        assert audit_info.baseline == "stable_diffusion_xl"
        assert audit_info.nsfw is False

    def test_factory_with_custom_handler(self) -> None:
        """Test factory can use custom handler."""

        class CustomHandler(ModelAuditInfoHandler):
            """Custom handler for testing."""

            def can_handle(self, model_record: GenericModelRecord) -> bool:
                """Check if model name starts with 'custom_'."""
                return model_record.name.startswith("custom_")

            def create_audit_info(
                self,
                *,
                model_name: str,
                model_record: GenericModelRecord,
                statistics: CombinedModelStatistics | None,
                category_total_usage: int,
                category: MODEL_REFERENCE_CATEGORY,
            ) -> ModelAuditInfo:
                """Create custom audit info with hardcoded risk score."""
                from horde_model_reference.analytics.audit_analysis import (
                    DeletionRiskFlags,
                    ModelAuditInfo,
                    UsageTrend,
                )

                # Custom logic: all custom models are at risk
                flags = DeletionRiskFlags(missing_description=True)

                return ModelAuditInfo(
                    name=model_name,
                    category=category,
                    deletion_risk_flags=flags,
                    at_risk=True,
                    risk_score=999,  # Custom risk score
                    usage_trend=UsageTrend(),
                    worker_count=0,
                    usage_day=0,
                    usage_month=0,
                    usage_total=0,
                )

        factory = ModelAuditInfoFactory()
        factory.register_handler(CustomHandler())
        factory.register_handler(ImageGenerationModelAuditHandler())
        factory.register_handler(GenericModelAuditHandler())

        # Create a model that will be handled by custom handler
        custom_model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="custom_special_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            nsfw=False,
            description="Custom model",
            config=GenericModelRecordConfig(),
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
        )

        audit_info = factory.create_audit_info(
            model_name="custom_special_model",
            model_record=custom_model,
            statistics=None,
            category_total_usage=0,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        assert audit_info.risk_score == 999
        assert audit_info.at_risk is True

    def test_factory_handler_order(self) -> None:
        """Test that handlers are checked in registration order."""

        class FirstHandler(ModelAuditInfoHandler):
            """Handler that accepts all models."""

            def can_handle(self, model_record: GenericModelRecord) -> bool:
                return True

            def create_audit_info(
                self,
                *,
                model_name: str,
                model_record: GenericModelRecord,
                statistics: CombinedModelStatistics | None,
                category_total_usage: int,
                category: MODEL_REFERENCE_CATEGORY,
            ) -> ModelAuditInfo:
                from horde_model_reference.analytics.audit_analysis import (
                    DeletionRiskFlags,
                    ModelAuditInfo,
                    UsageTrend,
                )

                return ModelAuditInfo(
                    name=model_name,
                    category=category,
                    deletion_risk_flags=DeletionRiskFlags(),
                    at_risk=False,
                    risk_score=1,  # First handler marker
                    usage_trend=UsageTrend(),
                    worker_count=0,
                    usage_day=0,
                    usage_month=0,
                    usage_total=0,
                )

        class SecondHandler(ModelAuditInfoHandler):
            """Handler that also accepts all models."""

            def can_handle(self, model_record: GenericModelRecord) -> bool:
                return True

            def create_audit_info(
                self,
                *,
                model_name: str,
                model_record: GenericModelRecord,
                statistics: CombinedModelStatistics | None,
                category_total_usage: int,
                category: MODEL_REFERENCE_CATEGORY,
            ) -> ModelAuditInfo:
                from horde_model_reference.analytics.audit_analysis import (
                    DeletionRiskFlags,
                    ModelAuditInfo,
                    UsageTrend,
                )

                return ModelAuditInfo(
                    name=model_name,
                    category=category,
                    deletion_risk_flags=DeletionRiskFlags(),
                    at_risk=False,
                    risk_score=2,  # Second handler marker
                    usage_trend=UsageTrend(),
                    worker_count=0,
                    usage_day=0,
                    usage_month=0,
                    usage_total=0,
                )

        # First handler registered first, should be used
        factory = ModelAuditInfoFactory()
        factory.register_handler(FirstHandler())
        factory.register_handler(SecondHandler())

        model_record = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="test",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            nsfw=False,
            config=GenericModelRecordConfig(),
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
        )

        audit_info = factory.create_audit_info(
            model_name="test",
            model_record=model_record,
            statistics=None,
            category_total_usage=0,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        # Should use FirstHandler since it was registered first
        assert audit_info.risk_score == 1

    def test_factory_no_matching_handler(self) -> None:
        """Test factory raises error when no handler matches."""

        class NeverMatchHandler(ModelAuditInfoHandler):
            """Handler that never matches."""

            def can_handle(self, model_record: GenericModelRecord) -> bool:
                return False

            def create_audit_info(
                self,
                *,
                model_name: str,
                model_record: GenericModelRecord,
                statistics: CombinedModelStatistics | None,
                category_total_usage: int,
                category: MODEL_REFERENCE_CATEGORY,
            ) -> ModelAuditInfo:
                raise NotImplementedError("Should never be called")

        factory = ModelAuditInfoFactory()
        factory.register_handler(NeverMatchHandler())

        model_record = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="test",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            nsfw=False,
            config=GenericModelRecordConfig(),
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
        )

        with pytest.raises(ValueError, match="No handler found"):
            factory.create_audit_info(
                model_name="test",
                model_record=model_record,
                statistics=None,
                category_total_usage=0,
                category=MODEL_REFERENCE_CATEGORY.image_generation,
            )

    def test_analyze_with_custom_factory(self) -> None:
        """Test analyze_models_for_audit accepts custom factory."""

        class AlwaysAtRiskHandler(ModelAuditInfoHandler):
            """Handler that marks all models as at risk."""

            def can_handle(self, model_record: GenericModelRecord) -> bool:
                return True

            def create_audit_info(
                self,
                *,
                model_name: str,
                model_record: GenericModelRecord,
                statistics: CombinedModelStatistics | None,
                category_total_usage: int,
                category: MODEL_REFERENCE_CATEGORY,
            ) -> ModelAuditInfo:
                from horde_model_reference.analytics.audit_analysis import (
                    DeletionRiskFlags,
                    ModelAuditInfo,
                    UsageTrend,
                )

                return ModelAuditInfo(
                    name=model_name,
                    category=category,
                    deletion_risk_flags=DeletionRiskFlags(zero_usage_month=True),
                    at_risk=True,
                    risk_score=10,
                    usage_trend=UsageTrend(),
                    worker_count=0,
                    usage_day=0,
                    usage_month=0,
                    usage_total=0,
                )

        custom_factory = ModelAuditInfoFactory()
        custom_factory.register_handler(AlwaysAtRiskHandler())

        model_records = {
            "model1": ImageGenerationModelRecord(
                record_type=MODEL_REFERENCE_CATEGORY.image_generation,
                name="model1",
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
                nsfw=False,
                config=GenericModelRecordConfig(),
                model_classification=ModelClassification(
                    domain=MODEL_DOMAIN.image,
                    purpose=MODEL_PURPOSE.generation,
                ),
            )
        }

        audit_models = custom_factory.analyze_models(
            model_records,
            {},
            0,
            MODEL_REFERENCE_CATEGORY.image_generation,
        )

        assert len(audit_models) == 1
        assert audit_models[0].at_risk is True
        assert audit_models[0].risk_score == 10


class TestSemanticBusinessLogic:
    """Semantic tests verifying actual business rules and edge cases."""

    def test_low_usage_threshold_boundary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that 0.1% threshold correctly identifies low usage models.

        Business rule: Models with < 0.1% of category usage are flagged as low_usage.
        """
        from horde_model_reference import horde_model_reference_settings

        # Override the default threshold to match this test's business rule
        monkeypatch.setattr(horde_model_reference_settings, "low_usage_threshold_percentage", 0.1)

        factory = ModelAuditInfoFactory.create_default()

        # Category has 10,000 total monthly usage
        # 0.1% threshold = 10 usage
        category_total = 10000

        # Model with 9 usage (0.09%) should be flagged as low_usage
        low_usage_model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="low_usage_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            config=GenericModelRecordConfig(),
            model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        )

        # Model with 11 usage (0.11%) should NOT be flagged
        acceptable_usage_model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="acceptable_usage_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            config=GenericModelRecordConfig(),
            model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        )

        low_stats = CombinedModelStatistics(usage_stats=UsageStats(day=1, month=9, total=100))
        acceptable_stats = CombinedModelStatistics(usage_stats=UsageStats(day=1, month=11, total=100))

        low_audit = factory.create_audit_info(
            model_name="low_usage_model",
            model_record=low_usage_model,
            statistics=low_stats,
            category_total_usage=category_total,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        acceptable_audit = factory.create_audit_info(
            model_name="acceptable_usage_model",
            model_record=acceptable_usage_model,
            statistics=acceptable_stats,
            category_total_usage=category_total,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        # Verify semantic meaning: usage below threshold is flagged
        assert low_audit.deletion_risk_flags.low_usage
        assert not acceptable_audit.deletion_risk_flags.low_usage

        # Verify percentage calculations
        assert low_audit.usage_percentage_of_category == pytest.approx(0.09, abs=0.001)
        assert acceptable_audit.usage_percentage_of_category == pytest.approx(0.11, abs=0.001)

    def test_low_usage_with_zero_category_usage(self) -> None:
        """Test low_usage flag when category has zero total usage (edge case)."""
        factory = ModelAuditInfoFactory.create_default()

        model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="test_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            config=GenericModelRecordConfig(),
            model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        )

        stats = CombinedModelStatistics(usage_stats=UsageStats(day=0, month=0, total=0))

        audit = factory.create_audit_info(
            model_name="test_model",
            model_record=model,
            statistics=stats,
            category_total_usage=0,  # Division by zero case
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        # When category total is 0, percentage should be 0 (not NaN or error)
        assert audit.usage_percentage_of_category == 0.0
        # Should not be flagged as low_usage when we can't calculate percentage
        assert not audit.deletion_risk_flags.low_usage

    def test_is_critical_requires_both_conditions(self) -> None:
        """Test that is_critical requires BOTH zero month usage AND no active workers.

        Business rule: A model is only critical if it has BOTH conditions, not just one.
        """
        factory = ModelAuditInfoFactory.create_default()

        model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="test_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            config=GenericModelRecordConfig(),
            model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        )

        # Case 1: Zero usage but HAS workers - NOT critical
        stats_with_workers = CombinedModelStatistics(
            usage_stats=UsageStats(day=0, month=0, total=0),
            worker_summaries={
                "worker1": WorkerSummary(
                    id="1", name="worker1", performance="1.0", online=True, trusted=True, uptime=100
                )
            },
        )

        audit_with_workers = factory.create_audit_info(
            model_name="test_model",
            model_record=model,
            statistics=stats_with_workers,
            category_total_usage=1000,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        # Case 2: No workers but HAS usage - NOT critical
        stats_with_usage = CombinedModelStatistics(
            usage_stats=UsageStats(day=10, month=100, total=1000),
            worker_summaries={},
        )

        audit_with_usage = factory.create_audit_info(
            model_name="test_model",
            model_record=model,
            statistics=stats_with_usage,
            category_total_usage=10000,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        # Case 3: BOTH zero usage AND no workers - IS critical
        stats_critical = CombinedModelStatistics(
            usage_stats=UsageStats(day=0, month=0, total=0),
            worker_summaries={},
        )

        audit_critical = factory.create_audit_info(
            model_name="test_model",
            model_record=model,
            statistics=stats_critical,
            category_total_usage=1000,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        # Verify semantic meaning: critical requires BOTH conditions
        assert not audit_with_workers.is_critical, "Model with workers should not be critical"
        assert not audit_with_usage.is_critical, "Model with usage should not be critical"
        assert audit_critical.is_critical, "Model with zero usage AND no workers should be critical"

    def test_cost_benefit_calculation(self) -> None:
        """Test cost-benefit score calculation with concrete values.

        Business rule: cost_benefit = usage_month / size_gb
        Higher scores indicate better value (more usage per GB).
        """
        factory = ModelAuditInfoFactory.create_default()

        # Model with 1000 monthly usage and 5GB size
        # Expected cost-benefit: 1000 / 5 = 200.0
        model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="efficient_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            size_on_disk_bytes=5 * 1024 * 1024 * 1024,  # 5GB in bytes
            config=GenericModelRecordConfig(),
            model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        )

        stats = CombinedModelStatistics(usage_stats=UsageStats(day=50, month=1000, total=5000))

        audit = factory.create_audit_info(
            model_name="efficient_model",
            model_record=model,
            statistics=stats,
            category_total_usage=10000,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        # Verify cost-benefit calculation
        assert audit.size_gb == pytest.approx(5.0, abs=0.01)
        assert audit.cost_benefit_score == pytest.approx(200.0, abs=0.01)

    def test_cost_benefit_with_zero_size(self) -> None:
        """Test cost-benefit score when model has no size info (edge case)."""
        factory = ModelAuditInfoFactory.create_default()

        model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="no_size_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            size_on_disk_bytes=None,  # No size info
            config=GenericModelRecordConfig(),
            model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        )

        stats = CombinedModelStatistics(usage_stats=UsageStats(day=10, month=100, total=500))

        audit = factory.create_audit_info(
            model_name="no_size_model",
            model_record=model,
            statistics=stats,
            category_total_usage=1000,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        # When no size info, cost-benefit should be None
        assert audit.size_gb is None
        assert audit.cost_benefit_score is None

    def test_usage_trend_ratios(self) -> None:
        """Test usage trend ratio calculations.

        Business rule: Ratios help identify momentum and activity patterns.
        - day_to_month_ratio > 1.0 indicates accelerating usage
        - month_to_total_ratio shows recent vs historical activity
        """
        factory = ModelAuditInfoFactory.create_default()

        # Accelerating model: day usage is 20% of month (high recent activity)
        accelerating_model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="accelerating",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            config=GenericModelRecordConfig(),
            model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        )

        accelerating_stats = CombinedModelStatistics(
            usage_stats=UsageStats(day=200, month=1000, total=2000)  # day is 20% of month
        )

        accelerating_audit = factory.create_audit_info(
            model_name="accelerating",
            model_record=accelerating_model,
            statistics=accelerating_stats,
            category_total_usage=10000,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        # Declining model: day usage is only 1% of month (low recent activity)
        declining_model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="declining",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            config=GenericModelRecordConfig(),
            model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        )

        declining_stats = CombinedModelStatistics(
            usage_stats=UsageStats(day=10, month=1000, total=10000)  # day is 1% of month
        )

        declining_audit = factory.create_audit_info(
            model_name="declining",
            model_record=declining_model,
            statistics=declining_stats,
            category_total_usage=50000,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        # Verify trend ratios
        assert accelerating_audit.usage_trend.day_to_month_ratio == pytest.approx(0.2, abs=0.01)
        assert accelerating_audit.usage_trend.month_to_total_ratio == pytest.approx(0.5, abs=0.01)

        assert declining_audit.usage_trend.day_to_month_ratio == pytest.approx(0.01, abs=0.001)
        assert declining_audit.usage_trend.month_to_total_ratio == pytest.approx(0.1, abs=0.01)

    def test_usage_trend_division_by_zero(self) -> None:
        """Test usage trend ratios handle division by zero gracefully."""
        factory = ModelAuditInfoFactory.create_default()

        model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="zero_usage",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            config=GenericModelRecordConfig(),
            model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        )

        # Zero month usage (can't calculate day_to_month_ratio)
        stats = CombinedModelStatistics(usage_stats=UsageStats(day=5, month=0, total=100))

        audit = factory.create_audit_info(
            model_name="zero_usage",
            model_record=model,
            statistics=stats,
            category_total_usage=1000,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        # Ratios should be None when denominator is zero
        assert audit.usage_trend.day_to_month_ratio is None
        assert audit.usage_trend.month_to_total_ratio is not None  # total is not zero

    def test_multiple_download_hosts(self) -> None:
        """Test that models with downloads from multiple hosts are flagged.

        Business rule: Models should consolidate downloads on a single host.
        """
        factory = ModelAuditInfoFactory.create_default()

        model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="multi_host_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            config=GenericModelRecordConfig(
                download=[
                    DownloadRecord(
                        file_name="model1.safetensors",
                        file_url="https://huggingface.co/model1.safetensors",
                        sha256sum="abc123",
                    ),
                    DownloadRecord(
                        file_name="model2.safetensors",
                        file_url="https://civitai.com/model2.safetensors",
                        sha256sum="def456",
                    ),
                ]
            ),
            model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        )

        audit = factory.create_audit_info(
            model_name="multi_host_model",
            model_record=model,
            statistics=None,
            category_total_usage=1000,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        # Should be flagged for multiple hosts
        assert audit.deletion_risk_flags.has_multiple_hosts
        assert len(audit.download_hosts) == 2

        # download_hosts already contains hostnames, not full URLs
        assert "huggingface.co" in audit.download_hosts
        assert "civitai.com" in audit.download_hosts

    def test_non_preferred_host_detection(self) -> None:
        """Test that non-preferred hosts are detected.

        Business rule: Only huggingface.co is preferred (from settings).
        """
        factory = ModelAuditInfoFactory.create_default()

        # Model hosted on civitai (non-preferred)
        non_preferred_model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="civitai_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            config=GenericModelRecordConfig(
                download=[
                    DownloadRecord(
                        file_name="model.safetensors",
                        file_url="https://civitai.com/model.safetensors",
                        sha256sum="abc123",
                    )
                ]
            ),
            model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        )

        # Model hosted on huggingface (preferred)
        preferred_model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="hf_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            config=GenericModelRecordConfig(
                download=[
                    DownloadRecord(
                        file_name="model.safetensors",
                        file_url="https://huggingface.co/model.safetensors",
                        sha256sum="abc123",
                    )
                ]
            ),
            model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        )

        non_preferred_audit = factory.create_audit_info(
            model_name="civitai_model",
            model_record=non_preferred_model,
            statistics=None,
            category_total_usage=1000,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        preferred_audit = factory.create_audit_info(
            model_name="hf_model",
            model_record=preferred_model,
            statistics=None,
            category_total_usage=1000,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        # Verify host preference detection
        assert non_preferred_audit.deletion_risk_flags.has_non_preferred_host
        assert not preferred_audit.deletion_risk_flags.has_non_preferred_host

    def test_malformed_url_handling(self) -> None:
        """Test that malformed download URLs are flagged as unknown hosts."""
        factory = ModelAuditInfoFactory.create_default()

        model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="bad_url_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            config=GenericModelRecordConfig(
                download=[
                    DownloadRecord(
                        file_name="model.safetensors",
                        file_url="not-a-valid-url",  # Malformed URL
                        sha256sum="abc123",
                    )
                ]
            ),
            model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        )

        audit = factory.create_audit_info(
            model_name="bad_url_model",
            model_record=model,
            statistics=None,
            category_total_usage=1000,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        # Malformed URLs should be flagged (either as unknown host or no valid URLs)
        assert audit.deletion_risk_flags.has_unknown_host or audit.deletion_risk_flags.no_download_urls

    def test_scenario_popular_model_not_flagged(self) -> None:
        """Scenario test: A popular, well-configured model should have no risk flags.

        Business scenario: The most popular model in a category should be
        considered safe and not flagged for deletion.
        """
        factory = ModelAuditInfoFactory.create_default()

        popular_model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="popular_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            nsfw=False,
            description="A very popular and well-maintained model",
            size_on_disk_bytes=7 * 1024 * 1024 * 1024,  # 7GB
            config=GenericModelRecordConfig(
                download=[
                    DownloadRecord(
                        file_name="model.safetensors",
                        file_url="https://huggingface.co/popular/model.safetensors",
                        sha256sum="abc123",
                    )
                ]
            ),
            model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        )

        # High usage (50% of category total) with many active workers
        stats = CombinedModelStatistics(
            usage_stats=UsageStats(day=500, month=5000, total=50000),
            worker_summaries={
                f"worker{i}": WorkerSummary(
                    id=str(i), name=f"worker{i}", performance="10", online=True, trusted=True, uptime=10000
                )
                for i in range(50)  # 50 active workers
            },
        )

        audit = factory.create_audit_info(
            model_name="popular_model",
            model_record=popular_model,
            statistics=stats,
            category_total_usage=10000,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        # Popular model should have NO risk flags
        assert not audit.at_risk
        assert audit.risk_score == 0
        assert not audit.is_critical
        assert not audit.has_warning
        assert audit.worker_count == 50
        assert audit.usage_percentage_of_category == 50.0
        assert audit.cost_benefit_score is not None and audit.cost_benefit_score > 0

    def test_scenario_abandoned_model_is_critical(self) -> None:
        """Scenario test: An abandoned model should be flagged as critical.

        Business scenario: A model with no downloads, no usage, and no workers
        should be clearly identified as a candidate for deletion.
        """
        factory = ModelAuditInfoFactory.create_default()

        abandoned_model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="abandoned_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            # No description
            description=None,
            # No downloads
            config=GenericModelRecordConfig(download=[]),
            model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        )

        # Zero usage, no workers
        stats = CombinedModelStatistics(
            usage_stats=UsageStats(day=0, month=0, total=0),
            worker_summaries={},
        )

        audit = factory.create_audit_info(
            model_name="abandoned_model",
            model_record=abandoned_model,
            statistics=stats,
            category_total_usage=10000,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        # Abandoned model should be critical with multiple risk flags
        assert audit.at_risk
        assert audit.is_critical
        assert audit.risk_score > 3  # Multiple flags
        assert audit.deletion_risk_flags.no_download_urls
        assert audit.deletion_risk_flags.zero_usage_month
        assert audit.deletion_risk_flags.no_active_workers
        assert audit.deletion_risk_flags.missing_description

    def test_scenario_niche_model_warning_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Scenario test: A niche model with low usage but active workers.

        Business scenario: A model with very low usage but still has active
        workers serving it should be flagged for review but not critical.
        """
        from horde_model_reference import horde_model_reference_settings

        # Override threshold to 0.1% to match test expectations (0.05% usage should be flagged)
        monkeypatch.setattr(horde_model_reference_settings, "low_usage_threshold_percentage", 0.1)

        factory = ModelAuditInfoFactory.create_default()

        niche_model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="niche_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            description="A specialized niche model for specific use cases",
            config=GenericModelRecordConfig(
                download=[
                    DownloadRecord(
                        file_name="model.safetensors",
                        file_url="https://huggingface.co/niche/model.safetensors",
                        sha256sum="abc123",
                    )
                ]
            ),
            model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        )

        # Low usage (0.05% of category) but has dedicated workers
        stats = CombinedModelStatistics(
            usage_stats=UsageStats(day=0, month=5, total=100),  # Only 5 monthly usage
            worker_summaries={
                "worker1": WorkerSummary(
                    id="1", name="dedicated_worker", performance="10", online=True, trusted=True, uptime=10000
                )
            },
        )

        audit = factory.create_audit_info(
            model_name="niche_model",
            model_record=niche_model,
            statistics=stats,
            category_total_usage=10000,  # 5/10000 = 0.05%
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        # Should be at risk due to low usage, but NOT critical (has workers)
        assert audit.at_risk
        assert not audit.is_critical
        assert audit.deletion_risk_flags.low_usage
        assert not audit.deletion_risk_flags.no_active_workers
        assert audit.worker_count == 1
        assert audit.usage_percentage_of_category == pytest.approx(0.05, abs=0.001)


class TestDeletionRiskFlagsFactory:
    """Tests for DeletionRiskFlagsFactory and related handlers."""

    def test_create_default_factory(self) -> None:
        """Test creating a factory with default handlers."""
        factory = DeletionRiskFlagsFactory.create_default()

        assert factory is not None
        assert len(factory._handlers) == 3  # Image, Text, Generic

    def test_image_generation_handler_can_handle(self) -> None:
        """Test ImageGenerationDeletionRiskFlagsHandler can handle image models."""
        handler = ImageGenerationDeletionRiskFlagsHandler()

        image_model = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="test_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            nsfw=False,
            config=GenericModelRecordConfig(),
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
        )

        assert handler.can_handle(image_model)

    def test_text_generation_handler_can_handle(self) -> None:
        """Test TextGenerationDeletionRiskFlagsHandler can handle text models."""
        from horde_model_reference.model_reference_records import TextGenerationModelRecord

        handler = TextGenerationDeletionRiskFlagsHandler()

        text_model = TextGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.text_generation,
            name="test_model",
            baseline="llama3",
            parameters=7000000000,
            nsfw=False,
            config=GenericModelRecordConfig(),
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.text,
                purpose=MODEL_PURPOSE.generation,
            ),
        )

        assert handler.can_handle(text_model)

    def test_generic_handler_can_handle_all(self) -> None:
        """Test GenericDeletionRiskFlagsHandler accepts all model types."""
        handler = GenericDeletionRiskFlagsHandler()

        generic_model = GenericModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="test_model",
            config=GenericModelRecordConfig(),
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
        )

        assert handler.can_handle(generic_model)

    def test_factory_create_flags_image_model(self) -> None:
        """Test factory creates flags for image generation model."""
        factory = DeletionRiskFlagsFactory.create_default()

        model_record = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="test_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            nsfw=False,
            config=GenericModelRecordConfig(
                download=[
                    DownloadRecord(
                        file_name="model.safetensors",
                        file_url="https://civitai.com/model.safetensors",
                        sha256sum="abc123",
                    )
                ]
            ),
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
        )

        statistics = CombinedModelStatistics(
            usage_stats=UsageStats(day=0, month=0, total=0),
        )

        flags = factory.create_flags(
            model_record=model_record,
            statistics=statistics,
            category_total_usage=1000,
        )

        assert isinstance(flags, DeletionRiskFlags)
        assert flags.no_active_workers
        assert flags.zero_usage_month
        assert flags.has_non_preferred_host  # civitai is not preferred

    def test_factory_create_flags_text_model(self) -> None:
        """Test factory creates flags for text generation model."""
        from horde_model_reference.model_reference_records import TextGenerationModelRecord

        factory = DeletionRiskFlagsFactory.create_default()

        model_record = TextGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.text_generation,
            name="test_model",
            baseline="llama3",
            parameters=7000000000,
            nsfw=False,
            config=GenericModelRecordConfig(
                download=[
                    DownloadRecord(
                        file_name="model.safetensors",
                        file_url="https://huggingface.co/model.safetensors",
                        sha256sum="abc123",
                    )
                ]
            ),
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.text,
                purpose=MODEL_PURPOSE.generation,
            ),
        )

        statistics = CombinedModelStatistics(
            usage_stats=UsageStats(day=10, month=100, total=1000),
            worker_summaries={
                "worker1": WorkerSummary(
                    id="worker1",
                    name="Worker 1",
                    performance="10",
                    online=True,
                    trusted=True,
                    uptime=3600,
                )
            },
        )

        flags = factory.create_flags(
            model_record=model_record,
            statistics=statistics,
            category_total_usage=10000,
        )

        assert isinstance(flags, DeletionRiskFlags)
        assert not flags.no_active_workers
        assert not flags.zero_usage_month
        assert not flags.has_non_preferred_host  # huggingface is preferred

    def test_custom_handler_registration(self) -> None:
        """Test registering a custom handler to the factory."""

        class CustomDeletionRiskFlagsHandler(DeletionRiskFlagsHandler):
            """Custom handler for testing."""

            def can_handle(self, model_record: GenericModelRecord) -> bool:
                """Return True for all models."""
                return True

            def create_flags(
                self,
                *,
                model_record: GenericModelRecord,
                statistics: CombinedModelStatistics | None,
                category_total_usage: int,
            ) -> DeletionRiskFlags:
                """Return flags with all risks marked."""
                return DeletionRiskFlags(
                    zero_usage_day=True,
                    zero_usage_month=True,
                    zero_usage_total=True,
                    no_active_workers=True,
                )

        factory = DeletionRiskFlagsFactory()
        factory.register_handler(CustomDeletionRiskFlagsHandler())

        model_record = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="test_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            nsfw=False,
            config=GenericModelRecordConfig(),
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
        )

        flags = factory.create_flags(
            model_record=model_record,
            statistics=None,
            category_total_usage=0,
        )

        assert flags.zero_usage_day
        assert flags.zero_usage_month
        assert flags.zero_usage_total
        assert flags.no_active_workers

    def test_factory_no_handler_found_raises_error(self) -> None:
        """Test factory raises error when no handler can process the model."""
        factory = DeletionRiskFlagsFactory()  # Empty factory

        model_record = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="test_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            nsfw=False,
            config=GenericModelRecordConfig(),
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
        )

        with pytest.raises(ValueError, match="No handler found for model record type"):
            factory.create_flags(
                model_record=model_record,
                statistics=None,
                category_total_usage=0,
            )

    def test_audit_handlers_use_flags_factory(self) -> None:
        """Test that audit handlers can be initialized with custom flags factory."""

        class AlwaysCriticalFlagsHandler(DeletionRiskFlagsHandler):
            """Custom handler that marks everything as critical."""

            def can_handle(self, model_record: GenericModelRecord) -> bool:
                """Return True for all models."""
                return True

            def create_flags(
                self,
                *,
                model_record: GenericModelRecord,
                statistics: CombinedModelStatistics | None,
                category_total_usage: int,
            ) -> DeletionRiskFlags:
                """Return critical flags."""
                return DeletionRiskFlags(
                    zero_usage_month=True,
                    no_active_workers=True,
                )

        custom_flags_factory = DeletionRiskFlagsFactory()
        custom_flags_factory.register_handler(AlwaysCriticalFlagsHandler())

        # Create audit handler with custom flags factory
        audit_handler = ImageGenerationModelAuditHandler(flags_factory=custom_flags_factory)

        model_record = ImageGenerationModelRecord(
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            name="test_model",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            nsfw=False,
            config=GenericModelRecordConfig(),
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
        )

        # Even with good stats, the custom factory should mark it critical
        statistics = CombinedModelStatistics(
            usage_stats=UsageStats(day=100, month=1000, total=10000),
            worker_summaries={
                "worker1": WorkerSummary(
                    id="worker1",
                    name="Worker 1",
                    performance="10",
                    online=True,
                    trusted=True,
                    uptime=3600,
                )
            },
        )

        audit_info = audit_handler.create_audit_info(
            model_name="test_model",
            model_record=model_record,
            statistics=statistics,
            category_total_usage=10000,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        assert audit_info.deletion_risk_flags.zero_usage_month
        assert audit_info.deletion_risk_flags.no_active_workers
        assert audit_info.is_critical
