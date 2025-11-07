# Horde Model Reference Tests

This directory contains the test suite for the Horde Model Reference library and service.

## Test Organization

### Core Unit Tests (Root Level)

- `test_canonical_format.py` - Tests for canonical format conversion
- `test_consts.py` - Tests for constants and configuration
- `test_converters.py` - Tests for model record converters
- `test_convert_legacy_database.py` - Tests for legacy database migration
- `test_env_example.py` - Tests for example environment file
- `test_examples.py` - Tests for example JSON files
- `test_metadata.py` - Tests for model metadata handling
- `test_model_reference_manager.py` - Tests for the main manager class
- `test_records.py` - Tests for model record classes
- `test_scripts.py` - Tests for utility scripts

### Backend Tests (`backends/`)

- `test_http_backend.py` - Tests for HTTP-based model reference backend
- `test_primary_mode.py` - Tests for primary deployment mode
- `test_redis_backend.py` - Tests for Redis caching backend

### Service API Tests (`service/`)

- `test_replica_backend_base.py` - Tests for replica backend base functionality
- `test_v1_api.py` - Tests for v1 API endpoints
- `test_v2_api.py` - Tests for v2 API CRUD operations

### Statistics and Audit Tests (`statistics_and_audit/`)

- `test_audit_analysis.py` - Unit tests for audit analysis logic
- `test_statistics.py` - Tests for statistics calculations
- `test_statistics_cache.py` - Tests for statistics caching
- `test_text_model_grouping.py` - Tests for text model grouping logic
- `test_text_model_parser.py` - Tests for text model name parsing

### Sync Tests (`sync/`)

- `test_comparator.py` - Tests for model reference comparison
- `test_comparator_integration.py` - Integration tests for comparator with live data
- `test_config.py` - Tests for sync configuration
- `test_legacy_text_validator.py` - Tests for legacy text model validation

### Horde API Integration Tests (`horde_api/`)

- `test_audit_analysis_live.py` - Live audit endpoint integration tests
- `test_data_merger.py` - Tests for merging data from multiple sources
- `test_horde_api_integration.py` - Mocked Horde API integration tests
- `test_horde_api_integration_live.py` - Live Horde API integration tests
- `test_indexed_horde_types.py` - Tests for indexed Horde type mappings

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/statistics_and_audit/test_audit_analysis.py
```

### Run Tests by Directory

```bash
# Run all service API tests
pytest tests/service/

# Run all backend tests
pytest tests/backends/

# Run all statistics and audit tests
pytest tests/statistics_and_audit/

# Run all sync tests
pytest tests/sync/

# Run all Horde API integration tests
pytest tests/horde_api/
```

### Run Tests with Coverage

```bash
pytest --cov=horde_model_reference --cov-report=html
```

### Run Integration Tests Only

Integration tests are marked with `@pytest.mark.integration` and require network access to the live Horde API.

```bash
pytest -m integration
```

### Run Non-Integration Tests Only

```bash
pytest -m "not integration"
```


## Audit Analysis Tests

### Golden Models

Certain models are designated as "golden models" that should never be flagged for deletion regardless of usage statistics. These include:

**Image Generation:**

- `stable_diffusion`
- `stable_diffusion_2.1`
- `stable_diffusion_xl`
- `stable_diffusion_3`

**Text Generation:**

- _(To be defined)_

Golden model validation is tested in:

- `test_audit_analysis_live.py::TestAuditLiveIntegration::test_live_golden_models_consistency`

### Performance Thresholds

Audit endpoints must complete within **15 seconds** for full category analysis. This is validated in:

- `test_audit_analysis_live.py::TestAuditPerformance::test_audit_completes_within_threshold`
- `test_audit_analysis_live.py::TestAuditLiveIntegration::test_live_audit_performance`

## Test Fixtures

### Common Fixtures (tests/conftest.py)

- `api_client` - FastAPI TestClient for service endpoint tests
- `model_reference_manager` - Session-scoped ModelReferenceManager instance
- `restore_manager_singleton` - Reset manager singleton between tests
- `env_var_checks` - Validate test environment variables
- `base_path_for_tests` - Base directory for test data outputs
- `primary_base` - Isolated temporary directory for PRIMARY mode operations
- `legacy_path` - Legacy format directory path
- `dependency_override` - Register and cleanup FastAPI dependency overrides
- `primary_manager_override_factory` - Factory for creating PRIMARY mode managers
- `legacy_canonical_mode` - Switch to legacy canonical format for a test
- `v2_canonical_mode` - Switch to v2 canonical format for a test
- `v1_canonical_manager` - PRIMARY mode manager with legacy format for v1 API tests
- `caplog` - Capture log messages during tests (loguru-compatible)

## Environment Variables

Tests automatically set the following environment variables (via conftest.py):

- `TESTS_ONGOING=1` - Marks test environment
- `AI_HORDE_TESTING=True` - Enables test-specific isolation logic
- `HORDE_MODEL_REFERENCE_REPLICATE_MODE=PRIMARY` - Sets PRIMARY mode for tests
- `HORDE_MODEL_REFERENCE_CANONICAL_FORMAT=legacy` - Default format (v1 tests override to v2)

The following critical environment variables are automatically **cleared** before tests to ensure isolation:

- `HORDE_MODEL_REFERENCE_REDIS_USE_REDIS`
- `HORDE_MODEL_REFERENCE_REDIS_URL`
- `HORDE_MODEL_REFERENCE_PRIMARY_API_URL`
- `HORDE_MODEL_REFERENCE_CACHE_TTL_SECONDS`
- `HORDE_MODEL_REFERENCE_MAKE_FOLDERS`
- `HORDE_MODEL_REFERENCE_GITHUB_SEED_ENABLED`

## Test Data

- `tests/test_data_results/` - Test output directory (logs, generated files)
- 
## Writing New Tests

### Unit Tests

```python
import pytest
from horde_model_reference import SomeClass

def test_some_functionality():
    instance = SomeClass()
    result = instance.do_something()
    assert result == expected_value
```


### Integration Tests (Live)

```python
import pytest

@pytest.mark.integration
class TestLiveAPI:
    def test_live_endpoint(self, api_client):
        """This test hits the real API."""
        response = api_client.get("/v2/some/endpoint")
        assert response.status_code == 200
```

## Continuous Integration

In CI environments:

- Non-integration tests run on every commit
- Integration tests (marked with `@pytest.mark.integration`) may be run separately

## Troubleshooting

### Slow Tests

Check if you're accidentally running integration tests:

```bash
pytest -m "not integration"  # Exclude integration tests
```

### Import Errors

Ensure the service optional dependencies are installed:

```bash
pip install -e ".[service]"
```

### Redis Connection Errors

Tests automatically clear Redis-related environment variables. If you need to test Redis functionality specifically, check the `backends/test_redis_backend.py` tests which use `fakeredis` for mocking.

## Contributing

When adding new tests:

1. Follow existing patterns and naming conventions
2. Add docstrings explaining what the test validates
3. Mark live API tests with `@pytest.mark.integration`
4. Validate golden models are not flagged for deletion
5. Ensure tests complete within performance thresholds
6. Update this README if adding new test categories