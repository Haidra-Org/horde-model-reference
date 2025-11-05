# Horde Model Reference Examples

This directory contains working examples demonstrating various use cases for the horde-model-reference library.

## Running the Examples

All examples can be run directly with Python after installing the library:

```bash
# Install the library
pip install horde-model-reference

# Run any example
python scenario_1_model_browser.py
python scenario_2_worker_integration.py
python scenario_3_model_downloader.py
python scenario_4_capability_checker.py
python scenario_5_type_safe_config.py
python scenario_6_multi_category.py
```

## Example Scenarios

### Scenario 1: Model Browser UI
**File:** `scenario_1_model_browser.py`

Demonstrates how to build a web UI for browsing AI models:
- Grouping models by baseline
- Filtering by style (anime, realistic, etc.)
- Filtering SFW/NSFW models
- Type-safe access to model metadata

### Scenario 2: AI-Horde Worker Integration
**File:** `scenario_2_worker_integration.py`

Shows how to integrate model references into an AI-Horde worker:
- Filtering models by GPU capabilities
- Checking bridge version requirements
- Managing local model files
- Identifying models to download

### Scenario 3: Model Downloader Tool
**File:** `scenario_3_model_downloader.py`

Demonstrates building a CLI tool for model management:
- Listing available models
- Displaying model information
- Accessing download URLs and checksums
- (Download functionality demonstrated but commented out)

### Scenario 4: Model Capability Checker
**File:** `scenario_4_capability_checker.py`

Shows how to check model capabilities:
- Finding models with inpainting support
- Checking model requirements
- Finding models by baseline
- Searching models by tags

### Scenario 5: Type-Safe Configuration
**File:** `scenario_5_type_safe_config.py`

Demonstrates using Pydantic for type-safe configuration:
- Validating model names at configuration time
- Using enums for type safety
- Filtering models based on configuration
- Catching configuration errors early

### Scenario 6: Multi-Category Explorer
**File:** `scenario_6_multi_category.py`

Shows working with multiple model categories:
- Summarizing all categories
- Accessing text generation models (LLMs)
- Working with ControlNet models
- Exploring utility models (CLIP, BLIP, upscalers)

### Logging Control Example
**File:** `scenario_logging_control.py`

Demonstrates controlling logging verbosity:
- Default quiet operation (WARNING and above)
- Enabling DEBUG logging for troubleshooting
- Using INFO level for visibility
- Completely silencing all logs
- Environment variable control

## Key Concepts Demonstrated

### Type Safety
All examples emphasize type-safe code using:
- Enum types (`MODEL_REFERENCE_CATEGORY`, `MODEL_STYLE`)
- Pydantic model types (`ImageGenerationModelRecord`, etc.)
- Type annotations for IDE support

### Idiomatic Python
Examples follow Python best practices:
- List and dict comprehensions
- Context managers where appropriate
- Type hints throughout
- Clear separation of concerns

### Error Handling
Proper error handling patterns:
- Using try/except for model lookups
- Checking for None values
- Validating configuration at startup

### Performance
Efficiency patterns demonstrated:
- Caching manager instances
- Reusing model references
- Lazy loading when appropriate

### Logging Control
By default, the library is quiet (WARNING level only):
- Use `enable_debug_logging()` for troubleshooting
- Use `configure_logger("INFO")` for visibility
- Use `disable_logging()` for complete silence
- Or set `HORDE_MODEL_REFERENCE_LOG_LEVEL` environment variable

See `scenario_logging_control.py` for detailed examples.

## Testing

All examples have been tested and verified to work as-is. You can run them in order to understand progressively more complex use cases.

## Need Help?

- **Documentation:** See the [full documentation](https://haidra-org.github.io/horde-model-reference/)
- **Onboarding Guide:** Check out [docs/onboarding.md](../docs/onboarding.md)
- **Discord:** Join [AI Horde Discord](https://discord.gg/3DxrhksKzn)
- **Issues:** Report at [GitHub Issues](https://github.com/Haidra-Org/horde-model-reference/issues)
