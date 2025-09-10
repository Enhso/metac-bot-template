# Configuration Management

This document describes the centralized configuration management pattern implemented in this project.

## Overview

The project uses a centralized configuration loading system that ensures consistent setup across all entry points and prevents misconfiguration errors. All configuration is stored in `config/bot_config.yaml` and loaded through a single factory function.

## Key Components

### 1. Configuration Factory Function

The `create_ensemble_forecaster()` function in `main.py` serves as the centralized factory for creating properly configured `EnsembleForecaster` instances:

```python
def create_ensemble_forecaster(config_path: str | Path | None = None) -> EnsembleForecaster:
    """
    Create an EnsembleForecaster instance with configuration loaded from YAML.
    
    This is the centralized factory function that ensures consistent configuration
    loading across all entry points. It provides a single source of truth for
    bot configuration and prevents misconfiguration errors.
    """
```

### 2. Configuration File

The `config/bot_config.yaml` file contains all bot and LLM configurations in a structured format:

```yaml
bot:
  research_reports_per_question: 3
  publish_reports_to_metaculus: true
  skip_previously_forecasted_questions: true
  folder_to_save_reports_to: "./forecast_reports"

llms:
  default:
    type: general
    model: "openrouter/openai/gpt-5"
    temperature: 0.3
    # ... additional LLM configurations
```

### 3. Configuration Loader

The `config/loader.py` module provides the `load_bot_config()` function that parses the YAML and returns structured configuration data.

## Usage Pattern

### Recommended Usage

All entry points should use the centralized factory:

```python
from main import create_ensemble_forecaster

# Create forecaster with default config
forecaster = create_ensemble_forecaster()

# Or with custom config path
forecaster = create_ensemble_forecaster("/path/to/custom/config.yaml")
```

### Entry Points Using This Pattern

1. **main.py** - Primary execution entry point
2. **community_benchmark.py** - Benchmarking script

Both scripts now use the same `create_ensemble_forecaster()` function to ensure consistency.

## Benefits

1. **Single Source of Truth**: All configuration comes from one YAML file
2. **Consistency**: All entry points use the same configuration loading logic
3. **Validation**: The factory function validates essential configuration
4. **Logging**: Configuration loading is logged for debugging
5. **Error Prevention**: Reduces risk of misconfiguration across different scripts

## Configuration Validation

The factory function validates that essential configuration is present:

```python
if 'research_reports_per_question' not in bot_cfg:
    raise ValueError("Missing required configuration: research_reports_per_question")
```

## Logging

Configuration loading is extensively logged to help with debugging:

```
INFO - Loading bot configuration from: /path/to/config.yaml
INFO - Configuration loaded successfully:
INFO -   - research_reports_per_question: 3
INFO -   - publish_reports_to_metaculus: True
INFO -   - skip_previously_forecasted_questions: True
INFO -   - folder_to_save_reports_to: ./forecast_reports
INFO -   - Number of LLMs configured: 8
INFO - EnsembleForecaster created successfully with loaded configuration
```

## Migration from Old Pattern

### Before (Manual Configuration Loading)
```python
bot_cfg, llms = load_bot_config(args.config)
bot_one = EnsembleForecaster(llms=llms, **bot_cfg)
```

### After (Centralized Factory)
```python
bot_one = create_ensemble_forecaster(args.config)
```

This change eliminates the need to manually handle configuration loading in each entry point, reducing code duplication and potential errors.
