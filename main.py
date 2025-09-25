import argparse
import asyncio
import logging
from pathlib import Path
from typing import Literal

from forecasting_tools import MetaculusApi
from config.loader import load_bot_config, default_config_path
from forecasters import EnsembleForecaster
from forecasters.bias_aware_ensemble import BiasAwareEnsembleForecaster
from forecasters.contradiction_aware_ensemble import ContradictionAwareEnsembleForecaster
from forecasters.volatility_aware_ensemble import VolatilityAwareEnsembleForecaster
from forecasters.configurable_ensemble import (
    ConfigurableEnsembleForecaster,
    ForecasterConfig,
    create_configurable_ensemble_forecaster,
    create_configurable_bias_aware_forecaster,
    create_configurable_contradiction_aware_forecaster,
    create_configurable_volatility_aware_forecaster,
    create_custom_forecaster
)

logger = logging.getLogger(__name__)


def create_ensemble_forecaster(config_path: str | Path | None = None) -> EnsembleForecaster:
    """
    Create an EnsembleForecaster instance with configuration loaded from YAML.
    
    This is the centralized factory function that ensures consistent configuration
    loading across all entry points. It provides a single source of truth for
    bot configuration and prevents misconfiguration errors.
    
    Args:
        config_path: Path to the YAML configuration file. If None, uses the default path.
        
    Returns:
        Configured EnsembleForecaster instance ready for use.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If the configuration is invalid.
    """
    if config_path is None:
        config_path = str(default_config_path())
    
    logger.info(f"Loading bot configuration from: {config_path}")
    
    # Load bot configuration and llms from YAML
    bot_cfg, llms = load_bot_config(config_path)
    
    # Log key configuration values for verification
    logger.info(f"Configuration loaded successfully:")
    logger.info(f"  - research_reports_per_question: {bot_cfg.get('research_reports_per_question')}")
    logger.info(f"  - publish_reports_to_metaculus: {bot_cfg.get('publish_reports_to_metaculus')}")
    logger.info(f"  - skip_previously_forecasted_questions: {bot_cfg.get('skip_previously_forecasted_questions')}")
    logger.info(f"  - folder_to_save_reports_to: {bot_cfg.get('folder_to_save_reports_to')}")
    logger.info(f"  - Number of LLMs configured: {len(llms)}")
    
    # Validate that essential configuration is present
    if 'research_reports_per_question' not in bot_cfg:
        raise ValueError("Missing required configuration: research_reports_per_question")
    
    # Construct the forecaster from externalized configuration
    forecaster = EnsembleForecaster(llms=llms, **bot_cfg)
    
    logger.info("EnsembleForecaster created successfully with loaded configuration")
    return forecaster


def create_bias_aware_ensemble_forecaster(config_path: str | Path | None = None) -> BiasAwareEnsembleForecaster:
    """
    Create a BiasAwareEnsembleForecaster instance with cognitive bias self-correction.
    
    This enhanced forecaster includes systematic cognitive bias detection and correction
    to improve forecast accuracy and logical soundness.
    
    Args:
        config_path: Path to the YAML configuration file. If None, uses the default path.
        
    Returns:
        Configured BiasAwareEnsembleForecaster instance ready for use.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If the configuration is invalid.
    """
    if config_path is None:
        config_path = str(default_config_path())
    
    logger.info(f"Loading bias-aware bot configuration from: {config_path}")
    
    # Load bot configuration and llms from YAML
    bot_cfg, llms = load_bot_config(config_path)
    
    # Log key configuration values for verification
    logger.info(f"Bias-aware configuration loaded successfully:")
    logger.info(f"  - research_reports_per_question: {bot_cfg.get('research_reports_per_question')}")
    logger.info(f"  - publish_reports_to_metaculus: {bot_cfg.get('publish_reports_to_metaculus')}")
    logger.info(f"  - skip_previously_forecasted_questions: {bot_cfg.get('skip_previously_forecasted_questions')}")
    logger.info(f"  - folder_to_save_reports_to: {bot_cfg.get('folder_to_save_reports_to')}")
    logger.info(f"  - Number of LLMs configured: {len(llms)}")
    
    # Validate that essential configuration is present
    if 'research_reports_per_question' not in bot_cfg:
        raise ValueError("Missing required configuration: research_reports_per_question")
    
    # Construct the bias-aware forecaster from externalized configuration
    forecaster = BiasAwareEnsembleForecaster(llms=llms, **bot_cfg)
    
    logger.info("BiasAwareEnsembleForecaster created successfully with loaded configuration")
    return forecaster


def create_contradiction_aware_ensemble_forecaster(config_path: str | Path | None = None) -> ContradictionAwareEnsembleForecaster:
    """
    Create a ContradictionAwareEnsembleForecaster instance with comprehensive analysis.
    
    This enhanced forecaster includes systematic cognitive bias detection, contradiction
    resolution, and uncertainty identification to improve forecast accuracy and robustness.
    
    Args:
        config_path: Path to the YAML configuration file. If None, uses the default path.
        
    Returns:
        Configured ContradictionAwareEnsembleForecaster instance ready for use.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If the configuration is invalid.
    """
    if config_path is None:
        config_path = str(default_config_path())
    
    logger.info(f"Loading contradiction-aware bot configuration from: {config_path}")
    
    # Load bot configuration and llms from YAML
    bot_cfg, llms = load_bot_config(config_path)
    
    # Log key configuration values for verification
    logger.info(f"Contradiction-aware configuration loaded successfully:")
    logger.info(f"  - research_reports_per_question: {bot_cfg.get('research_reports_per_question')}")
    logger.info(f"  - publish_reports_to_metaculus: {bot_cfg.get('publish_reports_to_metaculus')}")
    logger.info(f"  - skip_previously_forecasted_questions: {bot_cfg.get('skip_previously_forecasted_questions')}")
    logger.info(f"  - folder_to_save_reports_to: {bot_cfg.get('folder_to_save_reports_to')}")
    logger.info(f"  - Number of LLMs configured: {len(llms)}")
    
    # Validate that essential configuration is present
    if 'research_reports_per_question' not in bot_cfg:
        raise ValueError("Missing required configuration: research_reports_per_question")
    
    # Construct the contradiction-aware forecaster from externalized configuration
    forecaster = ContradictionAwareEnsembleForecaster(llms=llms, **bot_cfg)
    
    logger.info("ContradictionAwareEnsembleForecaster created successfully with loaded configuration")
    return forecaster


def create_volatility_aware_ensemble_forecaster(config_path: str | Path | None = None) -> VolatilityAwareEnsembleForecaster:
    """
    Create a VolatilityAwareEnsembleForecaster instance with comprehensive volatility analysis.
    
    This enhanced forecaster includes systematic cognitive bias detection, contradiction
    resolution, uncertainty identification, and information environment volatility assessment
    to dynamically adjust prediction confidence based on news sentiment and volume.
    
    Args:
        config_path: Path to the YAML configuration file. If None, uses the default path.
        
    Returns:
        Configured VolatilityAwareEnsembleForecaster instance ready for use.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If the configuration is invalid.
    """
    if config_path is None:
        config_path = str(default_config_path())
    
    logger.info(f"Loading volatility-aware bot configuration from: {config_path}")
    
    # Load bot configuration and llms from YAML
    bot_cfg, llms = load_bot_config(config_path)
    
    # Log key configuration values for verification
    logger.info(f"Volatility-aware configuration loaded successfully:")
    logger.info(f"  - research_reports_per_question: {bot_cfg.get('research_reports_per_question')}")
    logger.info(f"  - publish_reports_to_metaculus: {bot_cfg.get('publish_reports_to_metaculus')}")
    logger.info(f"  - skip_previously_forecasted_questions: {bot_cfg.get('skip_previously_forecasted_questions')}")
    logger.info(f"  - folder_to_save_reports_to: {bot_cfg.get('folder_to_save_reports_to')}")
    logger.info(f"  - Number of LLMs configured: {len(llms)}")
    
    # Validate that essential configuration is present
    if 'research_reports_per_question' not in bot_cfg:
        raise ValueError("Missing required configuration: research_reports_per_question")
    
    # Construct the volatility-aware forecaster from externalized configuration
    forecaster = VolatilityAwareEnsembleForecaster(llms=llms, **bot_cfg)
    
    logger.info("VolatilityAwareEnsembleForecaster created successfully with loaded configuration")
    return forecaster


# ============================================================================
# NEW CONFIGURABLE FORECASTER FACTORY FUNCTIONS
# ============================================================================

def create_new_configurable_ensemble_forecaster(config_path: str | Path | None = None) -> ConfigurableEnsembleForecaster:
    """
    Create a ConfigurableEnsembleForecaster instance without enhancements (standard ensemble).
    
    This is the new unified forecaster that can be configured with any combination of features.
    This variant creates a standard ensemble forecaster equivalent to EnsembleForecaster.
    
    Args:
        config_path: Path to the YAML configuration file. If None, uses the default path.
        
    Returns:
        Configured ConfigurableEnsembleForecaster instance ready for use.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If the configuration is invalid.
    """
    if config_path is None:
        config_path = str(default_config_path())
    
    logger.info(f"Loading configurable ensemble bot configuration from: {config_path}")
    
    # Load bot configuration and llms from YAML
    bot_cfg, llms = load_bot_config(config_path)
    
    # Log key configuration values for verification
    logger.info(f"Configurable ensemble configuration loaded successfully:")
    logger.info(f"  - research_reports_per_question: {bot_cfg.get('research_reports_per_question')}")
    logger.info(f"  - publish_reports_to_metaculus: {bot_cfg.get('publish_reports_to_metaculus')}")
    logger.info(f"  - skip_previously_forecasted_questions: {bot_cfg.get('skip_previously_forecasted_questions')}")
    logger.info(f"  - folder_to_save_reports_to: {bot_cfg.get('folder_to_save_reports_to')}")
    logger.info(f"  - Number of LLMs configured: {len(llms)}")
    
    # Validate that essential configuration is present
    if 'research_reports_per_question' not in bot_cfg:
        raise ValueError("Missing required configuration: research_reports_per_question")
    
    # Create standard configuration (no enhancements)
    forecaster_config = ForecasterConfig()
    
    # Construct the configurable forecaster from externalized configuration
    forecaster = ConfigurableEnsembleForecaster(
        llms=llms, 
        forecaster_config=forecaster_config, 
        **bot_cfg
    )
    
    logger.info("ConfigurableEnsembleForecaster created successfully with standard configuration")
    return forecaster


def create_new_bias_aware_ensemble_forecaster(config_path: str | Path | None = None) -> ConfigurableEnsembleForecaster:
    """
    Create a ConfigurableEnsembleForecaster with bias analysis enabled.
    
    This creates a forecaster equivalent to BiasAwareEnsembleForecaster using the new 
    configurable approach.
    
    Args:
        config_path: Path to the YAML configuration file. If None, uses the default path.
        
    Returns:
        Configured ConfigurableEnsembleForecaster instance ready for use.
    """
    if config_path is None:
        config_path = str(default_config_path())
    
    logger.info(f"Loading configurable bias-aware bot configuration from: {config_path}")
    
    # Load bot configuration and llms from YAML
    bot_cfg, llms = load_bot_config(config_path)
    
    # Log key configuration values for verification
    logger.info(f"Configurable bias-aware configuration loaded successfully:")
    logger.info(f"  - research_reports_per_question: {bot_cfg.get('research_reports_per_question')}")
    logger.info(f"  - publish_reports_to_metaculus: {bot_cfg.get('publish_reports_to_metaculus')}")
    logger.info(f"  - skip_previously_forecasted_questions: {bot_cfg.get('skip_previously_forecasted_questions')}")
    logger.info(f"  - folder_to_save_reports_to: {bot_cfg.get('folder_to_save_reports_to')}")
    logger.info(f"  - Number of LLMs configured: {len(llms)}")
    
    # Validate that essential configuration is present
    if 'research_reports_per_question' not in bot_cfg:
        raise ValueError("Missing required configuration: research_reports_per_question")
    
    # Create bias-aware configuration
    forecaster_config = ForecasterConfig(enable_bias_analysis=True)
    
    # Construct the configurable forecaster from externalized configuration
    forecaster = ConfigurableEnsembleForecaster(
        llms=llms, 
        forecaster_config=forecaster_config, 
        **bot_cfg
    )
    
    logger.info("ConfigurableEnsembleForecaster created successfully with bias-aware configuration")
    return forecaster


def create_new_contradiction_aware_ensemble_forecaster(config_path: str | Path | None = None) -> ConfigurableEnsembleForecaster:
    """
    Create a ConfigurableEnsembleForecaster with bias and contradiction analysis enabled.
    
    This creates a forecaster equivalent to ContradictionAwareEnsembleForecaster using 
    the new configurable approach.
    
    Args:
        config_path: Path to the YAML configuration file. If None, uses the default path.
        
    Returns:
        Configured ConfigurableEnsembleForecaster instance ready for use.
    """
    if config_path is None:
        config_path = str(default_config_path())
    
    logger.info(f"Loading configurable contradiction-aware bot configuration from: {config_path}")
    
    # Load bot configuration and llms from YAML
    bot_cfg, llms = load_bot_config(config_path)
    
    # Log key configuration values for verification
    logger.info(f"Configurable contradiction-aware configuration loaded successfully:")
    logger.info(f"  - research_reports_per_question: {bot_cfg.get('research_reports_per_question')}")
    logger.info(f"  - publish_reports_to_metaculus: {bot_cfg.get('publish_reports_to_metaculus')}")
    logger.info(f"  - skip_previously_forecasted_questions: {bot_cfg.get('skip_previously_forecasted_questions')}")
    logger.info(f"  - folder_to_save_reports_to: {bot_cfg.get('folder_to_save_reports_to')}")
    logger.info(f"  - Number of LLMs configured: {len(llms)}")
    
    # Validate that essential configuration is present
    if 'research_reports_per_question' not in bot_cfg:
        raise ValueError("Missing required configuration: research_reports_per_question")
    
    # Create contradiction-aware configuration
    forecaster_config = ForecasterConfig(
        enable_bias_analysis=True,
        enable_contradiction_analysis=True
    )
    
    # Construct the configurable forecaster from externalized configuration
    forecaster = ConfigurableEnsembleForecaster(
        llms=llms, 
        forecaster_config=forecaster_config, 
        **bot_cfg
    )
    
    logger.info("ConfigurableEnsembleForecaster created successfully with contradiction-aware configuration")
    return forecaster


def create_new_volatility_aware_ensemble_forecaster(config_path: str | Path | None = None) -> ConfigurableEnsembleForecaster:
    """
    Create a ConfigurableEnsembleForecaster with all analysis enhancements enabled.
    
    This creates a forecaster equivalent to VolatilityAwareEnsembleForecaster using 
    the new configurable approach.
    
    Args:
        config_path: Path to the YAML configuration file. If None, uses the default path.
        
    Returns:
        Configured ConfigurableEnsembleForecaster instance ready for use.
    """
    if config_path is None:
        config_path = str(default_config_path())
    
    logger.info(f"Loading configurable volatility-aware bot configuration from: {config_path}")
    
    # Load bot configuration and llms from YAML
    bot_cfg, llms = load_bot_config(config_path)
    
    # Log key configuration values for verification
    logger.info(f"Configurable volatility-aware configuration loaded successfully:")
    logger.info(f"  - research_reports_per_question: {bot_cfg.get('research_reports_per_question')}")
    logger.info(f"  - publish_reports_to_metaculus: {bot_cfg.get('publish_reports_to_metaculus')}")
    logger.info(f"  - skip_previously_forecasted_questions: {bot_cfg.get('skip_previously_forecasted_questions')}")
    logger.info(f"  - folder_to_save_reports_to: {bot_cfg.get('folder_to_save_reports_to')}")
    logger.info(f"  - Number of LLMs configured: {len(llms)}")
    
    # Validate that essential configuration is present
    if 'research_reports_per_question' not in bot_cfg:
        raise ValueError("Missing required configuration: research_reports_per_question")
    
    # Create full volatility-aware configuration
    forecaster_config = ForecasterConfig(
        enable_bias_analysis=True,
        enable_contradiction_analysis=True,
        enable_volatility_analysis=True
    )
    
    # Construct the configurable forecaster from externalized configuration
    forecaster = ConfigurableEnsembleForecaster(
        llms=llms, 
        forecaster_config=forecaster_config, 
        **bot_cfg
    )
    
    logger.info("ConfigurableEnsembleForecaster created successfully with full volatility-aware configuration")
    return forecaster


def create_custom_configurable_forecaster(
    config_path: str | Path | None = None,
    enable_bias_analysis: bool = False,
    enable_contradiction_analysis: bool = False,
    enable_volatility_analysis: bool = False
) -> ConfigurableEnsembleForecaster:
    """
    Create a ConfigurableEnsembleForecaster with custom feature selection.
    
    This allows creating forecasters with any combination of features, providing
    maximum flexibility that wasn't possible with the inheritance approach.
    
    Args:
        config_path: Path to the YAML configuration file. If None, uses the default path.
        enable_bias_analysis: Whether to enable cognitive bias analysis
        enable_contradiction_analysis: Whether to enable contradiction detection
        enable_volatility_analysis: Whether to enable volatility assessment
        
    Returns:
        Configured ConfigurableEnsembleForecaster instance ready for use.
        
    Examples:
        # Volatility-only forecaster (impossible with inheritance)
        forecaster = create_custom_configurable_forecaster(
            enable_volatility_analysis=True
        )
        
        # Bias + volatility (skipping contradiction analysis)
        forecaster = create_custom_configurable_forecaster(
            enable_bias_analysis=True,
            enable_volatility_analysis=True
        )
    """
    if config_path is None:
        config_path = str(default_config_path())
    
    features = []
    if enable_bias_analysis:
        features.append("bias_analysis")
    if enable_contradiction_analysis:
        features.append("contradiction_analysis") 
    if enable_volatility_analysis:
        features.append("volatility_analysis")
    
    logger.info(f"Loading custom configurable bot configuration from: {config_path}")
    logger.info(f"Requested features: {', '.join(features) if features else 'none (standard ensemble)'}")
    
    # Load bot configuration and llms from YAML
    bot_cfg, llms = load_bot_config(config_path)
    
    # Log key configuration values for verification
    logger.info(f"Custom configurable configuration loaded successfully:")
    logger.info(f"  - research_reports_per_question: {bot_cfg.get('research_reports_per_question')}")
    logger.info(f"  - publish_reports_to_metaculus: {bot_cfg.get('publish_reports_to_metaculus')}")
    logger.info(f"  - skip_previously_forecasted_questions: {bot_cfg.get('skip_previously_forecasted_questions')}")
    logger.info(f"  - folder_to_save_reports_to: {bot_cfg.get('folder_to_save_reports_to')}")
    logger.info(f"  - Number of LLMs configured: {len(llms)}")
    
    # Validate that essential configuration is present
    if 'research_reports_per_question' not in bot_cfg:
        raise ValueError("Missing required configuration: research_reports_per_question")
    
    # Create custom configuration
    forecaster_config = ForecasterConfig(
        enable_bias_analysis=enable_bias_analysis,
        enable_contradiction_analysis=enable_contradiction_analysis,
        enable_volatility_analysis=enable_volatility_analysis
    )
    
    # Construct the configurable forecaster from externalized configuration
    forecaster = ConfigurableEnsembleForecaster(
        llms=llms, 
        forecaster_config=forecaster_config, 
        **bot_cfg
    )
    
    logger.info(f"ConfigurableEnsembleForecaster created successfully with custom configuration: {forecaster_config}")
    return forecaster


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the EnsembleForecaster forecasting system with optional cognitive bias self-correction"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions", "minibench"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config_path()),
        help="Path to YAML config file (default: config/bot_config.yaml)",
    )
    parser.add_argument(
        "--bias-aware",
        action="store_true",
        help="Use the bias-aware ensemble forecaster with cognitive bias self-correction (default: False)",
    )
    parser.add_argument(
        "--contradiction-aware",
        action="store_true", 
        help="Use the contradiction-aware ensemble forecaster with both bias correction and contradiction detection (default: False)",
    )
    parser.add_argument(
        "--volatility-aware",
        action="store_true",
        help="Use the volatility-aware ensemble forecaster with bias correction, contradiction detection, and volatility-adjusted confidence (default: False)",
    )
    parser.add_argument(
        "--new-configurable",
        action="store_true",
        help="Use the new ConfigurableEnsembleForecaster (standard ensemble mode) - demonstrates the unified architecture (default: False)",
    )
    parser.add_argument(
        "--new-bias-aware",
        action="store_true",
        help="Use the new ConfigurableEnsembleForecaster in bias-aware mode - equivalent to BiasAwareEnsembleForecaster (default: False)",
    )
    parser.add_argument(
        "--new-contradiction-aware",
        action="store_true",
        help="Use the new ConfigurableEnsembleForecaster in contradiction-aware mode - equivalent to ContradictionAwareEnsembleForecaster (default: False)",
    )
    parser.add_argument(
        "--new-volatility-aware",
        action="store_true",
        help="Use the new ConfigurableEnsembleForecaster in volatility-aware mode - equivalent to VolatilityAwareEnsembleForecaster (default: False)",
    )
    parser.add_argument(
        "--custom-features",
        action="store_true",
        help="Use the new ConfigurableEnsembleForecaster with custom feature selection (demonstrates flexibility impossible with inheritance) (default: False)",
    )
    parser.add_argument(
        "--enable-bias", 
        action="store_true",
        help="Enable bias analysis (only used with --custom-features) (default: False)"
    )
    parser.add_argument(
        "--enable-contradiction",
        action="store_true", 
        help="Enable contradiction analysis (only used with --custom-features) (default: False)"
    )
    parser.add_argument(
        "--enable-volatility",
        action="store_true",
        help="Enable volatility analysis (only used with --custom-features) (default: False)"
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions", "minibench"] = (
        args.mode
    )
    assert run_mode in [
        "tournament",
        "quarterly_cup",
        "test_questions",
        "minibench",
    ], "Invalid run mode"

    # Create forecaster using the centralized configuration factory
    if args.custom_features:
        logger.info(f"Creating custom configurable ensemble forecaster: "
                   f"bias={args.enable_bias}, contradiction={args.enable_contradiction}, volatility={args.enable_volatility}")
        bot_one = create_custom_configurable_forecaster(
            config_path=args.config,
            enable_bias_analysis=args.enable_bias,
            enable_contradiction_analysis=args.enable_contradiction,
            enable_volatility_analysis=args.enable_volatility
        )
    elif args.new_volatility_aware:
        logger.info("Creating NEW volatility-aware ensemble forecaster (ConfigurableEnsembleForecaster with all features)")
        bot_one = create_new_volatility_aware_ensemble_forecaster(args.config)
    elif args.new_contradiction_aware:
        logger.info("Creating NEW contradiction-aware ensemble forecaster (ConfigurableEnsembleForecaster with bias + contradiction)")
        bot_one = create_new_contradiction_aware_ensemble_forecaster(args.config)
    elif args.new_bias_aware:
        logger.info("Creating NEW bias-aware ensemble forecaster (ConfigurableEnsembleForecaster with bias only)")
        bot_one = create_new_bias_aware_ensemble_forecaster(args.config)
    elif args.new_configurable:
        logger.info("Creating NEW standard ensemble forecaster (ConfigurableEnsembleForecaster with no enhancements)")
        bot_one = create_new_configurable_ensemble_forecaster(args.config)
    elif args.volatility_aware:
        logger.info("Creating volatility-aware ensemble forecaster with bias correction, contradiction detection, and volatility adjustment")
        bot_one = create_volatility_aware_ensemble_forecaster(args.config)
    elif args.contradiction_aware:
        logger.info("Creating contradiction-aware ensemble forecaster with bias correction and contradiction detection")
        bot_one = create_contradiction_aware_ensemble_forecaster(args.config)
    elif args.bias_aware:
        logger.info("Creating bias-aware ensemble forecaster with cognitive bias self-correction")
        bot_one = create_bias_aware_ensemble_forecaster(args.config)
    else:
        logger.info("Creating standard ensemble forecaster")
        bot_one = create_ensemble_forecaster(args.config)

    if run_mode == "tournament":
        forecast_reports = asyncio.run(
            bot_one.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
    elif run_mode == "quarterly_cup":
        # The quarterly cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564
        # The new quarterly cup may not be initialized near the beginning of a quarter
        bot_one.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            bot_one.forecast_on_tournament(
                MetaculusApi.CURRENT_QUARTERLY_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029        ]
        ]
        bot_one.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            bot_one.forecast_questions(questions, return_exceptions=True)
        )
    elif run_mode == "minibench":
        forecast_reports = asyncio.run(
            bot_one.forecast_on_tournament(
                MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )

    EnsembleForecaster.log_report_summary(forecast_reports)
