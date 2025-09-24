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
    if args.volatility_aware:
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
