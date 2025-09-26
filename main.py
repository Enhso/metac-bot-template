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
)

logger = logging.getLogger(__name__)


def create_forecaster(
    config_path: str | Path | None = None,
    bias_aware: bool = False,
    contradiction_aware: bool = False,
    volatility_aware: bool = False,
    use_legacy: bool = False
) -> EnsembleForecaster | ConfigurableEnsembleForecaster:
    """
    Create a forecaster instance with configuration loaded from YAML.
    
    This is the unified factory function that creates forecasters with the appropriate
    feature configuration based on command-line arguments. It centralizes forecaster
    creation and eliminates the need for multiple factory functions.
    
    Args:
        config_path: Path to the YAML configuration file. If None, uses the default path.
        bias_aware: Whether to enable cognitive bias analysis
        contradiction_aware: Whether to enable contradiction detection  
        volatility_aware: Whether to enable volatility assessment
        use_legacy: Whether to use legacy forecaster classes (for backward compatibility)
        
    Returns:
        Configured forecaster instance ready for use.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If the configuration is invalid.
    """
    if config_path is None:
        config_path = str(default_config_path())
    
    # Determine feature set based on arguments
    features: list[str] = []
    if bias_aware:
        features.append("bias_analysis")
    if contradiction_aware:
        features.append("contradiction_analysis") 
    if volatility_aware:
        features.append("volatility_analysis")
    
    logger.info(f"Loading bot configuration from: {config_path}")
    logger.info(f"Requested features: {', '.join(features) if features else 'none (standard ensemble)'}")
    
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
    
    # Create appropriate forecaster based on configuration
    if use_legacy:
        # Use legacy classes for backward compatibility
        if volatility_aware:
            forecaster = VolatilityAwareEnsembleForecaster(llms=llms, **bot_cfg)
            logger.info("VolatilityAwareEnsembleForecaster created successfully")
        elif contradiction_aware:
            forecaster = ContradictionAwareEnsembleForecaster(llms=llms, **bot_cfg)
            logger.info("ContradictionAwareEnsembleForecaster created successfully")
        elif bias_aware:
            forecaster = BiasAwareEnsembleForecaster(llms=llms, **bot_cfg)
            logger.info("BiasAwareEnsembleForecaster created successfully")
        else:
            forecaster = EnsembleForecaster(llms=llms, **bot_cfg)
            logger.info("EnsembleForecaster created successfully")
    else:
        # Use new ConfigurableEnsembleForecaster
        forecaster_config = ForecasterConfig(
            enable_bias_analysis=bias_aware,
            enable_contradiction_analysis=contradiction_aware,
            enable_volatility_analysis=volatility_aware
        )
        
        forecaster = ConfigurableEnsembleForecaster(
            llms=llms, 
            forecaster_config=forecaster_config, 
            **bot_cfg
        )
        logger.info(f"ConfigurableEnsembleForecaster created successfully with configuration: {forecaster_config}")
    
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
        help="Enable cognitive bias analysis (default: False)",
    )
    parser.add_argument(
        "--contradiction-aware",
        action="store_true", 
        help="Enable contradiction detection (default: False)",
    )
    parser.add_argument(
        "--volatility-aware",
        action="store_true",
        help="Enable volatility-adjusted confidence (default: False)",
    )
    parser.add_argument(
        "--use-legacy",
        action="store_true",
        help="Use legacy forecaster classes instead of ConfigurableEnsembleForecaster (default: False)",
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

    # Create forecaster using the unified factory function
    bot_one = create_forecaster(
        config_path=args.config,
        bias_aware=args.bias_aware,
        contradiction_aware=args.contradiction_aware,
        volatility_aware=args.volatility_aware,
        use_legacy=args.use_legacy
    )

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
