"""
Forecasters package containing different forecasting bot implementations.
"""

from .self_critique import SelfCritiqueForecaster
from .ensemble import EnsembleForecaster

__all__ = ['SelfCritiqueForecaster', 'EnsembleForecaster']
