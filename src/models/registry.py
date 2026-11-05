"""TURBINE-X Legacy Model Registry shim.
Redirects to src.models.model_registry for comprehensive benchmarking.
"""
from src.models.model_registry import model_registry, get_benchmark_models

__all__ = ["model_registry", "get_benchmark_models"]
