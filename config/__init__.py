"""Configuration package for Semantic SEO Analyzer.

This package contains modules for managing application configuration,
including API settings and analysis parameters.
"""

# Import configuration classes
from config.api_config import APIConfig

# Export the main classes
__all__ = ['APIConfig']

# Default configuration instance that can be imported directly
default_config = APIConfig()
