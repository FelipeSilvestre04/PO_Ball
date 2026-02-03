"""
Stub module for scenario entity.
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class Scenario:
    """Represents a planning scenario with all input data."""
    name: Optional[str] = None
    production_need: Optional[pd.DataFrame] = None
    production_rate: Optional[pd.DataFrame] = None
    setup_matrix: Optional[pd.DataFrame] = None
    initial_setup: Optional[str] = None
    line: Optional[str] = None
