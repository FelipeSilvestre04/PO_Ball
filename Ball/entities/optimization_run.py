"""
Stub module for optimization_run entity.
"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class OptzRun:
    """Represents an optimization run configuration."""
    id: Optional[str] = None
    name: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: Optional[str] = None
    # Required for line_scheduling_v1
    setup_cost: float = 1.0
    min_setup_time: float = 30.0
    max_setup_time: float = 120.0
    planning_slot_size: int = 12

