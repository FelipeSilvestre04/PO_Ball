"""
Stub module for optimization_result entities.
These are data classes used to store optimization results.
"""
from dataclasses import dataclass
from typing import Optional, Any
from enum import Enum
from datetime import datetime


class SolverStatus(Enum):
    """Status of the solver after optimization."""
    NOT_SOLVED = 0
    SOLVED_OPTIMAL = 1
    SOLVED_NOT_OPTIMAL = 2
    INFEASIBLE = 3


@dataclass
class ProductionItem:
    """Represents a production decision."""
    line: str
    plant: str
    product: str
    period_start: datetime
    period_end: datetime
    amount: float
    time_minutes: int


@dataclass
class InventoryItem:
    """Represents inventory level."""
    line: str
    plant: str
    product: str
    period_start: datetime
    period_end: datetime
    amount: float


@dataclass
class BacklogItem:
    """Represents backlog/unmet demand."""
    line: str
    plant: str
    product: str
    period_start: datetime
    period_end: datetime
    amount: float
    cost: float


@dataclass
class SetupItem:
    """Represents a setup transition."""
    line: str
    plant: str
    product_from: str
    product_to: str
    period_start: datetime
    period_end: datetime
    setup: bool
    time_minutes: int
    cost: float


@dataclass
class SetupCarryOverItem:
    """Represents a setup carry-over."""
    line: str
    plant: str
    product: str
    period_start: datetime
    period_end: datetime
    setup_carry_over: bool


@dataclass
class OptimizationResult:
    """Complete optimization result."""
    optz_run_id: Optional[str] = None
    time_s: Optional[float] = None
    objective_value: Optional[float] = None
    solver_status: Optional[Any] = None
    production: Optional[list] = None
    inventory: Optional[list] = None
    backlog: Optional[list] = None
    setup: Optional[list] = None
    setup_carry_over: Optional[list] = None
