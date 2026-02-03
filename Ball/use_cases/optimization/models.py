"""
Stub module for optimization models base class.
"""
from abc import ABC, abstractmethod


class Model(ABC):
    """Abstract base class for optimization models."""
    
    @abstractmethod
    def optimize(self):
        """Run the optimization."""
        pass
    
    @abstractmethod
    def build_results(self):
        """Build and return results."""
        pass
