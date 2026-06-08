"""Readable Monte Carlo tools for the classical two-dimensional Ising model."""

from .model import IsingModel
from .observables import summarize_samples

__all__ = ["IsingModel", "summarize_samples"]
