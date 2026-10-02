"""Isolated paper experiment branch for scenarios, optimization, and H0-H6.

The package has no import-time connection to the paper CLI.  The stable handoff
surface for the main pipeline is :func:`run_optimization_branch`.
"""

from dcvpp.research_branch.runner import run_optimization_branch

__all__ = ["run_optimization_branch"]
