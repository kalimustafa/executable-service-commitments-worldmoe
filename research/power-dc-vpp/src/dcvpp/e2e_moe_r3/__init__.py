"""Decision-native solver-free E2E-MoE-DART-VPP R3 package.

The neural policy is executed through an analytic constraint shield and an
independent validator.  Importing this package does not import Pyomo, Gurobi,
or another mathematical-programming backend.
"""

from importlib import import_module

from .losses import (
    E2ELoss,
    LossWeights,
    cvar_loss,
    differentiable_realized_settlement_cost,
    energy_score_loss,
    feasibility_loss,
    marginal_scenario_loss,
    migration_cost_loss,
    routing_balance_loss,
    routing_stability_loss,
    scenario_diversity_loss,
    scenario_weight_entropy_loss,
    sla_violation_loss,
    tail_coverage_loss,
    teacher_imitation_loss,
    variogram_score_loss,
)
from .model import (
    E2EMoEDARTVPP,
    E2EMoEModel,
    E2EMoER3,
    E2EMoER3Config,
    FrozenPhysicalTwinCore,
    SparseTop2MoE,
)
from .policy import (
    FeasibilityFallbackHead,
    NeuralRollingPolicy,
    should_full_optimize,
    should_use_safe_policy,
)
from .scenario_risk import (
    SCENARIO_COUNT,
    SCENARIO_VARIABLES,
    DecisionRiskHead,
    LearnedJointScenarioHead,
    ScenarioRiskHead,
    differentiable_weighted_cvar,
)

__all__ = [
    "SCENARIO_COUNT",
    "SCENARIO_VARIABLES",
    "CertifiedConstraintShield",
    "DecisionRiskHead",
    "E2ELoss",
    "E2EMoEDARTVPP",
    "E2EMoEModel",
    "E2EMoER3",
    "E2EMoER3Config",
    "FeasibilityFallbackHead",
    "FrozenPhysicalTwinCore",
    "InfeasibleStateError",
    "LearnedJointScenarioHead",
    "LossWeights",
    "NeuralRollingPolicy",
    "NeuralConstrainedRuntime",
    "R3DataError",
    "SelectionDay",
    "ScenarioRiskHead",
    "SparseTop2MoE",
    "cvar_loss",
    "differentiable_realized_settlement_cost",
    "differentiable_weighted_cvar",
    "energy_score_loss",
    "feasibility_loss",
    "marginal_scenario_loss",
    "migration_cost_loss",
    "load_frozen_physical",
    "load_selection_day",
    "records_frame",
    "routing_balance_loss",
    "routing_stability_loss",
    "scenario_diversity_loss",
    "scenario_weight_entropy_loss",
    "should_full_optimize",
    "should_use_safe_policy",
    "sla_violation_loss",
    "tail_coverage_loss",
    "teacher_imitation_loss",
    "variogram_score_loss",
    "validate_executed_action",
    "validate_record",
    "weighted_cvar",
]

__version__ = "0.1.0"


# Keep runtime exports lazy.  The reusable legacy ``research_branch`` package
# currently has optimization-oriented import-time side effects; importing the
# solver-free model package must not activate those optional dependencies.
_RUNTIME_EXPORTS = frozenset(
    {
        "CertifiedConstraintShield",
        "InfeasibleStateError",
        "NeuralConstrainedRuntime",
        "R3DataError",
        "SelectionDay",
        "load_frozen_physical",
        "load_selection_day",
        "records_frame",
        "validate_executed_action",
        "validate_record",
        "weighted_cvar",
    }
)


def __getattr__(name: str):
    """Lazily resolve solver-free runtime APIs without import-time solvers."""

    if name not in _RUNTIME_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    runtime = import_module(f"{__name__}.constrained_runtime")
    value = getattr(runtime, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _RUNTIME_EXPORTS)
