"""Public API for the independent baseline-anchored R4-V7 lineage."""

from .config import (
    ARCHITECTURE_VERSION,
    PHASE_ORDER,
    V7ConfigError,
    load_v7_config,
    validate_v7_config,
)
from .data import (
    FrozenResidualAnchor,
    V7DataError,
    issue_center_from_batch,
    load_frozen_residual_anchor,
)
from .distribution import (
    BaselineAnchoredConditionalTransport,
    BaselineAnchoredWorld,
    DistributionDecisionAdapter,
    StructuredDistributionMemory,
)
from .distribution_gate import (
    D0DistributionGateError,
    evaluate_d0_distribution_gate,
    gate_and_select_d0_distribution,
    require_d0_selected_checkpoint,
)
from .model import (
    BaselineAnchoredJointWorldPolicyMoEV7,
    R4V7Model,
    V7ModelConfig,
)
from .policy import (
    ActionConditionedRiskCritic,
    DecisionTop2MoE,
    JointWorldPolicy,
    OriginWorldReader,
    PhysicalActionLimits,
    SharedPlanDecoder,
    resolve_action_limits,
)

__all__ = [
    "ARCHITECTURE_VERSION",
    "ActionConditionedRiskCritic",
    "BaselineAnchoredConditionalTransport",
    "BaselineAnchoredJointWorldPolicyMoEV7",
    "BaselineAnchoredWorld",
    "DecisionTop2MoE",
    "DistributionDecisionAdapter",
    "D0DistributionGateError",
    "FrozenResidualAnchor",
    "JointWorldPolicy",
    "OriginWorldReader",
    "PHASE_ORDER",
    "PhysicalActionLimits",
    "R4V7Model",
    "SharedPlanDecoder",
    "StructuredDistributionMemory",
    "V7ConfigError",
    "V7DataError",
    "V7ModelConfig",
    "issue_center_from_batch",
    "evaluate_d0_distribution_gate",
    "gate_and_select_d0_distribution",
    "load_frozen_residual_anchor",
    "load_v7_config",
    "resolve_action_limits",
    "require_d0_selected_checkpoint",
    "validate_v7_config",
]
