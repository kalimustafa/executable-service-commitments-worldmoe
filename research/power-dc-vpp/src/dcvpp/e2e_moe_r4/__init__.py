"""Solver-free Hybrid-CfC-Transformer-MoE R4 public API."""

from .liquid import ClosedFormLiquidCell, ClosedFormLiquidSequence, LiquidState
from .model import (
    ARCHITECTURE_VERSION,
    HybridCfCTransformerMoER4,
    R4Config,
    R4HybridModel,
)
from .policy import LiquidRollingPolicy
from .post_shield_particle_replay import (
    PostShieldParticleReplay,
    PostShieldParticleReplayBinding,
    PostShieldParticleReplayError,
    replay_post_shield_particle_risk,
)
from .physical_twin import (
    ArtifactBackedFrozenPhysicalTwinCore,
    PhysicalTwinArtifactError,
    build_formal_r4_model,
    load_frozen_physical_twin_core,
)
from .scenario_risk import (
    EquationBasedRisk,
    LearnedJointScenarioGenerator,
    SCENARIO_COUNT,
    exact_weighted_cvar,
)

__all__ = [
    "ARCHITECTURE_VERSION",
    "ArtifactBackedFrozenPhysicalTwinCore",
    "ClosedFormLiquidCell",
    "ClosedFormLiquidSequence",
    "EquationBasedRisk",
    "HybridCfCTransformerMoER4",
    "LearnedJointScenarioGenerator",
    "LiquidRollingPolicy",
    "LiquidState",
    "PhysicalTwinArtifactError",
    "PostShieldParticleReplay",
    "PostShieldParticleReplayBinding",
    "PostShieldParticleReplayError",
    "R4Config",
    "R4HybridModel",
    "SCENARIO_COUNT",
    "exact_weighted_cvar",
    "build_formal_r4_model",
    "load_frozen_physical_twin_core",
    "replay_post_shield_particle_risk",
]
