"""One-shot validation-only diagnostic for the R4 PUE residual head."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from dcvpp.e2e_moe_r4.capability_materialization import (
    LiveRepresentationCapabilityEvaluator,
)
from dcvpp.e2e_moe_r4.config import load_r4_config
from dcvpp.e2e_moe_r4.physical_twin import build_formal_r4_model
from dcvpp.e2e_moe_r4.training_data import PreloadedR4Corpus


class WorkloadDerivedPUE(nn.Module):
    """Diagnostic wrapper deriving PUE from forecast workload and frozen twin."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch):
        output = self.model(batch)
        forecasts = dict(output["forecasts"])
        workload = forecasts["workload"]
        core = self.model.physical_stream.physical_twin_core
        capacity = core.capacity_work_units.to(
            device=workload.device, dtype=workload.dtype
        )
        total_capacity = capacity.sum()
        served = workload.squeeze(-1).clamp(min=0.0, max=float(total_capacity))
        allocation = served[..., None] * capacity / total_capacity
        utilization = allocation / capacity
        physical_static = torch.as_tensor(
            batch["physical_static"], device=workload.device, dtype=workload.dtype
        )
        forecasts["pue"] = core(physical_static, utilization)["pue"].to(workload.dtype)
        return {**output, "forecasts": forecasts}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "configs"
        / "e2e_moe_dart_vpp_R4_hybrid_solver_free_formal.yaml"
    )
    source = (
        root
        / "artifacts"
        / "e2e_moe_hybrid_r4_v4_formal"
        / "seed=20240813"
        / "phase=representation"
        / "provisional_best"
        / "representation"
        / "seed=20240813"
        / "epoch0070.pt"
    )
    output = (
        root
        / "artifacts"
        / "run_control"
        / "r4_v4"
        / "pue_workload_twin_probe"
    )
    output.mkdir(parents=True, exist_ok=True)

    config = load_r4_config(config_path)
    corpus = PreloadedR4Corpus(config, root)
    model = build_formal_r4_model(config, root)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)

    diagnostic = dict(payload)
    diagnostic["model_state"] = model.state_dict()
    diagnostic["checkpoint_role"] = "validation_only_workload_derived_pue_counterfactual"
    diagnostic_path = output / "epoch0070_workload_derived_pue_counterfactual.pt"
    torch.save(diagnostic, diagnostic_path)

    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    wrapped = WorkloadDerivedPUE(model)
    evaluator = LiveRepresentationCapabilityEvaluator(
        corpus, output, seed=20240813
    )
    result = evaluator(
        phase="representation",
        horizon=96,
        checkpoint_path=str(diagnostic_path),
        epoch=70,
        model=wrapped,
    )
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
