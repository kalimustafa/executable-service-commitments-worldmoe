from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd
import torch


TASK = Path(__file__).resolve().parents[1]
DATA = TASK / "data_processed"
WORKSPACE = TASK.parent
R2_ROOT = WORKSPACE / "R2_JointDecision_20260927"
ENGINE = WORKSPACE / "ExecutableServiceCommitments_Revision_20260928" / "engine"
sys.path.insert(0, str(R2_ROOT))
sys.path.insert(0, str(ENGINE))

from src.data.r2_data import ZONES as NYISO_ZONES  # noqa: E402
from src.models.factory import make_model  # noqa: E402
from src.models.structured_decision import StructuredDecision  # noqa: E402


PORTFOLIOS = {
    "ALL": [
        "CONNECTICUT", "NEMASSBOST", "SEMASS", "WCMASS",
        "MAINE", "NEWHAMPSHIRE", "VERMONT", "RHODEISLAND",
    ],
    "ISO-NE-SOUTH": ["CONNECTICUT", "NEMASSBOST", "SEMASS", "WCMASS"],
    "ISO-NE-NORTH": ["MAINE", "NEWHAMPSHIRE", "VERMONT", "RHODEISLAND"],
}
CYCLIC = {
    "P1": ("WEST", "CAPITL", "N.Y.C.", "LONGIL"),
    "P2": ("CAPITL", "N.Y.C.", "LONGIL", "WEST"),
    "P3": ("N.Y.C.", "LONGIL", "WEST", "CAPITL"),
    "P4": ("LONGIL", "WEST", "CAPITL", "N.Y.C."),
}
WORLD_FIELDS = [
    "w0_local", "w1_local", "w1_global", "expert_bank", "expert_prior",
    "tail_context", "tail_probability", "regime", "log_scale",
    "early", "middle", "late", "site",
]


class ExternalR2Data:
    def __init__(self, frame: pd.DataFrame, normalization: dict, embedding: str, portfolio: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.names = normalization["feature_names"]
        self.normalization = normalization
        zones = PORTFOLIOS[portfolio]
        frame = frame[frame["zone_code"].isin(zones)].copy()
        order = {zone: i for i, zone in enumerate(zones)}
        frame["site_order"] = frame["zone_code"].map(order)
        frame = frame.sort_values(["timestamp_utc", "site_order"], kind="stable").reset_index(drop=True)
        frame["zone"] = frame["zone_code"]
        self.frame = frame

        center = np.asarray(normalization["feature_center"], dtype=np.float32)
        scale = np.asarray(normalization["feature_scale"], dtype=np.float32)
        raw = frame[self.names].to_numpy(np.float32)
        finite = np.isfinite(raw)
        self.nonfinite_feature_count = int((~finite).sum())
        raw = np.where(finite, raw, center[None, :])
        x = np.clip((raw - center[None, :]) / scale[None, :], -float(normalization["feature_clip"]), float(normalization["feature_clip"]))
        self.numerical = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        neutral_center = np.asarray(normalization["target_center"], dtype=np.float32).mean(axis=0)
        neutral_scale = np.asarray(normalization["target_scale"], dtype=np.float32).mean(axis=0)
        self.target_center = torch.as_tensor(np.broadcast_to(neutral_center, (4, 3)).copy(), dtype=torch.float32, device=self.device)
        self.target_scale = torch.as_tensor(np.broadcast_to(neutral_scale, (4, 3)).copy(), dtype=torch.float32, device=self.device)
        self.historical_rt = torch.full((4,), float(neutral_center[2]), dtype=torch.float32, device=self.device)
        self.neutral_center = neutral_center
        self.neutral_scale = neutral_scale

        if embedding == "neutral_mean":
            self.zone = torch.zeros(len(frame), dtype=torch.long, device=self.device)
        else:
            assignment = CYCLIC[embedding]
            if portfolio == "ALL":
                raise ValueError("Cyclic embedding sensitivity is run by portfolio, not ALL.")
            by_site_order = {zone: NYISO_ZONES.index(old_zone) for zone, old_zone in zip(zones, assignment)}
            self.zone = torch.tensor(frame["zone_code"].map(by_site_order).to_numpy(), dtype=torch.long, device=self.device)

        labels = frame[["da_lbmp", "spread", "rt_integrated"]].to_numpy(np.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        self.lead = torch.as_tensor(frame["lead_hours"].to_numpy(np.float32), device=self.device)
        utc_midnight = pd.to_datetime(frame["date"], utc=True)
        self.utc_offset = torch.as_tensor((frame["timestamp_utc"] - utc_midnight).dt.total_seconds().to_numpy(np.float32) / 3600, device=self.device)

        grouped = frame.groupby(["date", "zone_code"], sort=False).indices
        profile_index = np.zeros((len(grouped), 25), dtype=np.int64)
        profile_mask = np.zeros((len(grouped), 25), dtype=bool)
        row_group = np.empty(len(frame), dtype=np.int64)
        row_position = np.empty(len(frame), dtype=np.int64)
        for group_id, indices in enumerate(grouped.values()):
            if not 1 <= len(indices) <= 25:
                raise ValueError("Unexpected daily profile length for ISO-NE local day.")
            profile_index[group_id, :len(indices)] = indices
            profile_mask[group_id, :len(indices)] = True
            row_group[indices] = group_id
            row_position[indices] = np.arange(len(indices))
        self.profile_index = torch.as_tensor(profile_index, device=self.device)
        self.profile_mask = torch.as_tensor(profile_mask, device=self.device)
        self.row_group = torch.as_tensor(row_group, device=self.device)
        self.row_position = torch.as_tensor(row_position, device=self.device)

        # ISO-NE has load-zone LMPs, not the NYISO W0/W1 40-entity packet.
        # Use the model's frozen explicit missing-world path with its mask off.
        self.world_index = torch.zeros(len(frame), dtype=torch.long, device=self.device)
        self.world_available = torch.zeros(len(frame), dtype=torch.float32, device=self.device)
        self.world = {
            "w0_local": torch.zeros((1, 4, 256), dtype=torch.float16, device=self.device),
            "w1_local": torch.zeros((1, 4, 256), dtype=torch.float16, device=self.device),
            "w1_global": torch.zeros((1, 256), dtype=torch.float16, device=self.device),
            "expert_bank": torch.zeros((1, 7, 256), dtype=torch.float16, device=self.device),
            "expert_prior": torch.zeros((1, 7), dtype=torch.float16, device=self.device),
            "tail_context": torch.zeros((1, 256), dtype=torch.float16, device=self.device),
            "tail_probability": torch.zeros((1,), dtype=torch.float16, device=self.device),
            "regime": torch.zeros((1, 8), dtype=torch.float16, device=self.device),
            "log_scale": torch.zeros((1, 32), dtype=torch.float16, device=self.device),
            "early": torch.zeros((1, 40, 256), dtype=torch.float16, device=self.device),
            "middle": torch.zeros((1, 40, 256), dtype=torch.float16, device=self.device),
            "late": torch.zeros((1, 40, 256), dtype=torch.float16, device=self.device),
            "site": torch.zeros((1, 4, 256), dtype=torch.float16, device=self.device),
        }

    def batch(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        indices = indices.to(self.device)
        world_rows = self.world_index[indices]
        zone = self.zone[indices]
        output = {
            key: (value[world_rows, zone] if key in {"w0_local", "w1_local"} or key.startswith("hourly_") else value[world_rows])
            for key, value in self.world.items()
        }
        output.update(
            numerical=self.numerical[indices],
            zone=zone,
            lead_hours=self.lead[indices],
            world_available=self.world_available[indices],
            target_utc_offset=self.utc_offset[indices],
        )
        groups = self.row_group[indices]
        profile = self.profile_index[groups]
        output.update(
            profile_numerical=self.numerical[profile],
            profile_da=self.labels[profile, 0],
            profile_mask=self.profile_mask[groups],
            profile_position=self.row_position[indices],
        )
        return output


def install_neutral_adapters(model: StructuredDecision, embedding: str) -> None:
    # Deterministic pooled NYISO target center/scale and native calibration.
    base = model.r2.base
    with torch.no_grad():
        center_names = ("da_center", "dart_center")
        scale_names = ("da_scale", "dart_scale")
        for name in center_names + scale_names:
            tensor = getattr(base, name)
            tensor.copy_(tensor.mean().expand_as(tensor))
        model.target_scales.copy_(model.target_scales.mean(dim=0, keepdim=True).expand_as(model.target_scales))
        model.historical_rt.copy_(model.historical_rt.mean().expand_as(model.historical_rt))

    calibration = model.calibration
    def neutral_calibration(self, da, dart, zone, gate, scales):
        batch_size = len(zone)
        da_center = self.da_shift.mean()
        da_width = self.da_width.mean().clamp(-2, 2).exp()
        da_scale = scales[:, 0].mean()
        calibrated_da = da[:, 3:4] + da_center * da_scale + da_width * (da - da[:, 3:4])
        gate_index = gate.long()
        shift_table = self.dart_shift.mean(dim=1)
        width_table = self.dart_width.mean(dim=1).clamp(-2, 2).exp()
        width_scale = scales[:, 1].mean()
        calibrated_dart = dart[:, 3:4] + shift_table[gate_index, None] * width_scale + width_table[gate_index, None] * (dart - dart[:, 3:4])
        return calibrated_da, calibrated_dart
    calibration.forward = MethodType(neutral_calibration, calibration)

    query = model.r2.market_query
    original_forward = query.forward
    def mapped_query(self, z_base, batch, publication, anchor_scaled):
        lead = self.lead(batch["lead_hours"].round().long().clamp(0, 96))
        pub = self.publication(publication.long())
        zone_embedding = (
            self.zone.weight.mean(dim=0).expand(len(z_base), -1)
            if embedding == "neutral_mean"
            else self.zone(batch["zone"])
        )
        projected = self.project(torch.cat([
            z_base, lead, zone_embedding, pub,
            publication[:, None], (publication * anchor_scaled)[:, None],
        ], -1))
        return projected, torch.cat([lead, pub], -1)
    query.forward = MethodType(mapped_query, query)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", choices=["neutral_mean", *CYCLIC], default="neutral_mean")
    parser.add_argument("--portfolio", choices=list(PORTFOLIOS), default="ALL")
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalization = json.loads((TASK / "configs" / "frozen_nyiso_r2_normalization.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(DATA / "isone_r2_features_2025.csv", parse_dates=["timestamp_utc", "issue_utc", "feature_last_interval_end_utc"], dtype={"zone_code": "string", "hour_ending_label": "string"})
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["issue_utc"] = pd.to_datetime(frame["issue_utc"], utc=True)
    frame["feature_last_interval_end_utc"] = pd.to_datetime(frame["feature_last_interval_end_utc"], utc=True)
    data = ExternalR2Data(frame, normalization, args.embedding, args.portfolio)

    checkpoint = R2_ROOT / "checkpoints" / "StructuredR2_seed20260920_best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base = make_model("C1_ProfileResidual", 20260920, data)
    model = StructuredDecision(base, data.target_scale, data.historical_rt, torch.ones(4, dtype=torch.float32, device=device)).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    install_neutral_adapters(model, args.embedding)

    times = pd.DatetimeIndex(pd.to_datetime(data.frame["timestamp_utc"], utc=True))
    hours = times.floor("h").unique()
    if not (hours == pd.date_range("2025-01-01 05:00", "2026-01-01 05:00", freq="h", inclusive="left", tz="UTC")).all():
        raise ValueError("Expected a continuous local-calendar 2025 UTC-hour sequence.")
    arrays = {
        "hours": hours.asi8,
        "zone_codes": np.asarray(PORTFOLIOS[args.portfolio], dtype="U32"),
        "pue_reference": np.ones(len(PORTFOLIOS[args.portfolio]), dtype=np.float32),
        "historical_rt": np.full(len(PORTFOLIOS[args.portfolio]), float(data.neutral_center[2]), dtype=np.float32),
    }
    labels = data.labels
    tau = torch.tensor([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=torch.float32, device=device)
    score_rows = []
    with torch.inference_mode():
        for gate_value in (0, 1):
            forecast_parts = []
            latent_parts = []
            losses = []
            covered = []
            for indices in torch.arange(len(data.frame), device=device).split(1024):
                batch = data.batch(indices)
                gate = torch.full((len(indices),), float(gate_value), dtype=torch.float32, device=device)
                anchor = labels[indices, 0] if gate_value else torch.zeros(len(indices), dtype=torch.float32, device=device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    output = model(batch, gate, anchor)
                rt = output["rt"].float()
                target = labels[indices, 2].float()
                error = target[:, None] - rt
                losses.append(torch.maximum(tau * error, (tau - 1) * error).mean(-1).cpu().numpy())
                covered.append(((target >= rt[:, 0]) & (target <= rt[:, -1])).float().cpu().numpy())
                forecast_parts.append(rt.cpu().numpy())
                latent_parts.append(output["decision_latent"].float().cpu().numpy())
            forecasts = np.concatenate(forecast_parts).reshape(len(hours), len(PORTFOLIOS[args.portfolio]), 7)
            latents = np.concatenate(latent_parts).reshape(len(hours), len(PORTFOLIOS[args.portfolio]), 128)
            arrays[f"rt_{gate_value}"] = forecasts
            arrays[f"latent_{gate_value}"] = latents
            score_rows.append({
                "market": "ISO-NE", "portfolio": args.portfolio, "embedding_rule": args.embedding,
                "gate": "post_publication" if gate_value else "pre_publication",
                "forecast_pinball_rt_usd_mwh": float(np.concatenate(losses).mean()),
                "coverage90_pct": 100.0 * float(np.concatenate(covered).mean()),
                "rows": int(len(data.frame)), "dates": 365,
            })

    output_path = DATA / f"FORECAST_CACHE_{args.portfolio}_{args.embedding}.npz"
    np.savez_compressed(output_path, **arrays)
    meta = {
        "status": "FROZEN_INFERENCE_COMPLETE",
        "market": "ISO-NE",
        "portfolio": args.portfolio,
        "embedding_rule": args.embedding,
        "base_checkpoint": str(checkpoint),
        "world_context": "zero with explicit world_available=0 frozen fallback; no unsupported NYISO node features are mapped into ISO-NE load zones",
        "external_scaler_fit": False,
        "external_calibration_fit": False,
        "optimizer_created": False,
        "backward_called": False,
        "device": str(device),
        "scores": score_rows,
    }
    (DATA / f"FORECAST_CACHE_{args.portfolio}_{args.embedding}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    score_path = TASK / "outputs" / "market_only" / "forecast_metrics.csv"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(score_rows).to_csv(score_path, mode="a", header=not score_path.exists(), index=False)
    print({"cache": str(output_path), "rows": len(data.frame), "hours": len(hours), "scores": score_rows, "frozen_inference": True})


if __name__ == "__main__":
    main()
