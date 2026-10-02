from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

WORKSPACE = Path(__file__).resolve().parents[2]
TASK = WORKSPACE / "zero_shot_isone_20260928"
PROJECT = WORKSPACE / "ExecutableServiceCommitments_Revision_20260928"
ENGINE = PROJECT / "engine"
sys.path.insert(0, str(ENGINE))
from src.models.coupled_rollout import ArchitectedR2QPHead


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["before", "after"], required=True)
    args = parser.parse_args()
    config = json.loads((TASK / "configs/selected_w1.json").read_text(encoding="utf-8-sig"))
    cdf_selection_files = [
        TASK / "configs/cdf_selection_seed20260920.json",
        TASK / "configs/cdf_selection_seed20260921.json",
        TASK / "configs/cdf_selection_seed20260922.json",
    ]
    selections = [json.loads(p.read_text(encoding="utf-8-sig")) for p in cdf_selection_files]
    resources = [
        {
            "identifier": "W0 multiseed_20240829 formal_primary_best",
            "path": Path(config["parent"]["path"]),
            "recorded_sha256": config["parent"]["sha256"],
            "parameter_count": config.get("frozen_parameters", "NA"),
            "parameter_count_basis": "selected W1 metadata; same frozen architecture",
            "kind": "checkpoint",
        },
        {
            "identifier": "W1 multiseed_20240829 epoch80 formal_primary_best",
            "path": Path(config["checkpoint"]),
            "recorded_sha256": config["recorded_sha256"],
            "parameter_count": config.get("frozen_parameters", "NA"),
            "parameter_count_basis": "selected_w1.json frozen_parameters",
            "kind": "checkpoint",
        },
    ]
    for selection in selections:
        seed = int(selection["seed"])
        epoch = int(selection["refinement_epoch"])
        resources.append({
            "identifier": f"CdfR2 seed {seed} epoch {epoch}",
            "path": PROJECT / "engine/checkpoints" / f"CdfR2_seed{seed}_epoch{epoch:02d}.pt",
            "recorded_sha256": selection["forecast_checkpoint_sha256"],
            "parameter_count": "computed below",
            "parameter_count_basis": "state_dict tensor count; frozen inference module",
            "kind": "checkpoint",
            "selection": selection,
        })
    resources.extend([
        {
            "identifier": "NYISO frozen 57-feature normalizer",
            "path": TASK / "configs/frozen_nyiso_r2_normalization.json",
            "recorded_sha256": "NA",
            "parameter_count": "NA",
            "parameter_count_basis": "57 frozen feature fields",
            "kind": "normalizer",
        },
        {
            "identifier": "NYISO structured market cache 2025 seed20260920 epoch10",
            "path": PROJECT / "engine/cache/structured_market2025_s20260920_ep10.npz",
            "recorded_sha256": "NA",
            "parameter_count": "NA",
            "parameter_count_basis": "cache identifier only; ISO-NE inference will require external market features",
            "kind": "cache",
        },
    ])

    torch.set_grad_enabled(False)
    rows = []
    for item in resources:
        path = Path(item["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        params = item["parameter_count"]
        requires_grad_status = "not_loaded"
        if item["kind"] == "checkpoint" and item["identifier"].startswith("CdfR2"):
            selection = item["selection"]
            state = torch.load(path, map_location="cpu", weights_only=False)
            state_dict = state["state_dict"]
            model = ArchitectedR2QPHead()
            model.load_state_dict(state_dict)
            model.eval().requires_grad_(False)
            params = sum(parameter.numel() for parameter in model.parameters())
            requires_grad_status = str(all(not parameter.requires_grad for parameter in model.parameters())).lower()
            if requires_grad_status != "true":
                raise RuntimeError(f"requires_grad guard failed for {item['identifier']}")
            del model, state, state_dict
        elif item["kind"] == "checkpoint":
            requires_grad_status = "frozen_by_selection_metadata"
        actual = sha256(path)
        expected = str(item["recorded_sha256"])
        rows.append({
            "stage": args.stage,
            "identifier": item["identifier"],
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
            "recorded_sha256": expected,
            "recorded_hash_matches": ("true" if actual.lower() == expected.lower() else "false") if expected != "NA" else "not_recorded",
            "parameter_count": params,
            "parameter_count_basis": item["parameter_count_basis"],
            "requires_grad": requires_grad_status,
            "optimizer_created": "false",
            "backward_called": "false",
        })

    before_file = TASK / "logs/hash_before.csv"
    if args.stage == "after" and before_file.exists():
        with before_file.open("r", encoding="utf-8-sig", newline="") as handle:
            before = {r["identifier"]: r["sha256"] for r in csv.DictReader(handle)}
        for row in rows:
            row["matches_hash_before"] = str(before.get(row["identifier"]) == row["sha256"]).lower()
        if any(row["matches_hash_before"] != "true" for row in rows):
            raise RuntimeError("A frozen checkpoint/cache/scaler hash changed between audit stages.")
    csv_write(TASK / f"logs/hash_{args.stage}.csv", rows)
    if args.stage == "before":
        csv_write(before_file, rows)
    print(json.dumps({
        "stage": args.stage,
        "resources": len(rows),
        "all_recorded_hashes_match": all(r["recorded_hash_matches"].lower() in ("true", "not_recorded") for r in rows),
        "all_hashes_match_before": all(r.get("matches_hash_before", "true") == "true" for r in rows),
        "training": "not performed",
        "optimizer_created": False,
        "backward_called": False,
    }, indent=2))


if __name__ == "__main__":
    main()
