from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dcvpp.core import atomic_json, load_yaml, sha256_file, sha256_json, utc_now


def deploy_demo_twins(root: Path, calibration_manifest: Path, artifact_root: Path) -> dict[str, Any]:
    sites = load_yaml(root / "configs/sites.yaml")["sites"]
    calibration_hash = sha256_file(calibration_manifest)
    rows = []
    for twin_id, (site_id, site) in enumerate(sites.items(), start=1):
        rows.append(
            {
                "site": site_id,
                "zone": site["zone"],
                "city": site["city"],
                "latitude": site["latitude"],
                "longitude": site["longitude"],
                "observed_facility": False,
                "counterfactual": True,
                "calibration_id": calibration_hash[:16],
                "twin_id": f"NYISO-TWIN-{twin_id}",
                "capacity_scale": 1.0,
                "transfer_uncertainty_regime": "demo_central",
                "evidence_role": "counterfactual",
            }
        )
    table = artifact_root / "tables/digital_twin_transfer.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(table, index=False)
    manifest = {
        "created_at": utc_now(),
        "profile": "demo",
        "physical_calibration_sha256": calibration_hash,
        "observed_nyiso_facilities": False,
        "counterfactual": True,
        "twins": rows,
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    path = artifact_root / "manifests/digital_twin_transfer_manifest.json"
    atomic_json(path, manifest)
    return {"manifest": path, "table": table, "twins": rows}

