from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Callable

from dcvpp.core import (
    PipelineState,
    StageRecord,
    atomic_json,
    code_tree_hash,
    environment_snapshot,
    load_yaml,
    sha256_file,
    sha256_json,
    utc_now,
)
from dcvpp.data.gates import validate_demo_domains
from dcvpp.data.synthetic import generate_demo_domains
from dcvpp.forecasting import run_demo_energy, run_demo_workload
from dcvpp.optimization import run_demo_policies
from dcvpp.physical import calibrate_demo_twin
from dcvpp.reporting import build_demo_report, build_run_manifest
from dcvpp.scenarios import build_demo_scenarios
from dcvpp.transfer import deploy_demo_twins


class PaperBlocked(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _relative(root: Path, paths: list[Path]) -> list[str]:
    return [path.relative_to(root).as_posix() for path in paths]


class Orchestrator:
    def __init__(self, root: Path, config_path: Path, resume: bool = True) -> None:
        self.root = root.resolve()
        self.config_path = config_path.resolve()
        self.config = load_yaml(self.config_path)
        self.resume = resume
        self.state = PipelineState(self.root)
        self.config_hash = sha256_file(self.config_path)

    def _fingerprint(self, name: str, inputs: list[Path] | None = None) -> str:
        return sha256_json(
            {
                "stage": name,
                "config_hash": self.config_hash,
                "code_hash": code_tree_hash(self.root),
                "inputs": {str(path): sha256_file(path) for path in (inputs or []) if path.exists()},
            }
        )

    def stage(self, name: str, outputs: list[Path], function: Callable[[], Any], inputs: list[Path] | None = None) -> Any:
        fingerprint = self._fingerprint(name, inputs)
        if self.resume and self.state.completed(name, fingerprint):
            return {"skipped": True, "outputs": outputs}
        record = StageRecord(name=name, status="running", fingerprint=fingerprint, started_at=utc_now(), outputs=_relative(self.root, outputs))
        self.state.set(record)
        try:
            result = function()
            missing = [str(path) for path in outputs if not path.exists()]
            if missing:
                raise RuntimeError(f"stage did not create required outputs: {missing}")
            record.status = "completed"
            record.finished_at = utc_now()
            self.state.set(record)
            return result
        except Exception as exc:
            record.status = "failed"
            record.finished_at = utc_now()
            record.message = f"{type(exc).__name__}: {exc}"
            self.state.set(record)
            failure_path = self.root / "artifacts/tables/failures.csv"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            header = not failure_path.exists()
            with failure_path.open("a", encoding="utf-8") as handle:
                if header:
                    handle.write("stage,work_unit,status,reason\n")
                handle.write(f'"{name}","stage","failed","{str(exc).replace(chr(34), chr(39))}"\n')
            raise

    def run_demo(self) -> dict[str, Any]:
        artifact_root = self.root / "artifacts/demo"
        contracts = [
            self.root.parent / "Data_Center_VPP_English_Manuscript_and_Experimental_Protocol_NLR_NYISO_Revised.docx",
            self.root.parent / "CODEX_Executable_Experiment_Specification_English_NLR_NYISO_Revised.md",
        ]
        source_manifest = artifact_root / "manifests/source_manifest.json"
        self.stage(
            "contract",
            [source_manifest],
            lambda: atomic_json(
                source_manifest,
                {
                    "governing_files": [
                        {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in contracts
                    ],
                    "environment": environment_snapshot(self.root),
                },
            ),
            contracts,
        )

        demo_dir = self.root / "data/processed/demo"
        domain_paths = {
            "market": demo_dir / "nyiso_market_15m.parquet",
            "physical": demo_dir / "nlr_physical_15m.parquet",
            "workload": demo_dir / "nlr_workload_15m.parquet",
        }
        self.stage("demo_data", list(domain_paths.values()), lambda: generate_demo_domains(self.root, self.config))
        gate_path = artifact_root / "manifests/demo_gate.json"
        self.stage("demo_gate", [gate_path], lambda: atomic_json(gate_path, validate_demo_domains(domain_paths)), list(domain_paths.values()))
        physical_outputs = [artifact_root / "manifests/physical_calibration_manifest.json", artifact_root / "tables/nlr_twin_validation.csv"]
        self.stage("demo_physical", physical_outputs, lambda: calibrate_demo_twin(domain_paths["physical"], artifact_root, int(self.config["seed"])), [domain_paths["physical"]])
        energy_table = artifact_root / "tables/nyiso_forecast_main.csv"
        self.stage("demo_energy", [energy_table], lambda: run_demo_energy(domain_paths["market"], artifact_root, self.config), [domain_paths["market"]])
        workload_table = artifact_root / "tables/workload_forecast_main.csv"
        self.stage("demo_workload", [workload_table], lambda: run_demo_workload(domain_paths["workload"], artifact_root, self.config), [domain_paths["workload"]])
        transfer_outputs = [artifact_root / "manifests/digital_twin_transfer_manifest.json", artifact_root / "tables/digital_twin_transfer.csv"]
        self.stage("demo_transfer", transfer_outputs, lambda: deploy_demo_twins(self.root, physical_outputs[0], artifact_root), [physical_outputs[0]])
        scenario_outputs = [artifact_root / "scenarios/demo_reduced_scenarios.npz", artifact_root / "tables/scenario_diagnostics.csv", artifact_root / "manifests/scenario_manifest.json"]
        self.stage("demo_scenarios", scenario_outputs, lambda: build_demo_scenarios(domain_paths["market"], domain_paths["workload"], artifact_root, self.config), [domain_paths["market"], domain_paths["workload"]])
        optimization_outputs = [artifact_root / "tables/counterfactual_optimization_main.csv", artifact_root / "tables/failures.csv"]
        self.stage("demo_optimization", optimization_outputs, lambda: run_demo_policies(scenario_outputs[0], artifact_root, self.config), [scenario_outputs[0]])
        report_path = artifact_root / "report/run_summary.md"
        self.stage("demo_report", [report_path], lambda: build_demo_report(self.root, artifact_root, {}), physical_outputs + [energy_table, workload_table] + optimization_outputs)
        run_manifest = artifact_root / "manifests/run_manifest.json"
        self.stage(
            "demo_manifest",
            [run_manifest],
            lambda: build_run_manifest(
                self.root,
                artifact_root,
                {
                    "profile": "demo",
                    "synthetic": True,
                    "status": "DEMO_COMPLETE",
                    "config_hash": self.config_hash,
                    "code_hash": code_tree_hash(self.root),
                },
            ),
        )
        return {"status": "DEMO_COMPLETE", "artifact_root": artifact_root, "run_manifest": run_manifest}

    def reusable_demo(self, demo_config: Path) -> dict[str, Any] | None:
        """Reuse an unchanged, manifest-backed demo instead of repeating passed work."""
        artifact_root = self.root / "artifacts/demo"
        manifest_path = artifact_root / "manifests/run_manifest.json"
        gate_path = artifact_root / "manifests/demo_gate.json"
        if not (manifest_path.is_file() and gate_path.is_file()):
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            manifest.get("status") != "DEMO_COMPLETE"
            or manifest.get("synthetic") is not True
            or manifest.get("config_hash") != sha256_file(demo_config)
            or gate.get("status") != "passed"
        ):
            return None
        required = {
            "artifacts/demo/tables/nlr_twin_validation.csv",
            "artifacts/demo/tables/nyiso_forecast_main.csv",
            "artifacts/demo/tables/workload_forecast_main.csv",
            "artifacts/demo/tables/counterfactual_optimization_main.csv",
            "artifacts/demo/manifests/demo_gate.json",
        }
        recorded = {str(item.get("path")): item for item in manifest.get("artifacts", [])}
        if not required.issubset(recorded):
            return None
        for relative_path in required:
            path = self.root / relative_path
            if not path.is_file() or sha256_file(path) != recorded[relative_path].get("sha256"):
                return None
        return {
            "status": "DEMO_COMPLETE",
            "artifact_root": artifact_root,
            "run_manifest": manifest_path,
            "reused": True,
        }

    def run(self, target: str) -> dict[str, Any]:
        if target not in {"demo", "paper"}:
            raise ValueError("target must be demo or paper")
        if target == "demo":
            return self.run_demo()
        demo_config = self.root / "configs/demo.yaml"
        demo_orchestrator = Orchestrator(self.root, demo_config, self.resume)
        demo = demo_orchestrator.reusable_demo(demo_config) if self.resume else None
        if demo is None:
            demo = demo_orchestrator.run_demo()
        from dcvpp.paper import run_paper

        return run_paper(self.root, self.config_path, self.config, self.state, demo)


def run_pipeline(root: Path, config_path: Path, target: str, resume: bool = True) -> dict[str, Any]:
    return Orchestrator(root, config_path, resume).run(target)
