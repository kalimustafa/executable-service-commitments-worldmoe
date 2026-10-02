"""Causal, fail-closed migration evidence for VPP-WorldMoE.

The adapter deliberately separates three quantities that are easy to conflate:

* RIPE Atlas ping observations determine whether a route was known to be
  reachable at a rolling origin;
* a published access-link rate is only an operational upper-bound proxy; and
* the NLR workload-unit bridge converts transferable raw nodes into the same
  normalized work units used by the physical twin.

Only measurements whose result timestamp is no later than an origin may be
used.  Missing, stale, lossy, or unsupported routes have exactly zero capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor


MIGRATION_EVIDENCE_SCHEMA = "dcvpp.vpp_world_moe.migration-evidence.v1"
MIGRATION_TABLE_SCHEMA = "dcvpp.vpp_world_moe.ripe-route-15min.v1"
ZONE_NAMES = ("WEST", "CAPITL", "N.Y.C.", "LONGIL")
ZONE_TO_NODE = {name: index for index, name in enumerate(ZONE_NAMES)}
DEVELOPMENT_STOP = datetime(2025, 1, 1, tzinfo=timezone.utc)
INTERVAL_SECONDS = 900


class MigrationEvidenceError(ValueError):
    """Migration evidence is missing, non-causal, or semantically incompatible."""


@dataclass(frozen=True)
class RouteMeasurement:
    """One direct RIPE Atlas source-probe to target-anchor route."""

    source_zone: str
    destination_zone: str
    measurement_id: int
    source_probe_id: int
    target_anchor_id: int
    source_address: str
    destination_address: str

    @property
    def source_node(self) -> int:
        return ZONE_TO_NODE[self.source_zone]

    @property
    def destination_node(self) -> int:
        return ZONE_TO_NODE[self.destination_zone]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_zone": self.source_zone,
            "destination_zone": self.destination_zone,
            "source_node": self.source_node,
            "destination_node": self.destination_node,
            "measurement_id": self.measurement_id,
            "source_probe_id": self.source_probe_id,
            "target_anchor_id": self.target_anchor_id,
            "source_address": self.source_address,
            "destination_address": self.destination_address,
        }


DIRECT_ROUTE_MEASUREMENTS = (
    RouteMeasurement(
        source_zone="N.Y.C.",
        destination_zone="WEST",
        measurement_id=21802260,
        source_probe_id=6946,
        target_anchor_id=1791,
        source_address="156.154.35.254",
        destination_address="185.57.190.200",
    ),
    RouteMeasurement(
        source_zone="WEST",
        destination_zone="N.Y.C.",
        measurement_id=29786041,
        source_probe_id=6557,
        target_anchor_id=2492,
        source_address="185.57.190.200",
        destination_address="156.154.35.254",
    ),
)


@dataclass(frozen=True)
class MigrationSnapshot:
    """Route constraints and graph features known at one causal cutoff."""

    capacity: Tensor
    allowed: Tensor
    edge_features: Tensor
    age_minutes: Tensor
    observed_timestamp: Tensor

    @classmethod
    def unavailable(cls) -> "MigrationSnapshot":
        return cls(
            capacity=torch.zeros(4, 4),
            allowed=torch.zeros(4, 4, dtype=torch.bool),
            edge_features=torch.zeros(4, 4, 6),
            age_minutes=torch.full((4, 4), float("inf")),
            observed_timestamp=torch.full((4, 4), -1, dtype=torch.long),
        )


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 for one lineage source."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_reqmem_statistics(path: str | Path) -> dict[str, Any]:
    """Fit train-only GiB-per-requested-node quantiles from strict Slurm ReqMem.

    A record is admitted only when ReqMem is ``number + K/M/G/T + n/c`` and
    both ``nodes_used`` and ``processors_used`` are positive.  ``n`` already
    denotes memory per node.  ``c`` is converted from memory per CPU to memory
    per requested node with ``processors_used / nodes_used``.
    """

    import duckdb

    source = Path(path).resolve()
    if not source.is_file():
        raise MigrationEvidenceError(f"NLR workload events are missing: {source}")
    connection = duckdb.connect()
    try:
        connection.read_parquet(str(source)).create_view("nlr_events")
        row = connection.execute(
            r"""
            WITH base AS (
              SELECT trim(memory_requested_raw) AS req,
                     nodes_used,
                     processors_used
              FROM nlr_events
              WHERE split = 'train'
            ), parsed AS (
              SELECT *,
                CAST(regexp_extract(
                  req,
                  '^([0-9]+(?:\.[0-9]+)?)([KMGT])([nc])$',
                  1
                ) AS DOUBLE) AS amount,
                regexp_extract(
                  req,
                  '^([0-9]+(?:\.[0-9]+)?)([KMGT])([nc])$',
                  2
                ) AS unit,
                regexp_extract(
                  req,
                  '^([0-9]+(?:\.[0-9]+)?)([KMGT])([nc])$',
                  3
                ) AS mem_scope
              FROM base
              WHERE regexp_full_match(
                req, '^[0-9]+(?:\.[0-9]+)?[KMGT][nc]$'
              )
            ), admitted AS (
              SELECT mem_scope,
                amount
                * CASE unit
                    WHEN 'K' THEN 1.0 / 1048576.0
                    WHEN 'M' THEN 1.0 / 1024.0
                    WHEN 'G' THEN 1.0
                    WHEN 'T' THEN 1024.0
                  END
                * CASE mem_scope
                    WHEN 'n' THEN 1.0
                    ELSE processors_used / nodes_used
                  END AS gib_per_requested_node,
                amount
                * CASE unit
                    WHEN 'K' THEN 1.0 / 1048576.0
                    WHEN 'M' THEN 1.0 / 1024.0
                    WHEN 'G' THEN 1.0
                    WHEN 'T' THEN 1024.0
                  END
                * CASE mem_scope
                    WHEN 'n' THEN nodes_used
                    ELSE processors_used
                  END AS gib_per_job
              FROM parsed
              WHERE amount > 0
                AND nodes_used > 0
                AND processors_used > 0
            )
            SELECT
              (SELECT count(*) FROM base) AS train_rows,
              count(*) AS admitted_rows,
              count(*) FILTER (WHERE mem_scope = 'n') AS admitted_n_rows,
              count(*) FILTER (WHERE mem_scope = 'c') AS admitted_c_rows,
              quantile_cont(gib_per_requested_node, 0.50) AS p50,
              quantile_cont(gib_per_requested_node, 0.90) AS p90,
              quantile_cont(gib_per_requested_node, 0.95) AS p95,
              quantile_cont(gib_per_requested_node, 0.99) AS p99,
              quantile_cont(gib_per_job, 0.95) AS per_job_p95
            FROM admitted
            WHERE isfinite(gib_per_requested_node)
              AND gib_per_requested_node > 0
            """
        ).fetchone()
    finally:
        connection.close()
    if row is None or int(row[1]) <= 0 or float(row[6]) <= 0:
        raise MigrationEvidenceError("strict ReqMem parsing produced no payload data")
    total, admitted, admitted_n, admitted_c, p50, p90, p95, p99, per_job_p95 = row
    return {
        "source_path": source.as_posix(),
        "source_sha256": sha256_file(source),
        "fit_split": "train",
        "train_rows": int(total),
        "strict_admitted_rows": int(admitted),
        "strict_admitted_n_rows": int(admitted_n),
        "strict_admitted_c_rows": int(admitted_c),
        "strict_parse_coverage": float(admitted) / float(total),
        "strict_pattern": "number+[K|M|G|T]+[n|c]",
        "resource_validity": "nodes_used>0 and processors_used>0",
        "gib_per_requested_node_formula": {
            "n": "ReqMem converted to GiB",
            "c": "ReqMem_GiB_per_cpu * processors_used / nodes_used",
        },
        "gib_per_requested_node_p50": float(p50),
        "gib_per_requested_node_p90": float(p90),
        "gib_per_requested_node_p95": float(p95),
        "gib_per_requested_node_p99": float(p99),
        "gib_per_job_p95_diagnostic_only": float(per_job_p95),
    }


def load_work_unit_bridge(path: str | Path) -> dict[str, Any]:
    """Load the registered raw-node to physical-work-unit bridge."""

    source = Path(path).resolve()
    if not source.is_file():
        raise MigrationEvidenceError(f"work-unit bridge is missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    contract = payload.get("contract") if isinstance(payload, Mapping) else None
    if not isinstance(contract, Mapping) or (
        contract.get("schema_version") != "workload_unit_bridge_v1"
        or contract.get("source_quantity") != "submitted_nodes_requested"
        or contract.get("target_quantity")
        != "aggregate_normalized_physical_work_units"
        or contract.get("locked_2025_outcomes_used") is not False
    ):
        raise MigrationEvidenceError("work-unit bridge contract is incompatible")
    scale = float(contract.get("scale_work_units_per_raw_node", 0.0))
    train_q01 = float(contract.get("train_q01_raw_nodes", float("nan")))
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(train_q01):
        raise MigrationEvidenceError("work-unit bridge scale is invalid")
    return {
        "source_path": source.as_posix(),
        "source_sha256": sha256_file(source),
        "contract_sha256": str(payload.get("contract_sha256", "")),
        "schema_version": str(contract["schema_version"]),
        "scale_work_units_per_raw_node": scale,
        "train_q01_raw_nodes": train_q01,
        "training_start_utc": str(contract.get("training_start_utc")),
        "training_end_utc_inclusive": str(
            contract.get("training_end_utc_inclusive")
        ),
        "locked_2025_outcomes_used": False,
    }


def capacity_bridge(
    *,
    nominal_gbps: float,
    interval_seconds: int,
    reqmem: Mapping[str, Any],
    work_unit_bridge: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert a published link-rate upper bound to normalized work units."""

    p95 = float(reqmem["gib_per_requested_node_p95"])
    scale = float(work_unit_bridge["scale_work_units_per_raw_node"])
    q01 = float(work_unit_bridge["train_q01_raw_nodes"])
    transferable_gib = float(nominal_gbps) * 1.0e9 * int(interval_seconds) / 8.0
    transferable_gib /= float(1024**3)
    transferable_raw_nodes = transferable_gib / p95
    capacity = max(transferable_raw_nodes - q01, 0.0) * scale
    if not np.isfinite(capacity) or capacity <= 0:
        raise MigrationEvidenceError("capacity bridge did not produce a positive bound")
    return {
        "evidence_role": "operational_proxy_sensitivity_not_execution_log",
        "nominal_link_rate_gbps": float(nominal_gbps),
        "interval_seconds": int(interval_seconds),
        "transferable_gib_per_interval": transferable_gib,
        "payload_gib_per_requested_node_p95": p95,
        "transferable_raw_nodes_per_interval": transferable_raw_nodes,
        "raw_node_q01_offset": q01,
        "scale_work_units_per_raw_node": scale,
        "capacity_work_units_per_interval": capacity,
        "formula": (
            "max((Gbps*1e9*interval_seconds/8/2^30)/"
            "ReqMem_GiB_per_requested_node_p95-train_q01_raw_nodes,0)"
            "*scale_work_units_per_raw_node"
        ),
    }


def _rtt_values(result: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    raw = result.get("result")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if not isinstance(item, Mapping) or "rtt" not in item:
                continue
            value = float(item["rtt"])
            if np.isfinite(value) and value >= 0:
                values.append(value)
    return values


def aggregate_ripe_results(
    results: Sequence[Mapping[str, Any]], route: RouteMeasurement
) -> pd.DataFrame:
    """Aggregate direct ping results into causal 15-minute observations."""

    rows: list[dict[str, Any]] = []
    for result in results:
        if (
            int(result.get("msm_id", -1)) != route.measurement_id
            or int(result.get("prb_id", -1)) != route.source_probe_id
            or str(result.get("src_addr", result.get("from", "")))
            != route.source_address
            or str(result.get("dst_addr", "")) != route.destination_address
        ):
            raise MigrationEvidenceError("RIPE result does not match the direct route")
        timestamp = int(result.get("timestamp", -1))
        if timestamp < 0 or timestamp >= int(DEVELOPMENT_STOP.timestamp()):
            raise MigrationEvidenceError("RIPE result crosses the development window")
        sent = max(int(result.get("sent", 0)), 0)
        received = max(int(result.get("rcvd", 0)), 0)
        rows.append(
            {
                "timestamp_utc": pd.Timestamp(timestamp, unit="s", tz="UTC"),
                "sent": sent,
                "received": min(received, sent),
                "rtt_values": _rtt_values(result),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=(
                "schema",
                "bucket_start_utc",
                "timestamp_utc",
                "source_zone",
                "destination_zone",
                "source_node",
                "destination_node",
                "measurement_id",
                "source_probe_id",
                "target_anchor_id",
                "median_rtt_ms",
                "p95_rtt_ms",
                "packet_loss_rate",
                "sent_packets",
                "received_packets",
                "measurement_results",
                "rtt_samples",
            )
        )
    frame = pd.DataFrame(rows)
    frame["bucket_start_utc"] = frame["timestamp_utc"].dt.floor("15min")
    materialized: list[dict[str, Any]] = []
    for bucket, group in frame.groupby("bucket_start_utc", sort=True):
        rtts = np.asarray(
            [value for values in group["rtt_values"] for value in values],
            dtype=np.float64,
        )
        sent = int(group["sent"].sum())
        received = int(group["received"].sum())
        median = float(np.median(rtts)) if rtts.size else float("nan")
        p95 = float(np.quantile(rtts, 0.95)) if rtts.size else float("nan")
        materialized.append(
            {
                "schema": MIGRATION_TABLE_SCHEMA,
                "bucket_start_utc": bucket,
                "timestamp_utc": group["timestamp_utc"].max(),
                "source_zone": route.source_zone,
                "destination_zone": route.destination_zone,
                "source_node": route.source_node,
                "destination_node": route.destination_node,
                "measurement_id": route.measurement_id,
                "source_probe_id": route.source_probe_id,
                "target_anchor_id": route.target_anchor_id,
                "median_rtt_ms": median,
                "p95_rtt_ms": p95,
                "packet_loss_rate": (
                    float(sent - received) / float(sent) if sent else 1.0
                ),
                "sent_packets": sent,
                "received_packets": received,
                "measurement_results": int(len(group)),
                "rtt_samples": int(rtts.size),
            }
        )
    return pd.DataFrame(materialized)


def validate_migration_manifest(payload: Mapping[str, Any]) -> None:
    """Validate the scientific and temporal boundaries of a migration manifest."""

    if payload.get("schema") != MIGRATION_EVIDENCE_SCHEMA:
        raise MigrationEvidenceError("unexpected migration evidence schema")
    window = payload.get("development_window")
    if not isinstance(window, Mapping) or (
        window.get("stop_exclusive") != "2025-01-01T00:00:00+00:00"
        or window.get("opened_final_test_2025") is not False
    ):
        raise MigrationEvidenceError("migration evidence must keep 2025 sealed")
    contract = payload.get("causal_contract")
    if not isinstance(contract, Mapping) or (
        contract.get("selection") != "latest_result_timestamp_lte_origin"
        or contract.get("future_horizon")
        != "hold_forward_latest_origin_observation"
        or contract.get("require_simultaneous_reverse_anchor_measurement") is not False
        or contract.get("directed_routes_independently_gated") is not True
    ):
        raise MigrationEvidenceError("migration evidence causality is not locked")
    routes = payload.get("enabled_routes")
    if routes != [["WEST", "N.Y.C."], ["N.Y.C.", "WEST"]]:
        raise MigrationEvidenceError("only direct WEST/NYC routes may be enabled")
    missing = payload.get("unsupported_routes")
    if not isinstance(missing, Sequence) or len(missing) != 10:
        raise MigrationEvidenceError("unsupported direct routes are not enumerated")
    bridge = payload.get("capacity_bridge")
    if not isinstance(bridge, Mapping) or (
        bridge.get("evidence_role")
        != "operational_proxy_sensitivity_not_execution_log"
        or float(bridge.get("capacity_work_units_per_interval", 0.0)) <= 0
    ):
        raise MigrationEvidenceError("migration capacity bridge is invalid")


class MigrationEvidenceStore:
    """In-memory as-of index over materialized direct RIPE route evidence."""

    def __init__(
        self,
        frame: pd.DataFrame,
        manifest: Mapping[str, Any],
        *,
        max_age_minutes: float,
        max_packet_loss_rate: float,
        min_sent_packets: int,
    ) -> None:
        validate_migration_manifest(manifest)
        self.manifest = dict(manifest)
        self.max_age_minutes = float(max_age_minutes)
        self.max_packet_loss_rate = float(max_packet_loss_rate)
        self.min_sent_packets = int(min_sent_packets)
        if (
            self.max_age_minutes <= 0
            or not 0 <= self.max_packet_loss_rate <= 1
            or self.min_sent_packets <= 0
        ):
            raise MigrationEvidenceError("migration availability thresholds are invalid")
        required = {
            "schema",
            "timestamp_utc",
            "source_node",
            "destination_node",
            "median_rtt_ms",
            "p95_rtt_ms",
            "packet_loss_rate",
            "sent_packets",
        }
        if not required.issubset(frame.columns):
            raise MigrationEvidenceError("migration evidence table is incomplete")
        table = frame.copy()
        table["timestamp_utc"] = pd.to_datetime(table["timestamp_utc"], utc=True)
        if bool((table["timestamp_utc"] >= DEVELOPMENT_STOP).any()):
            raise MigrationEvidenceError("migration table opened 2025")
        if set(table["schema"].dropna().astype(str)) != {MIGRATION_TABLE_SCHEMA}:
            raise MigrationEvidenceError("migration table schema is inconsistent")
        self.capacity_work_units = float(
            manifest["capacity_bridge"]["capacity_work_units_per_interval"]
        )
        self._routes: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        for (source, destination), group in table.groupby(
            ["source_node", "destination_node"], sort=True
        ):
            key = (int(source), int(destination))
            if key not in {(0, 2), (2, 0)}:
                raise MigrationEvidenceError("table contains an unsupported route")
            ordered = group.sort_values("timestamp_utc")
            timestamps = ordered["timestamp_utc"].astype("int64").to_numpy()
            if timestamps.size and np.any(np.diff(timestamps) <= 0):
                raise MigrationEvidenceError("route timestamps must be strictly ordered")
            self._routes[key] = {
                "timestamp_ns": timestamps,
                "median_rtt_ms": ordered["median_rtt_ms"].to_numpy(np.float64),
                "p95_rtt_ms": ordered["p95_rtt_ms"].to_numpy(np.float64),
                "packet_loss_rate": ordered["packet_loss_rate"].to_numpy(
                    np.float64
                ),
                "sent_packets": ordered["sent_packets"].to_numpy(np.int64),
            }

    @classmethod
    def from_artifacts(
        cls,
        evidence_path: str | Path,
        manifest_path: str | Path,
        *,
        max_age_minutes: float,
        max_packet_loss_rate: float,
        min_sent_packets: int,
    ) -> "MigrationEvidenceStore":
        evidence = Path(evidence_path).resolve()
        manifest_file = Path(manifest_path).resolve()
        if not evidence.is_file() or not manifest_file.is_file():
            raise MigrationEvidenceError("migration evidence artifacts are missing")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        frame = pd.read_parquet(evidence)
        return cls(
            frame,
            manifest,
            max_age_minutes=max_age_minutes,
            max_packet_loss_rate=max_packet_loss_rate,
            min_sent_packets=min_sent_packets,
        )

    @property
    def supported_routes(self) -> frozenset[tuple[int, int]]:
        return frozenset(self._routes)

    def snapshot(self, cutoff: datetime) -> MigrationSnapshot:
        if cutoff.tzinfo is None:
            raise MigrationEvidenceError("migration cutoff must be timezone-aware")
        instant = cutoff.astimezone(timezone.utc)
        if instant >= DEVELOPMENT_STOP:
            raise MigrationEvidenceError("2025 migration evidence is sealed")
        cutoff_ns = int(pd.Timestamp(instant).value)
        snapshot = MigrationSnapshot.unavailable()
        capacity = snapshot.capacity.clone()
        allowed = snapshot.allowed.clone()
        features = snapshot.edge_features.clone()
        ages = snapshot.age_minutes.clone()
        observed = snapshot.observed_timestamp.clone()
        for (source, destination), values in self._routes.items():
            index = int(
                np.searchsorted(values["timestamp_ns"], cutoff_ns, side="right") - 1
            )
            if index < 0:
                continue
            timestamp_ns = int(values["timestamp_ns"][index])
            age = (cutoff_ns - timestamp_ns) / 60.0e9
            median = float(values["median_rtt_ms"][index])
            p95 = float(values["p95_rtt_ms"][index])
            loss = float(values["packet_loss_rate"][index])
            sent = int(values["sent_packets"][index])
            available = (
                0 <= age <= self.max_age_minutes
                and np.isfinite(median)
                and median >= 0
                and np.isfinite(p95)
                and p95 >= median
                and np.isfinite(loss)
                and 0 <= loss <= self.max_packet_loss_rate
                and sent >= self.min_sent_packets
            )
            ages[source, destination] = float(age)
            observed[source, destination] = timestamp_ns // 1_000_000_000
            if not available:
                continue
            capacity[source, destination] = self.capacity_work_units
            allowed[source, destination] = True
            # Per-edge typed semantics: normalized route capacity, median RTT,
            # p95 RTT, packet loss, evidence age, and availability.
            features[source, destination] = torch.tensor(
                (
                    self.capacity_work_units,
                    median / 100.0,
                    p95 / 100.0,
                    loss,
                    age / self.max_age_minutes,
                    1.0,
                ),
                dtype=torch.float32,
            )
        return MigrationSnapshot(capacity, allowed, features, ages, observed)

    def causal_day_series(
        self, operating_day: date, *, origin: int, steps: int = 96
    ) -> tuple[Tensor, Tensor, MigrationSnapshot]:
        """Build history and origin-held future features without future results."""

        if operating_day.year >= 2025:
            raise MigrationEvidenceError("2025 migration series is sealed")
        if not 0 <= int(origin) < int(steps):
            raise MigrationEvidenceError("rolling origin is outside the day")
        day_start = datetime.combine(
            operating_day, datetime.min.time(), tzinfo=timezone.utc
        )
        origin_time = day_start + timedelta(minutes=15 * int(origin))
        features: list[Tensor] = []
        masks: list[Tensor] = []
        origin_snapshot: MigrationSnapshot | None = None
        for step in range(int(steps)):
            effective = min(step, int(origin))
            snapshot = self.snapshot(day_start + timedelta(minutes=15 * effective))
            features.append(snapshot.edge_features)
            masks.append(snapshot.allowed)
            if step == int(origin):
                origin_snapshot = snapshot
        if origin_snapshot is None:
            origin_snapshot = self.snapshot(origin_time)
        return torch.stack(features), torch.stack(masks), origin_snapshot


__all__ = [
    "DEVELOPMENT_STOP",
    "DIRECT_ROUTE_MEASUREMENTS",
    "INTERVAL_SECONDS",
    "MIGRATION_EVIDENCE_SCHEMA",
    "MIGRATION_TABLE_SCHEMA",
    "MigrationEvidenceError",
    "MigrationEvidenceStore",
    "MigrationSnapshot",
    "RouteMeasurement",
    "ZONE_NAMES",
    "aggregate_ripe_results",
    "capacity_bridge",
    "load_work_unit_bridge",
    "sha256_file",
    "strict_reqmem_statistics",
    "validate_migration_manifest",
]
