from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import gzip
from html import unescape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dcvpp.vpp_world_moe.migration import (  # noqa: E402
    DEVELOPMENT_STOP,
    DIRECT_ROUTE_MEASUREMENTS,
    INTERVAL_SECONDS,
    MIGRATION_EVIDENCE_SCHEMA,
    MigrationEvidenceError,
    RouteMeasurement,
    ZONE_NAMES,
    aggregate_ripe_results,
    capacity_bridge,
    load_work_unit_bridge,
    sha256_file,
    strict_reqmem_statistics,
    validate_migration_manifest,
)


START = datetime(2022, 1, 1, tzinfo=timezone.utc)
STOP = DEVELOPMENT_STOP
RIPE_RESULTS = "https://atlas.ripe.net/api/v2/measurements/{measurement}/results/"


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignored += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored:
            self._ignored -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored:
            self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"\s+", " ", unescape(" ".join(self.parts))).strip()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _chunks(start: datetime, stop: datetime, days: int) -> list[tuple[datetime, datetime]]:
    values: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < stop:
        next_ = min(cursor + timedelta(days=int(days)), stop)
        values.append((cursor, next_))
        cursor = next_
    return values


def _route_label(route: RouteMeasurement) -> str:
    source = route.source_zone.replace(".", "").replace(" ", "_")
    destination = route.destination_zone.replace(".", "").replace(" ", "_")
    return f"{source}_to_{destination}"


def _download_json(url: str, *, params: Mapping[str, Any] | None = None) -> Any:
    last: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(
                url,
                params=dict(params or {}),
                timeout=(20, 120),
                headers={"User-Agent": "dcvpp-worldmoe-migration-evidence/1.0"},
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt < 3:
                time.sleep(2**attempt)
    raise MigrationEvidenceError(f"download failed: {url}: {last}")


def _download_chunk(
    *,
    route: RouteMeasurement,
    start: datetime,
    stop: datetime,
    raw_root: Path,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    label = _route_label(route)
    raw_path = raw_root / label / (
        f"{start:%Y%m%dT%H%M%SZ}_{stop:%Y%m%dT%H%M%SZ}.json.gz"
    )
    url = RIPE_RESULTS.format(measurement=route.measurement_id)
    params = {
        "start": int(start.timestamp()),
        "stop": int(stop.timestamp()) - 1,
        "probe_ids": route.source_probe_id,
        "format": "json",
    }
    if raw_path.is_file():
        payload = json.loads(gzip.decompress(raw_path.read_bytes()).decode("utf-8"))
    else:
        payload = _download_json(url, params=params)
        encoded = gzip.compress(
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            compresslevel=6,
        )
        _atomic_bytes(raw_path, encoded)
    if not isinstance(payload, list):
        raise MigrationEvidenceError("RIPE results endpoint did not return a list")
    filtered = [
        item
        for item in payload
        if isinstance(item, Mapping)
        and int(start.timestamp()) <= int(item.get("timestamp", -1))
        < int(stop.timestamp())
    ]
    timestamps = [int(item["timestamp"]) for item in filtered]
    return filtered, {
        "route": [route.source_zone, route.destination_zone],
        "measurement_id": route.measurement_id,
        "source_probe_id": route.source_probe_id,
        "request_url": requests.Request("GET", url, params=params).prepare().url,
        "raw_path": raw_path.relative_to(ROOT).as_posix(),
        "raw_sha256": sha256_file(raw_path),
        "result_rows": len(filtered),
        "first_timestamp": min(timestamps) if timestamps else None,
        "last_timestamp": max(timestamps) if timestamps else None,
    }


def _download_source_snapshot(
    *,
    name: str,
    url: str,
    source_root: Path,
    expected_pattern: str | None = None,
) -> dict[str, Any]:
    response = requests.get(
        url,
        timeout=(20, 120),
        headers={"User-Agent": "dcvpp-worldmoe-migration-evidence/1.0"},
    )
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    suffix = ".json" if "application/json" in content_type else ".html"
    path = source_root / f"{name}{suffix}"
    _atomic_bytes(path, response.content)
    verified = True
    excerpt: str | None = None
    if expected_pattern is not None:
        parser = _VisibleText()
        parser.feed(response.text)
        text = parser.text()
        match = re.search(expected_pattern, text, flags=re.IGNORECASE)
        verified = match is not None
        if match is not None:
            excerpt = match.group(0)
    if not verified:
        raise MigrationEvidenceError(f"official source wording not found: {url}")
    return {
        "name": name,
        "url": url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_path": path.relative_to(ROOT).as_posix(),
        "snapshot_sha256": sha256_file(path),
        "expected_claim_verified": verified,
        "matched_excerpt": excerpt,
    }


def _unsupported_routes() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    enabled = {("WEST", "N.Y.C."), ("N.Y.C.", "WEST")}
    for source in ZONE_NAMES:
        for destination in ZONE_NAMES:
            if source == destination or (source, destination) in enabled:
                continue
            reason = (
                "no_direct_2022_2024_ripe_campus_endpoint_evidence"
                if "CAPITL" in {source, destination}
                or "LONGIL" in {source, destination}
                else "no_direct_route_measurement"
            )
            result.append(
                {
                    "route": [source, destination],
                    "enabled": False,
                    "reason": reason,
                    "via_nyc_inference_allowed": False,
                }
            )
    return result


def materialize(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.start != "2022-01-01" or args.stop_exclusive != "2025-01-01":
        raise MigrationEvidenceError("registered development window is 2022--2024")
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    stop = datetime.fromisoformat(args.stop_exclusive).replace(tzinfo=timezone.utc)
    if stop != DEVELOPMENT_STOP:
        raise MigrationEvidenceError("2025 must remain unopened")

    output_root = (ROOT / args.output_root).resolve()
    raw_root = output_root / "raw"
    source_root = output_root / "sources"
    evidence_path = output_root / "ripe_direct_routes_15min.parquet"
    manifest_path = output_root / "migration_evidence_manifest.json"

    jobs = [
        (route, chunk_start, chunk_stop)
        for route in DIRECT_ROUTE_MEASUREMENTS
        for chunk_start, chunk_stop in _chunks(start, stop, args.chunk_days)
    ]
    route_rows: dict[tuple[str, str], list[Mapping[str, Any]]] = {
        (route.source_zone, route.destination_zone): []
        for route in DIRECT_ROUTE_MEASUREMENTS
    }
    requests_manifest: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {
            pool.submit(
                _download_chunk,
                route=route,
                start=chunk_start,
                stop=chunk_stop,
                raw_root=raw_root,
            ): route
            for route, chunk_start, chunk_stop in jobs
        }
        completed = 0
        for future in as_completed(futures):
            route = futures[future]
            rows, request_record = future.result()
            route_rows[(route.source_zone, route.destination_zone)].extend(rows)
            requests_manifest.append(request_record)
            completed += 1
            if completed % 12 == 0 or completed == len(jobs):
                print(f"RIPE chunks {completed}/{len(jobs)}", flush=True)

    route_frames: list[pd.DataFrame] = []
    coverage: list[dict[str, Any]] = []
    for route in DIRECT_ROUTE_MEASUREMENTS:
        key = (route.source_zone, route.destination_zone)
        rows = sorted(route_rows[key], key=lambda item: int(item["timestamp"]))
        frame = aggregate_ripe_results(rows, route)
        route_frames.append(frame)
        coverage.append(
            {
                "route": list(key),
                "raw_results": len(rows),
                "fifteen_minute_rows": len(frame),
                "first_observation_utc": (
                    frame["timestamp_utc"].min().isoformat() if len(frame) else None
                ),
                "last_observation_utc": (
                    frame["timestamp_utc"].max().isoformat() if len(frame) else None
                ),
                "observed_calendar_days": (
                    int(frame["bucket_start_utc"].dt.date.nunique())
                    if len(frame)
                    else 0
                ),
            }
        )
    evidence = pd.concat(route_frames, ignore_index=True)
    evidence = evidence.sort_values(
        ["source_node", "destination_node", "timestamp_utc"]
    ).drop_duplicates(["source_node", "destination_node", "timestamp_utc"])
    if evidence.empty:
        raise MigrationEvidenceError("no direct RIPE route observations materialized")
    temporary_parquet = evidence_path.with_name(
        f".{evidence_path.name}.{os.getpid()}.tmp"
    )
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_parquet(temporary_parquet, index=False, compression="zstd")
    os.replace(temporary_parquet, evidence_path)

    reqmem = strict_reqmem_statistics(ROOT / args.workload_events)
    unit_bridge = load_work_unit_bridge(ROOT / args.work_unit_bridge)
    bridge = capacity_bridge(
        nominal_gbps=float(args.nominal_gbps),
        interval_seconds=INTERVAL_SECONDS,
        reqmem=reqmem,
        work_unit_bridge=unit_bridge,
    )

    official_sources = [
        _download_source_snapshot(
            name="university_at_buffalo_nysernet_capacity",
            url=(
                "https://www.buffalo.edu/studentaccounts/tuition-and-fees/"
                "broad-based-fees/comprehensive-fee/comprehensive-fee-components/"
                "technology.html"
            ),
            source_root=source_root,
            expected_pattern=r"Two 100Gbps connections to NYSERNet",
        ),
        _download_source_snapshot(
            name="nysernet_service_speed",
            url="https://nysernet.org/research-education-network",
            source_root=source_root,
            expected_pattern=r"From 1Gb to 100Gb\+",
        ),
    ]
    for anchor_id in (1791, 2492):
        official_sources.append(
            _download_source_snapshot(
                name=f"ripe_anchor_{anchor_id}",
                url=f"https://atlas.ripe.net/api/v2/anchors/{anchor_id}/",
                source_root=source_root,
            )
        )
    for route in DIRECT_ROUTE_MEASUREMENTS:
        official_sources.append(
            _download_source_snapshot(
                name=f"ripe_measurement_{route.measurement_id}",
                url=(
                    "https://atlas.ripe.net/api/v2/measurements/"
                    f"{route.measurement_id}/"
                ),
                source_root=source_root,
            )
        )

    manifest = {
        "schema": MIGRATION_EVIDENCE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "MATERIALIZED",
        "development_window": {
            "start": start.isoformat(),
            "stop_exclusive": stop.isoformat(),
            "years_read": [2022, 2023, 2024],
            "opened_final_test_2025": False,
        },
        "causal_contract": {
            "selection": "latest_result_timestamp_lte_origin",
            "history_tokens": "each token uses result timestamp lte token timestamp",
            "future_horizon": "hold_forward_latest_origin_observation",
            "within_day_future_results_allowed": False,
            "require_simultaneous_reverse_anchor_measurement": False,
            "directed_routes_independently_gated": True,
            "max_age_minutes": float(args.max_age_minutes),
            "max_packet_loss_rate": float(args.max_packet_loss_rate),
            "min_sent_packets": int(args.min_sent_packets),
        },
        "enabled_routes": [["WEST", "N.Y.C."], ["N.Y.C.", "WEST"]],
        "direct_route_measurements": [
            route.as_dict() for route in DIRECT_ROUTE_MEASUREMENTS
        ],
        "unsupported_routes": _unsupported_routes(),
        "route_coverage": coverage,
        "capacity_bridge": bridge,
        "strict_reqmem": reqmem,
        "work_unit_bridge": unit_bridge,
        "capacity_claim_boundary": {
            "published_100gbps_is_endpoint_access_upper_bound": True,
            "measured_end_to_end_available_bandwidth": False,
            "measured_migration_execution_log": False,
            "use": "operational_proxy_sensitivity",
        },
        "graph_feature_order": [
            "capacity_work_units_per_interval",
            "median_rtt_ms_div_100",
            "p95_rtt_ms_div_100",
            "packet_loss_rate",
            "age_div_max_age",
            "available",
        ],
        "artifacts": {
            "evidence_path": evidence_path.relative_to(ROOT).as_posix(),
            "evidence_sha256": sha256_file(evidence_path),
            "evidence_rows": len(evidence),
            "raw_request_chunks": sorted(
                requests_manifest,
                key=lambda item: (item["route"], item["request_url"]),
            ),
        },
        "official_sources": official_sources,
    }
    validate_migration_manifest(manifest)
    _atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "evidence": evidence_path.relative_to(ROOT).as_posix(),
                "manifest": manifest_path.relative_to(ROOT).as_posix(),
                "rows": len(evidence),
                "capacity_work_units_per_interval": bridge[
                    "capacity_work_units_per_interval"
                ],
                "strict_reqmem_coverage": reqmem["strict_parse_coverage"],
            },
            indent=2,
        ),
        flush=True,
    )
    return evidence_path, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize causal RIPE/official-link migration evidence"
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/vpp_world_moe_r5_v1_2025final/data/migration",
    )
    parser.add_argument(
        "--workload-events",
        default=(
            "data/processed/research/physical/run_id=paper-physical-v1/"
            "nlr295_workload_events.parquet"
        ),
    )
    parser.add_argument(
        "--work-unit-bridge",
        default=(
            "artifacts/optimization_branch/paper-optimization-v1/"
            "workload_normalization_admission.json"
        ),
    )
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--stop-exclusive", default="2025-01-01")
    parser.add_argument("--nominal-gbps", type=float, default=100.0)
    parser.add_argument("--max-age-minutes", type=float, default=30.0)
    parser.add_argument("--max-packet-loss-rate", type=float, default=0.05)
    parser.add_argument("--min-sent-packets", type=int, default=3)
    parser.add_argument("--chunk-days", type=int, default=31)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    materialize(parse_args())
