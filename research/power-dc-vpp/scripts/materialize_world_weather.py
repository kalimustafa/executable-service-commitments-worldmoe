"""Materialise causal NOAA weather forecasts for VPP-WorldMoE.

The formal V1 transport is ``mixed``: four-site TMP comes from compact HRRR
forecast-Zarr chunks and DSWRF comes from the matching HRRR GRIB2 message
selected through its official ``.idx`` byte range.  Historical forecast-Zarr
DSWRF arrays expose metadata but no data chunks for much of 2022--2024, so the
two transports keep one HRRR run and one conservative availability time.

Formal resumable command::

    python scripts/materialize_world_weather.py \
      --cache-root data/processed/vpp_world_weather \
      --transport mixed --source hrrr \
      --start-date 2022-01-01 --end-date 2024-12-31 \
      --cycles 0,6,12,18 --forecast-hours 1-48 \
      --include-prior-18z --workers 4

The command rejects 2025 before constructing any remote object request.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dcvpp.vpp_world_moe.weather import (  # noqa: E402
    EccodesPointDecoder,
    HRRR_DEFAULT_FEATURES,
    HRRR_ZARR_VARIABLES,
    HrrrMixedMaterializer,
    HrrrZarrMaterializer,
    NBM_DEFAULT_FEATURES,
    NYISO_WEATHER_SITES,
    WEATHER_SOURCES,
    WeatherRangeMaterializer,
    WorldWeatherStore,
    iter_run_initialisations,
)


UTC = timezone.utc
V1_MIXED_FEATURES = ("air_temperature_2m_k", "downward_shortwave_wm2")


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"descending integer range: {item}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("integer list cannot be empty")
    return tuple(dict.fromkeys(values))


def _parse_run_init(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("--run-init must contain an explicit UTC offset")
    return parsed.astimezone(UTC)


def _build_runs(arguments: argparse.Namespace) -> tuple[datetime, ...]:
    if arguments.run_init:
        if arguments.start_date or arguments.end_date:
            raise SystemExit("use either --run-init or --start-date/--end-date, not both")
        runs = tuple(_parse_run_init(value) for value in arguments.run_init)
    else:
        if not arguments.start_date or not arguments.end_date:
            raise SystemExit("provide --run-init or both --start-date and --end-date")
        start = date.fromisoformat(arguments.start_date)
        end = date.fromisoformat(arguments.end_date)
        runs = iter_run_initialisations(
            start,
            end,
            cycles_utc=_parse_csv_ints(arguments.cycles),
        )
        if arguments.include_prior_18z:
            prior = datetime.combine(
                start - timedelta(days=1), datetime.min.time(), UTC
            ).replace(hour=18)
            runs = (prior, *runs)
    if any(run.year >= 2025 for run in runs):
        raise SystemExit("2025 weather is sealed; request rejected before remote I/O")
    return runs


def _uses_next_day_18z_schedule(
    run: datetime,
    arguments: argparse.Namespace,
) -> bool:
    return (
        arguments.transport == "mixed"
        and not arguments.run_init
        and run.hour == 18
    )


def _operating_date(run: datetime, arguments: argparse.Namespace) -> date:
    if arguments.operating_date:
        result = date.fromisoformat(arguments.operating_date)
    elif run.year == 2021 or _uses_next_day_18z_schedule(run, arguments):
        result = run.date() + timedelta(days=1)
    else:
        result = run.date()
    if result.year >= 2025:
        raise SystemExit("2025 operating weather is sealed")
    return result


def _features(arguments: argparse.Namespace) -> tuple[str, ...]:
    if arguments.features:
        result = tuple(
            dict.fromkeys(
                value.strip() for value in arguments.features.split(",") if value.strip()
            )
        )
        if not result:
            raise SystemExit("--features cannot be empty")
        return result
    if arguments.transport == "mixed":
        return V1_MIXED_FEATURES
    if arguments.transport == "zarr":
        return ("air_temperature_2m_k",)
    return HRRR_DEFAULT_FEATURES if arguments.source == "hrrr" else NBM_DEFAULT_FEATURES


def _usable_forecast_hours(
    run: datetime,
    arguments: argparse.Namespace,
    requested_hours: Sequence[int],
) -> tuple[int, ...]:
    hours = tuple(
        int(hour)
        for hour in requested_hours
        if (run + timedelta(hours=int(hour))).year in (2022, 2023, 2024)
    )
    if _uses_next_day_18z_schedule(run, arguments) or run.year == 2021:
        hours = tuple(hour for hour in hours if 6 <= hour <= 29)
    return hours


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source", choices=tuple(WEATHER_SOURCES), default="hrrr")
    parser.add_argument(
        "--transport",
        choices=("mixed", "zarr", "grib-range"),
        default="mixed",
        help="formal V1 uses mixed; zarr and grib-range are explicit probes/ablations",
    )
    parser.add_argument("--run-init", action="append")
    parser.add_argument("--operating-date")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--cycles", default="0,6,12,18")
    parser.add_argument("--forecast-hours", default="1-48")
    parser.add_argument("--features")
    parser.add_argument("--include-prior-18z", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--summary-path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    runs = _build_runs(arguments)
    if arguments.max_runs is not None:
        if arguments.max_runs < 1:
            raise SystemExit("--max-runs must be positive")
        runs = runs[: arguments.max_runs]
    hours = _parse_csv_ints(arguments.forecast_hours)
    if arguments.workers < 1:
        raise SystemExit("--workers must be positive")
    if arguments.transport in {"mixed", "zarr"} and arguments.source != "hrrr":
        raise SystemExit("mixed/zarr transports are HRRR-only; NBM stays independent")
    if arguments.transport != "grib-range" and arguments.probe_only:
        raise SystemExit("--probe-only is available only with --transport grib-range")
    selected_features = _features(arguments)
    if arguments.transport == "mixed" and selected_features != V1_MIXED_FEATURES:
        raise SystemExit("formal mixed transport fixes TMP+DSWRF; do not change --features")
    if arguments.transport == "zarr":
        unknown = set(selected_features) - set(HRRR_ZARR_VARIABLES)
        if unknown:
            raise SystemExit(f"unsupported HRRR Zarr feature(s): {sorted(unknown)}")

    cache_root = arguments.cache_root.resolve()
    if arguments.transport == "mixed" and not arguments.run_init:
        # The 18Z extended run is the coherent origin-0 path for the next
        # operating day.  Never construct the 2024-12-31 18Z -> 2025 target.
        runs = tuple(
            run
            for run in runs
            if not (
                _uses_next_day_18z_schedule(run, arguments)
                and (run.date() + timedelta(days=1)).year >= 2025
            )
        )
    if arguments.transport == "grib-range":
        jobs: list[tuple[datetime, int | None]] = [
            (run, hour) for run in runs for hour in hours
        ]
        if arguments.max_objects is not None:
            if arguments.max_objects < 1:
                raise SystemExit("--max-objects must be positive")
            jobs = jobs[: arguments.max_objects]
    else:
        jobs = [(run, None) for run in runs]

    def execute(job: tuple[datetime, int | None]) -> dict[str, object]:
        run, one_hour = job
        store = WorldWeatherStore(cache_root)
        operating = _operating_date(run, arguments)
        usable_hours = _usable_forecast_hours(run, arguments, hours)
        zarr_hours = tuple(
            hour
            for hour in range(1, 49)
            if (run + timedelta(hours=hour)).year in (2022, 2023, 2024)
        )
        if arguments.transport == "mixed":
            existing = None
            if not arguments.no_resume:
                existing = store.find_complete_run(
                    source="hrrr",
                    run_init=run,
                    forecast_hours=usable_hours,
                    features=V1_MIXED_FEATURES,
                )
            path = existing or HrrrMixedMaterializer(cache_root).materialize_run(
                store=store,
                run_init=run,
                operating_date=operating,
                forecast_hours=usable_hours,
            )
            materialized = pd.read_parquet(
                path,
                columns=[
                    "feature",
                    "transport",
                    "transport_substitution_reason",
                ],
            )
            feature_transports = {
                feature: sorted(
                    materialized.loc[
                        materialized["feature"].eq(feature), "transport"
                    ].unique()
                )
                for feature in V1_MIXED_FEATURES
            }
            return {
                "source": "hrrr",
                "transport": "same-run HRRR feature-adaptive mixed",
                "feature_transports": feature_transports,
                "transport_substitution_reasons": sorted(
                    set(materialized["transport_substitution_reason"].astype(str))
                ),
                "run_init_utc": run.isoformat().replace("+00:00", "Z"),
                "forecast_hours": list(usable_hours),
                "features": list(V1_MIXED_FEATURES),
                "materialized_path": str(path),
                "resumed": existing is not None,
            }
        if arguments.transport == "zarr":
            existing = None
            if not arguments.no_resume:
                existing = store.find_complete_run(
                    source="hrrr",
                    run_init=run,
                    forecast_hours=zarr_hours,
                    features=selected_features,
                )
            path = existing or HrrrZarrMaterializer(cache_root).materialize_run(
                store=store,
                run_init=run,
                operating_date=operating,
                features=selected_features,
                forecast_hours=zarr_hours,
            )
            return {
                "source": "hrrr",
                "transport": "hrrrzarr_150x150_chunk",
                "run_init_utc": run.isoformat().replace("+00:00", "Z"),
                "forecast_hours": list(zarr_hours),
                "features": list(selected_features),
                "materialized_path": str(path),
                "resumed": existing is not None,
            }

        assert one_hour is not None
        materializer = WeatherRangeMaterializer(
            cache_root,
            decoder=None if arguments.probe_only else EccodesPointDecoder(),
        )
        if arguments.probe_only:
            records = materializer.acquire(
                source=arguments.source,
                run_init=run,
                forecast_hour=one_hour,
                features=selected_features,
                sites=tuple(NYISO_WEATHER_SITES.values()),
            )
            return {
                "source": arguments.source,
                "transport": "grib2_idx_byte_range_probe",
                "run_init_utc": run.isoformat().replace("+00:00", "Z"),
                "forecast_hour": one_hour,
                "features": list(selected_features),
                "fragment_record_ids": [record.record_id for record in records],
                "downloaded_range_bytes": sum(record.byte_count for record in records),
                "visible_at_utc": max(record.visible_at_utc for record in records),
            }
        path = materializer.materialize(
            store=store,
            source=arguments.source,
            run_init=run,
            forecast_hour=one_hour,
            features=selected_features,
            sites=tuple(NYISO_WEATHER_SITES.values()),
        )
        return {
            "source": arguments.source,
            "transport": "grib2_idx_byte_range",
            "run_init_utc": run.isoformat().replace("+00:00", "Z"),
            "forecast_hour": one_hour,
            "features": list(selected_features),
            "materialized_path": str(path),
        }

    completed: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {executor.submit(execute, job): job for job in jobs}
        for future in as_completed(futures):
            run, forecast_hour = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:  # noqa: BLE001 - persisted batch failure evidence
                failed.append(
                    {
                        "run_init_utc": run.isoformat().replace("+00:00", "Z"),
                        "forecast_hour": forecast_hour,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    completed.sort(
        key=lambda item: (
            str(item["run_init_utc"]),
            int(item.get("forecast_hour", -1)),
        )
    )
    failed.sort(
        key=lambda item: (
            str(item["run_init_utc"]),
            int(item["forecast_hour"] or -1),
        )
    )
    summary = {
        "schema": "dcvpp.vpp_world_moe.weather-materialization-run.v2",
        "mode": "probe_only" if arguments.probe_only else "materialize",
        "opened_final_test_2025": False,
        "cache_root": str(cache_root),
        "source": arguments.source,
        "transport": arguments.transport,
        "workers": arguments.workers,
        "resume": not arguments.no_resume,
        "site_zones": list(NYISO_WEATHER_SITES),
        "object_count": len(completed),
        "objects": completed,
        "failed_job_count": len(failed),
        "failed_jobs": failed,
        "status": "COMPLETE" if not failed else "PARTIAL_FAILED",
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if arguments.summary_path:
        target = arguments.summary_path.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
