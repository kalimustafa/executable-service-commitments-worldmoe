"""Runtime-only, verified NPZ I/O acceleration for the DART pipeline.

The formal DART modules use ``numpy.savez_compressed`` for every archive.  That
is useful for repetitive arrays but spends substantial CPU time trying to
compress large floating-point scenario arrays whose sampled DEFLATE ratio is
close to one.  This compatibility context leaves the frozen source files
untouched and temporarily replaces only their module-level ``_atomic_npz``
writers.

Large, high-entropy payloads are written with ``numpy.savez`` (ZIP_STORED);
small or demonstrably compressible payloads retain ``numpy.savez_compressed``.
Every temporary archive is fsynced and then re-opened with ``allow_pickle=False``
to verify the exact key set, shape, dtype, and byte-level content hash before a
same-directory atomic replace.  Object arrays are rejected because the formal
readers prohibit pickle-backed NPZ members.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
from time import perf_counter
import tempfile
import threading
from typing import Any, Iterator, Mapping
import zipfile
import zlib

import numpy as np


POLICY_ID = "sampled_entropy_adaptive_verified_atomic_npz_v1"
SCHEMA_VERSION = "dcvpp.dart_npz_efficiency_compatibility.v1"

# These hashes cover only the function being replaced, not an entire module.
# Other compatibility work can therefore extend the surrounding modules while
# this layer remains fail-closed about the exact serialization boundary.
EXPECTED_ATOMIC_NPZ_SOURCE_SHA256 = {
    "dcvpp.experiments.dart_training": (
        "3b29a899901cf1a55d7e0f9b9ce7279ffcf245194810615d59fedf8cad356b72"
    ),
    "dcvpp.experiments.dart_forecast_oos": (
        "3b29a899901cf1a55d7e0f9b9ce7279ffcf245194810615d59fedf8cad356b72"
    ),
    "dcvpp.experiments.dart_real_evidence": (
        "de995724d97e0cf0b648b9a42f30138ec1ecd83175db843d774feeeab7b5b713"
    ),
}


class DARTNPZEfficiencyCompatibilityError(RuntimeError):
    """Raised when accelerated NPZ serialization cannot be proven exact."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256(function: Any) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        "w+b",
        prefix=".dcvpp_npz_ledger_",
        suffix=".json.tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_arrays(arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not arrays:
        raise DARTNPZEfficiencyCompatibilityError(
            "an NPZ artifact must contain at least one named array"
        )
    normalized: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if not isinstance(key, str) or not key or "/" in key or "\\" in key:
            raise DARTNPZEfficiencyCompatibilityError(
                f"invalid NPZ member key: {key!r}"
            )
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise DARTNPZEfficiencyCompatibilityError(
                f"object array is incompatible with allow_pickle=False: {key}"
            )
        # Preserve scalar shape.  np.ascontiguousarray would turn a 0-D array
        # into shape (1,), whereas a 0-D array is already C-contiguous.
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        normalized[key] = array
    return normalized


def _array_content_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    dtype_contract = {
        "str": array.dtype.str,
        "descr": array.dtype.descr if array.dtype.fields is not None else None,
    }
    digest.update(
        json.dumps(dtype_contract, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    if array.nbytes:
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _array_signatures(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "dtype_descr": (
                array.dtype.descr if array.dtype.fields is not None else None
            ),
            "nbytes": int(array.nbytes),
            "content_sha256": _array_content_sha256(array),
        }
        for key, array in sorted(arrays.items())
    }


def _signature_panel_sha256(signatures: Mapping[str, Any]) -> str:
    payload = json.dumps(
        signatures,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sample_bytes(
    arrays: Mapping[str, np.ndarray], *, maximum_sample_bytes: int
) -> bytes:
    """Collect bounded, spatially distributed contiguous byte samples."""

    nonempty = [array for array in arrays.values() if array.nbytes]
    if not nonempty:
        return b""
    per_array = max(1, int(maximum_sample_bytes) // len(nonempty))
    samples: list[bytes] = []
    for array in nonempty:
        # ``ndarray.view(uint8)`` rejects 0-D arrays when itemsize changes.
        # Reshaping to one element first preserves every byte while exposing a
        # one-dimensional buffer for bounded entropy sampling.
        view = array.reshape(-1).view(np.uint8).reshape(-1)
        budget = min(int(view.size), per_array)
        if budget == int(view.size):
            samples.append(view.tobytes())
            continue
        chunk_count = min(4, budget)
        chunk_size = max(1, budget // chunk_count)
        maximum_start = max(0, int(view.size) - chunk_size)
        for index in range(chunk_count):
            if chunk_count == 1:
                start = 0
            else:
                start = (maximum_start * index) // (chunk_count - 1)
            samples.append(view[start : start + chunk_size].tobytes())
    result = b"".join(samples)
    return result[:maximum_sample_bytes]


def _compression_probe_ratio(sample: bytes) -> float:
    if not sample:
        return 0.0
    return float(len(zlib.compress(sample, level=1)) / len(sample))


def _validate_archive(
    path: Path,
    *,
    expected_arrays: Mapping[str, np.ndarray],
    expected_signatures: Mapping[str, Mapping[str, Any]],
    storage_mode: str,
) -> dict[str, Any]:
    expected_keys = set(expected_arrays)
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        member_names = [info.filename for info in infos]
        expected_names = {f"{key}.npy" for key in expected_keys}
        if set(member_names) != expected_names or len(member_names) != len(expected_names):
            raise DARTNPZEfficiencyCompatibilityError(
                "NPZ ZIP member inventory changed during serialization"
            )
        expected_compression = (
            zipfile.ZIP_STORED
            if storage_mode == "uncompressed"
            else zipfile.ZIP_DEFLATED
        )
        if any(info.compress_type != expected_compression for info in infos):
            raise DARTNPZEfficiencyCompatibilityError(
                f"NPZ member compression does not match {storage_mode} policy"
            )

    loaded_signatures: dict[str, dict[str, Any]] = {}
    with np.load(path, allow_pickle=False) as loaded:
        loaded_keys = list(loaded.files)
        if set(loaded_keys) != expected_keys or len(loaded_keys) != len(expected_keys):
            raise DARTNPZEfficiencyCompatibilityError(
                "NPZ key inventory changed during serialization"
            )
        for key in sorted(expected_keys):
            actual = loaded[key]
            expected = expected_arrays[key]
            if actual.shape != expected.shape:
                raise DARTNPZEfficiencyCompatibilityError(
                    f"NPZ shape changed for {key}: {actual.shape} != {expected.shape}"
                )
            if actual.dtype != expected.dtype:
                raise DARTNPZEfficiencyCompatibilityError(
                    f"NPZ dtype changed for {key}: {actual.dtype} != {expected.dtype}"
                )
            signature = _array_signatures({key: actual})[key]
            if signature != dict(expected_signatures[key]):
                raise DARTNPZEfficiencyCompatibilityError(
                    f"NPZ content hash changed for {key}"
                )
            loaded_signatures[key] = signature

    return {
        "validated_key_count": len(expected_keys),
        "validated_keys": sorted(expected_keys),
        "loaded_signature_panel_sha256": _signature_panel_sha256(
            loaded_signatures
        ),
        "zip_member_count": len(expected_keys),
        "zip_storage_mode_verified": True,
    }


def _fsync_directory_best_effort(directory: Path) -> bool:
    if os.name == "nt":
        return False
    descriptor: int | None = None
    try:
        descriptor = os.open(directory, os.O_RDONLY)
        os.fsync(descriptor)
        return True
    except OSError:
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)


@contextmanager
def npz_io_efficiency_compatibility(
    *,
    ledger_path: Path | None = None,
    minimum_uncompressed_nbytes: int = 262_144,
    high_entropy_ratio: float = 0.85,
    entropy_sample_bytes: int = 1_048_576,
    ledger_checkpoint_interval: int = 64,
) -> Iterator[dict[str, Any]]:
    """Temporarily install verified adaptive NPZ writers in DART modules.

    The context is independent of the AMP, scheduler-failure, and LightGBM
    compatibility contexts and can be composed with them using ``ExitStack``.
    """

    if minimum_uncompressed_nbytes < 1:
        raise ValueError("minimum_uncompressed_nbytes must be positive")
    if not 0.0 < high_entropy_ratio <= 1.0:
        raise ValueError("high_entropy_ratio must be inside (0, 1]")
    if entropy_sample_bytes < 1024:
        raise ValueError("entropy_sample_bytes must be at least 1024")
    if ledger_checkpoint_interval < 1:
        raise ValueError("ledger_checkpoint_interval must be positive")

    from dcvpp.experiments import (
        dart_forecast_oos,
        dart_real_evidence,
        dart_training,
    )

    modules = (dart_training, dart_forecast_oos, dart_real_evidence)
    originals: dict[str, Any] = {}
    module_records: list[dict[str, Any]] = []
    for module in modules:
        name = str(module.__name__)
        original = module._atomic_npz
        source_sha256 = _source_sha256(original)
        expected = EXPECTED_ATOMIC_NPZ_SOURCE_SHA256[name]
        if source_sha256 != expected:
            raise DARTNPZEfficiencyCompatibilityError(
                f"frozen {name}._atomic_npz source changed: {source_sha256}"
            )
        module_path = Path(module.__file__).resolve()
        originals[name] = original
        module_records.append(
            {
                "module": name,
                "path": module_path.as_posix(),
                "module_sha256_at_install": _sha256_file(module_path),
                "atomic_npz_source_sha256": source_sha256,
            }
        )

    destination_ledger = Path(ledger_path).resolve() if ledger_path else None
    lock = threading.RLock()
    recent_events: deque[dict[str, Any]] = deque(maxlen=12)
    sampled_events: list[dict[str, Any]] = []
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ACTIVE",
        "policy_id": POLICY_ID,
        "installed_at_utc": _utc_now(),
        "scope": "DART NPZ serialization only",
        "patched_modules": module_records,
        "selection_model_data_and_numeric_values_changed": False,
        "minimum_uncompressed_nbytes": int(minimum_uncompressed_nbytes),
        "high_entropy_probe_ratio_threshold": float(high_entropy_ratio),
        "entropy_sample_bytes": int(entropy_sample_bytes),
        "validation_contract": [
            "exact_key_inventory",
            "exact_shape",
            "exact_dtype",
            "exact_byte_content_sha256",
            "allow_pickle_false_load",
            "zip_storage_mode",
        ],
        "atomic_same_directory_replace": True,
        "temporary_file_fsync_before_replace": True,
        "event_count": 0,
        "successful_event_count": 0,
        "failed_event_count": 0,
        "uncompressed_event_count": 0,
        "compressed_event_count": 0,
        "total_input_nbytes": 0,
        "total_output_bytes": 0,
        "total_write_seconds": 0.0,
        "total_validation_seconds": 0.0,
        "event_chain_sha256": hashlib.sha256(b"").hexdigest(),
        "sampled_events": sampled_events,
        "recent_events": [],
    }

    def write_ledger() -> None:
        if destination_ledger is not None:
            record["recent_events"] = list(recent_events)
            _atomic_json(destination_ledger, record)

    def record_event(event: dict[str, Any]) -> None:
        with lock:
            canonical = json.dumps(
                event,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            previous = bytes.fromhex(str(record["event_chain_sha256"]))
            record["event_chain_sha256"] = hashlib.sha256(
                previous + canonical
            ).hexdigest()
            record["event_count"] = int(record["event_count"]) + 1
            if event["status"] == "COMPLETE":
                record["successful_event_count"] = (
                    int(record["successful_event_count"]) + 1
                )
                mode_key = f"{event['storage_mode']}_event_count"
                record[mode_key] = int(record[mode_key]) + 1
                record["total_input_nbytes"] = int(record["total_input_nbytes"]) + int(
                    event["input_nbytes"]
                )
                record["total_output_bytes"] = int(record["total_output_bytes"]) + int(
                    event["output_bytes"]
                )
                record["total_write_seconds"] = float(
                    record["total_write_seconds"]
                ) + float(event["write_seconds"])
                record["total_validation_seconds"] = float(
                    record["total_validation_seconds"]
                ) + float(event["validation_seconds"])
            else:
                record["failed_event_count"] = int(record["failed_event_count"]) + 1
            if len(sampled_events) < 4:
                sampled_events.append(event)
            recent_events.append(event)
            if int(record["event_count"]) % ledger_checkpoint_interval == 0:
                write_ledger()

    def make_writer(module_name: str) -> Any:
        def accelerated_atomic_npz(path: Path, **arrays: Any) -> None:
            destination = Path(path).resolve()
            event: dict[str, Any] = {
                "recorded_at_utc": _utc_now(),
                "module": module_name,
                "destination_path": destination.as_posix(),
                "status": "STARTED",
            }
            temporary: Path | None = None
            try:
                if destination_ledger is not None and destination == destination_ledger:
                    raise DARTNPZEfficiencyCompatibilityError(
                        "NPZ destination collides with compatibility ledger"
                    )
                normalized = _normalize_arrays(arrays)
                signatures = _array_signatures(normalized)
                signature_panel_sha256 = _signature_panel_sha256(signatures)
                input_nbytes = sum(array.nbytes for array in normalized.values())
                sample = _sample_bytes(
                    normalized, maximum_sample_bytes=entropy_sample_bytes
                )
                probe_ratio = _compression_probe_ratio(sample)
                use_uncompressed = (
                    input_nbytes >= minimum_uncompressed_nbytes
                    and probe_ratio >= high_entropy_ratio
                )
                storage_mode = "uncompressed" if use_uncompressed else "compressed"
                destination.parent.mkdir(parents=True, exist_ok=True)

                write_started = perf_counter()
                with tempfile.NamedTemporaryFile(
                    "w+b",
                    prefix=".dcvpp_npz_",
                    suffix=".npz.tmp",
                    dir=destination.parent,
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    if use_uncompressed:
                        np.savez(handle, **normalized)
                    else:
                        np.savez_compressed(handle, **normalized)
                    handle.flush()
                    os.fsync(handle.fileno())
                write_seconds = perf_counter() - write_started

                validation_started = perf_counter()
                validation = _validate_archive(
                    temporary,
                    expected_arrays=normalized,
                    expected_signatures=signatures,
                    storage_mode=storage_mode,
                )
                validation_seconds = perf_counter() - validation_started
                output_bytes = temporary.stat().st_size
                os.replace(temporary, destination)
                temporary = None
                directory_fsynced = _fsync_directory_best_effort(destination.parent)

                event.update(
                    {
                        "status": "COMPLETE",
                        "storage_mode": storage_mode,
                        "selection_reason": (
                            "large_and_high_entropy"
                            if use_uncompressed
                            else "small_or_compressible"
                        ),
                        "input_nbytes": int(input_nbytes),
                        "output_bytes": int(output_bytes),
                        "archive_to_input_ratio": float(
                            output_bytes / max(input_nbytes, 1)
                        ),
                        "entropy_probe_bytes": len(sample),
                        "entropy_probe_compressed_ratio": float(probe_ratio),
                        "input_signature_panel_sha256": signature_panel_sha256,
                        "write_seconds": float(write_seconds),
                        "validation_seconds": float(validation_seconds),
                        "file_fsync_complete": True,
                        "directory_fsync_complete": directory_fsynced,
                        "same_directory_atomic_replace_complete": True,
                        **validation,
                    }
                )
            except Exception as exc:
                event.update(
                    {
                        "status": "FAILED",
                        "storage_mode": event.get("storage_mode", "unknown"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                record_event(event)
                raise DARTNPZEfficiencyCompatibilityError(
                    f"verified NPZ write failed for {destination}"
                ) from exc
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            record_event(event)

        accelerated_atomic_npz.__name__ = "_atomic_npz"
        accelerated_atomic_npz.__qualname__ = f"{module_name}._atomic_npz_compat"
        return accelerated_atomic_npz

    wrappers = {name: make_writer(name) for name in originals}
    write_ledger()
    for module in modules:
        module._atomic_npz = wrappers[module.__name__]

    context_error: BaseException | None = None
    try:
        yield record
    except BaseException as exc:
        context_error = exc
        raise
    finally:
        changed_modules: list[str] = []
        for module in modules:
            name = module.__name__
            if module._atomic_npz is not wrappers[name]:
                changed_modules.append(name)
            module._atomic_npz = originals[name]
        record["status"] = "CLOSED"
        record["closed_at_utc"] = _utc_now()
        record["writer_replaced_during_context"] = changed_modules
        write_ledger()
        if changed_modules and context_error is None:
            raise DARTNPZEfficiencyCompatibilityError(
                "DART NPZ writer changed during compatibility context: "
                + ", ".join(changed_modules)
            )


__all__ = [
    "DARTNPZEfficiencyCompatibilityError",
    "EXPECTED_ATOMIC_NPZ_SOURCE_SHA256",
    "POLICY_ID",
    "SCHEMA_VERSION",
    "npz_io_efficiency_compatibility",
]
