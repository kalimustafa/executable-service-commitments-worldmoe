"""Narrow Windows Unicode-path compatibility for DART LightGBM checkpoints.

LightGBM's native ``Booster.save_model`` can fail when its destination contains
non-ASCII Windows path components.  The formal DART adapter is frozen, so this
module changes no training code or model setting.  Inside the context manager
only, it replaces :func:`dcvpp.experiments.dart_training._save_booster` with an
equivalent atomic writer whose *native* LightGBM write goes to an ASCII-only
temporary path.  Python then copies the exact bytes to a same-filesystem
temporary file beside the requested destination and atomically replaces it.

The compatibility fails closed if the frozen adapter or save function changed,
no writable ASCII temporary directory is available, LightGBM emits no bytes,
or any source/staging/destination size or SHA-256 comparison differs.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator


FROZEN_DART_TRAINING_SHA256 = (
    "f280035c6a50ad4ccd952f7175d51e05bbc6560475b78cb50383bf9bb872db18"
)
FROZEN_SAVE_BOOSTER_SOURCE_SHA256 = (
    "95a181e9f185a51b6bfa62d705ba30b00f021d72c59c357e0fba4b7d39fd53f9"
)
POLICY_ID = "lightgbm_ascii_native_save_python_atomic_unicode_copy_v1"


class DARTLightGBMPathCompatibilityError(RuntimeError):
    """Raised when the narrow Unicode-path compatibility contract fails."""


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


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _ascii_candidates(explicit: Path | None) -> tuple[Path, ...]:
    if explicit is not None:
        return (Path(explicit).expanduser().resolve(),)
    candidates = [Path(tempfile.gettempdir()).resolve()]
    if os.name == "nt":
        system_root = os.environ.get("SystemRoot")
        if system_root:
            candidates.append((Path(system_root) / "Temp").resolve())
        candidates.append(Path("C:/Windows/Temp").resolve())
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _create_ascii_session_directory(explicit: Path | None) -> Path:
    failures: list[str] = []
    for root in _ascii_candidates(explicit):
        if not str(root).isascii():
            failures.append(f"non-ASCII temporary root: {root}")
            continue
        try:
            if explicit is not None:
                root.mkdir(parents=True, exist_ok=True)
            if not root.is_dir():
                failures.append(f"temporary root is not a directory: {root}")
                continue
            session = Path(tempfile.mkdtemp(prefix="dcvpp_lgbm_", dir=root)).resolve()
        except OSError as exc:
            failures.append(f"unwritable temporary root {root}: {exc}")
            continue
        if not str(session).isascii() or session.parent != root:
            shutil.rmtree(session, ignore_errors=True)
            failures.append(f"temporary session path is not ASCII-contained: {session}")
            continue
        return session
    raise DARTLightGBMPathCompatibilityError(
        "no writable ASCII-only temporary directory is available: "
        + "; ".join(failures)
    )


def _copy_verified_atomic(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file():
        raise DARTLightGBMPathCompatibilityError(
            "LightGBM native save did not create a model file"
        )
    source_size = source.stat().st_size
    if source_size <= 0:
        raise DARTLightGBMPathCompatibilityError(
            "LightGBM native save created an empty model file"
        )
    source_sha256_before = _sha256_file(source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        prefix=".dcvpp_lgbm_",
        suffix=destination.suffix + ".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        staging = Path(handle.name)
        with source.open("rb") as source_handle:
            shutil.copyfileobj(source_handle, handle, length=1024 * 1024)
        handle.flush()
        os.fsync(handle.fileno())

    try:
        source_sha256_after = _sha256_file(source)
        staging_size = staging.stat().st_size
        staging_sha256 = _sha256_file(staging)
        if (
            source_sha256_before != source_sha256_after
            or staging_size != source_size
            or staging_sha256 != source_sha256_before
        ):
            raise DARTLightGBMPathCompatibilityError(
                "LightGBM model bytes changed during the verified Python copy"
            )
        os.replace(staging, destination)
        destination_size = destination.stat().st_size
        destination_sha256 = _sha256_file(destination)
        if (
            destination_size != source_size
            or destination_sha256 != source_sha256_before
        ):
            raise DARTLightGBMPathCompatibilityError(
                "atomic destination does not match the native LightGBM bytes"
            )
        return {
            "byte_count": int(source_size),
            "source_sha256": source_sha256_before,
            "staging_sha256": staging_sha256,
            "destination_sha256": destination_sha256,
        }
    finally:
        staging.unlink(missing_ok=True)


@contextmanager
def lightgbm_unicode_path_compatibility(
    *,
    ledger_path: Path | None = None,
    ascii_temp_root: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Temporarily install exact-byte LightGBM Unicode destination support.

    ``ascii_temp_root`` is mainly useful for deterministic tests.  When omitted,
    an existing writable ASCII-only system temporary directory is selected.
    """

    from dcvpp.experiments import dart_training

    module_path = Path(dart_training.__file__).resolve()
    module_sha256 = _sha256_file(module_path)
    if module_sha256 != FROZEN_DART_TRAINING_SHA256:
        raise DARTLightGBMPathCompatibilityError(
            f"frozen DART training adapter SHA changed: {module_sha256}"
        )
    original = dart_training._save_booster
    original_source_sha256 = _source_sha256(original)
    if original_source_sha256 != FROZEN_SAVE_BOOSTER_SOURCE_SHA256:
        raise DARTLightGBMPathCompatibilityError(
            "frozen DART LightGBM save function changed: "
            f"{original_source_sha256}"
        )

    destination_ledger = Path(ledger_path).resolve() if ledger_path else None
    session_temp = _create_ascii_session_directory(ascii_temp_root)
    events: list[dict[str, Any]] = []
    record: dict[str, Any] = {
        "schema_version": "dcvpp.dart_lightgbm_path_compatibility.v1",
        "status": "ACTIVE",
        "policy_id": POLICY_ID,
        "frozen_dart_training_path": module_path.as_posix(),
        "frozen_dart_training_sha256": module_sha256,
        "frozen_save_booster_source_sha256": original_source_sha256,
        "installed_at_utc": _utc_now(),
        "scope": "LightGBM model serialization path only",
        "allowed_change": (
            "native save to ASCII temporary path followed by verified Python "
            "copy and same-filesystem atomic destination replace"
        ),
        "model_data_fit_predictions_and_selection_policy_changed": False,
        "native_save_path_ascii_required": True,
        "exact_byte_sha256_verification_required": True,
        "events": events,
    }

    def write_ledger() -> None:
        if destination_ledger is not None:
            _atomic_json(destination_ledger, record)

    def save_booster(booster: Any, path: Path) -> None:
        destination = Path(path)
        native_file: Path | None = None
        event: dict[str, Any] = {
            "recorded_at_utc": _utc_now(),
            "policy_id": POLICY_ID,
            "destination_path": destination.resolve().as_posix(),
            "destination_contains_non_ascii": not str(destination.resolve()).isascii(),
            "native_write_path_ascii": True,
            "python_binary_copy": True,
            "same_filesystem_atomic_replace": True,
            "status": "STARTED",
        }
        try:
            if not callable(getattr(booster, "save_model", None)):
                raise DARTLightGBMPathCompatibilityError(
                    "LightGBM booster has no callable save_model"
                )
            with tempfile.NamedTemporaryFile(
                "wb", suffix=".txt", dir=session_temp, delete=False
            ) as handle:
                native_file = Path(handle.name).resolve()
            if not str(native_file).isascii() or native_file.parent != session_temp:
                raise DARTLightGBMPathCompatibilityError(
                    "native LightGBM save path escaped the ASCII session directory"
                )
            booster.save_model(str(native_file))
            evidence = _copy_verified_atomic(native_file, destination)
            event.update(evidence)
            event["status"] = "COMPLETE"
        except Exception as exc:
            event["status"] = "FAILED"
            event["error_type"] = type(exc).__name__
            event["error"] = str(exc)
            raise DARTLightGBMPathCompatibilityError(
                f"LightGBM Unicode-path compatibility failed for {destination}"
            ) from exc
        finally:
            if native_file is not None:
                native_file.unlink(missing_ok=True)
            events.append(event)
            record["event_count"] = len(events)
            write_ledger()

    write_ledger()
    dart_training._save_booster = save_booster
    context_error: BaseException | None = None
    try:
        yield record
    except BaseException as exc:
        context_error = exc
        raise
    finally:
        replaced_during_context = dart_training._save_booster is not save_booster
        dart_training._save_booster = original
        shutil.rmtree(session_temp, ignore_errors=True)
        record["status"] = "CLOSED"
        record["closed_at_utc"] = _utc_now()
        record["event_count"] = len(events)
        record["save_function_replaced_during_context"] = replaced_during_context
        write_ledger()
        if replaced_during_context and context_error is None:
            raise DARTLightGBMPathCompatibilityError(
                "DART LightGBM save function was replaced during compatibility context"
            )


__all__ = [
    "DARTLightGBMPathCompatibilityError",
    "FROZEN_DART_TRAINING_SHA256",
    "FROZEN_SAVE_BOOSTER_SOURCE_SHA256",
    "POLICY_ID",
    "lightgbm_unicode_path_compatibility",
]
