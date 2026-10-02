from __future__ import annotations

from datetime import timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from dcvpp.data.gates import GateFailure


NLR295_REQUIRED_TZ_COLUMNS = ("submit_time_tz", "start_time_tz", "end_time_tz")
NLR300_LOCAL_TIMEZONE = "America/Denver"
NLR300_STRICT_TIMEZONE_POLICY = "strict_authoritative"
NLR300_UTC_SOURCE_LOCAL_ANALYSIS_POLICY = "user_authorized_utc_source_local_analysis"
NLR300_LOCAL_TIMEZONE_POLICY = "user_authorized_local_timezone"
NLR300_SOURCE_CLOCK_ASSUMPTION = "UTC"
NLR300_DEFAULT_AMBIGUOUS_POLICY = "standard_time"
NLR300_DEFAULT_NONEXISTENT_POLICY = "mark_invalid"


def require_nlr295_tz_columns(columns: list[str] | pd.Index) -> None:
    missing = [column for column in NLR295_REQUIRED_TZ_COLUMNS if column not in columns]
    if missing:
        raise GateFailure(f"NLR 295 authoritative timezone columns missing: {missing}")


def _validate_timezone_name(value: str) -> str:
    try:
        ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
        raise GateFailure(f"unknown IANA timezone: {value}") from exc
    return value


def resolve_nlr300_timezone(
    metadata: dict[str, Any],
    *,
    policy: str = NLR300_STRICT_TIMEZONE_POLICY,
    assumed_timezone: str = NLR300_LOCAL_TIMEZONE,
    source_clock_assumption: str = NLR300_SOURCE_CLOCK_ASSUMPTION,
) -> str:
    """Resolve the NLR300 timezone under an explicit evidence policy.

    ``strict_authoritative`` retains the original publisher-metadata gate.
    ``user_authorized_utc_source_local_analysis`` assumes the naive source
    clock is UTC and uses the configured civil timezone only for analysis.
    ``user_authorized_local_timezone`` remains an optional direct-wall-clock
    sensitivity and must not be described as publisher evidence.
    """
    if policy == NLR300_UTC_SOURCE_LOCAL_ANALYSIS_POLICY:
        _validate_timezone_name(assumed_timezone)
        return _validate_timezone_name(source_clock_assumption)
    if policy == NLR300_LOCAL_TIMEZONE_POLICY:
        return _validate_timezone_name(assumed_timezone)
    if policy != NLR300_STRICT_TIMEZONE_POLICY:
        raise ValueError(f"unsupported NLR300 timezone policy: {policy}")
    candidates = [metadata.get("timezone"), metadata.get("time_zone"), metadata.get("tz")]
    value = next((str(candidate).strip() for candidate in candidates if candidate), None)
    if not value or value.lower() in {"unknown", "na", "none"}:
        raise GateFailure("BLOCKED_TIMEZONE_NLR300")
    return _validate_timezone_name(value)


def nlr300_utc_source_local_analysis_policy_record(
    *,
    source_clock_assumption: str = NLR300_SOURCE_CLOCK_ASSUMPTION,
    analysis_timezone: str = NLR300_LOCAL_TIMEZONE,
) -> dict[str, Any]:
    """Describe the primary, user-authorized NLR300 clock assumption."""
    _validate_timezone_name(source_clock_assumption)
    _validate_timezone_name(analysis_timezone)
    if source_clock_assumption != "UTC":
        raise ValueError("the primary NLR300 source-clock assumption must be UTC")
    return {
        "policy": NLR300_UTC_SOURCE_LOCAL_ANALYSIS_POLICY,
        "evidence_role": "assumed",
        "source_clock_assumption": source_clock_assumption,
        "analysis_timezone": analysis_timezone,
        "publisher_metadata_claimed": False,
        "publisher_metadata_used": False,
        "preserve_source_timestamp": True,
        "timestamp_utc_rule": "tz_localize_UTC",
        "timestamp_local_rule": f"tz_convert_{analysis_timezone.replace('/', '_')}",
        "dst_semantics": (
            "local time is derived from a unique UTC instant; UTC offsets distinguish repeated "
            "fall-back wall times and no nonexistent local instant is fabricated"
        ),
    }


def nlr300_timezone_policy_record(
    *,
    timezone_name: str = NLR300_LOCAL_TIMEZONE,
    ambiguous: str = NLR300_DEFAULT_AMBIGUOUS_POLICY,
    nonexistent: str = NLR300_DEFAULT_NONEXISTENT_POLICY,
) -> dict[str, Any]:
    """Return auditable provenance for the user-authorized local-time policy."""
    _validate_timezone_name(timezone_name)
    _nlr300_ambiguous_argument(ambiguous)
    _nlr300_nonexistent_argument(nonexistent)
    return {
        "policy": NLR300_LOCAL_TIMEZONE_POLICY,
        "evidence_role": "assumed",
        "resolved_timezone": timezone_name,
        "publisher_metadata_claimed": False,
        "publisher_metadata_used": False,
        "civil_time_observes_dst": True,
        "dst_transition_policy": {
            "ambiguous": ambiguous,
            "ambiguous_semantics": (
                "choose the standard-time occurrence (fold=1) for a repeated local timestamp"
                if ambiguous == "standard_time"
                else "choose the daylight-time occurrence (fold=0) for a repeated local timestamp"
            ),
            "nonexistent": nonexistent,
            "nonexistent_semantics": (
                "preserve the source ts separately, emit UTC NaT, and set the QA invalid-time flag"
                if nonexistent == "mark_invalid"
                else "raise instead of constructing a UTC instant for a nonexistent local timestamp"
            ),
        },
    }


def nlr300_utc_source_to_local(
    values: pd.Series,
    *,
    source_clock_assumption: str = NLR300_SOURCE_CLOCK_ASSUMPTION,
    analysis_timezone: str = NLR300_LOCAL_TIMEZONE,
) -> pd.DataFrame:
    """Preserve naive source values, assume UTC, and derive Denver local time.

    UTC localization is one-to-one. The timezone-aware local result therefore
    handles DST transitions without choosing an ambiguous wall-clock branch or
    shifting a nonexistent local time.
    """
    policy_record = nlr300_utc_source_local_analysis_policy_record(
        source_clock_assumption=source_clock_assumption,
        analysis_timezone=analysis_timezone,
    )
    parsed = pd.to_datetime(values, errors="raise")
    if getattr(parsed.dt, "tz", None) is not None:
        raise GateFailure("NLR300 UTC-source policy requires timezone-naive ts values")
    timestamp_utc = parsed.dt.tz_localize(source_clock_assumption)
    timestamp_local = timestamp_utc.dt.tz_convert(analysis_timezone)
    return pd.DataFrame(
        {
            "source_ts": values.copy(),
            "parsed_source_ts": parsed,
            "timestamp_utc": timestamp_utc,
            "timestamp_local": timestamp_local,
            "source_clock_assumption_applied": parsed.notna(),
            "source_clock_assumption": policy_record["source_clock_assumption"],
            "analysis_timezone": policy_record["analysis_timezone"],
        },
        index=values.index,
    )


def _nlr300_ambiguous_argument(policy: str) -> bool:
    if policy == "standard_time":
        # pandas ambiguous=False selects the non-DST/standard-time occurrence.
        return False
    if policy == "daylight_time":
        return True
    raise ValueError(
        "NLR300 ambiguous policy must be 'standard_time' or 'daylight_time'"
    )


def _nlr300_nonexistent_argument(policy: str) -> str:
    if policy == "mark_invalid":
        return "NaT"
    if policy == "raise":
        return "raise"
    raise ValueError("NLR300 nonexistent policy must be 'mark_invalid' or 'raise'")


def nlr300_local_to_utc(
    values: pd.Series,
    *,
    timezone_name: str = NLR300_LOCAL_TIMEZONE,
    ambiguous: str = NLR300_DEFAULT_AMBIGUOUS_POLICY,
    nonexistent: str = NLR300_DEFAULT_NONEXISTENT_POLICY,
) -> pd.Series:
    """Interpret timezone-naive NLR300 timestamps as Denver civil time.

    Repeated fall-back timestamps deterministically select standard time by
    default. Nonexistent spring-forward timestamps become ``NaT`` by default,
    rather than being shifted into a fabricated or duplicate instant. The
    returned series is UTC-aware; callers needing row-level diagnostics should
    use :func:`nlr300_localize_with_qa`.
    """
    _validate_timezone_name(timezone_name)
    parsed = pd.to_datetime(values, errors="raise")
    if getattr(parsed.dt, "tz", None) is not None:
        raise GateFailure("NLR300 local-time policy requires timezone-naive ts values")
    localized = parsed.dt.tz_localize(
        timezone_name,
        ambiguous=_nlr300_ambiguous_argument(ambiguous),
        nonexistent=_nlr300_nonexistent_argument(nonexistent),
    )
    return localized.dt.tz_convert("UTC")


def nlr300_localize_with_qa(
    values: pd.Series,
    *,
    timezone_name: str = NLR300_LOCAL_TIMEZONE,
    ambiguous: str = NLR300_DEFAULT_AMBIGUOUS_POLICY,
    nonexistent: str = NLR300_DEFAULT_NONEXISTENT_POLICY,
) -> pd.DataFrame:
    """Preserve source timestamps and return UTC values with DST QA flags."""
    _validate_timezone_name(timezone_name)
    parsed = pd.to_datetime(values, errors="raise")
    if getattr(parsed.dt, "tz", None) is not None:
        raise GateFailure("NLR300 local-time policy requires timezone-naive ts values")

    utc_values = nlr300_local_to_utc(
        parsed,
        timezone_name=timezone_name,
        ambiguous=ambiguous,
        nonexistent=nonexistent,
    )
    ambiguous_probe = parsed.dt.tz_localize(
        timezone_name,
        ambiguous="NaT",
        nonexistent="shift_forward",
    )
    nonexistent_probe = parsed.dt.tz_localize(
        timezone_name,
        ambiguous=False,
        nonexistent="NaT",
    )
    return pd.DataFrame(
        {
            "source_ts": values.copy(),
            "parsed_local_ts": parsed,
            "timestamp_utc": utc_values,
            "qa_ambiguous_local_time": parsed.notna() & ambiguous_probe.isna(),
            "qa_nonexistent_local_time": parsed.notna() & nonexistent_probe.isna(),
            "timezone_assumption_applied": parsed.notna(),
        },
        index=values.index,
    )


def srrl_fixed_mst_to_utc(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise")
    fixed_mst = timezone(timedelta(hours=-7))
    if getattr(parsed.dt, "tz", None) is None:
        parsed = parsed.dt.tz_localize(fixed_mst)
    return parsed.dt.tz_convert("UTC")


def nyiso_local_to_utc(values: pd.Series, ambiguous: Any = "infer") -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise")
    if getattr(parsed.dt, "tz", None) is None:
        parsed = parsed.dt.tz_localize("America/New_York", ambiguous=ambiguous, nonexistent="shift_forward")
    return parsed.dt.tz_convert("UTC")


def assert_no_forbidden_training_columns(columns: list[str] | pd.Index) -> None:
    forbidden_tokens = ("xcel", "tariff", "demand_response", "future_realized_weather", "replay_market_date")
    bad = [column for column in columns if any(token in column.lower() for token in forbidden_tokens)]
    if bad:
        raise GateFailure(f"forbidden training columns: {bad}")
