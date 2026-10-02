from __future__ import annotations

from datetime import date
import hashlib
from typing import Iterable


def day_hash(day: str, study_seed: int) -> str:
    return hashlib.sha256(f"{day}|{study_seed}".encode("utf-8")).hexdigest()


def select_primary_days(
    eligible_dates: Iterable[str],
    *,
    study_seed: int,
    target_year: int | None = None,
) -> list[dict[str, str | int]]:
    """Select six weekdays and two weekend days per month, hash-only.

    If a stratum is short, the month is filled from the remaining lowest hashes.
    No price, load, weather, forecast, or outcome value is accepted by this API.
    """

    parsed: dict[str, date] = {}
    for value in eligible_dates:
        parsed[str(value)] = date.fromisoformat(str(value))
    if target_year is not None:
        parsed = {key: value for key, value in parsed.items() if value.year == target_year}

    selected: list[dict[str, str | int]] = []
    months = sorted({(value.year, value.month) for value in parsed.values()})
    for year, month in months:
        month_days = [key for key, value in parsed.items() if (value.year, value.month) == (year, month)]
        ranked = sorted(month_days, key=lambda key: (day_hash(key, study_seed), key))
        weekdays = [key for key in ranked if parsed[key].weekday() < 5]
        weekends = [key for key in ranked if parsed[key].weekday() >= 5]
        chosen = weekdays[:6] + weekends[:2]
        if len(chosen) < min(8, len(ranked)):
            chosen_set = set(chosen)
            chosen.extend(key for key in ranked if key not in chosen_set)
            chosen = chosen[: min(8, len(ranked))]
        for key in sorted(chosen, key=lambda item: (day_hash(item, study_seed), item)):
            selected.append(
                {
                    "date": key,
                    "year": year,
                    "month": month,
                    "day_type": "weekday" if parsed[key].weekday() < 5 else "weekend",
                    "selection_hash": day_hash(key, study_seed),
                    "selection_rule": "six_lowest_hash_weekdays_plus_two_lowest_hash_weekends_with_hash_fill",
                }
            )
    return selected
