"""Rigid-first and flexible EDF execution helpers for G3 feasibility screens."""

from __future__ import annotations

import numpy as np


def execute_rigid_flexible_earliest(
    rigid_arrivals: np.ndarray,
    flexible_arrivals: np.ndarray,
    capacity: np.ndarray,
    deadline_slots: int,
    max_tail_slots: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute rigid work as early as possible, then flexible work by EDF."""
    rigid_arrivals = np.asarray(rigid_arrivals, dtype=np.float64)
    flexible_arrivals = np.asarray(flexible_arrivals, dtype=np.float64)
    capacity = np.asarray(capacity, dtype=np.float64)
    if rigid_arrivals.shape != flexible_arrivals.shape:
        raise ValueError("Rigid and flexible arrival arrays must have the same shape.")
    if rigid_arrivals.ndim != 2 or rigid_arrivals.shape[1] != len(capacity):
        raise ValueError("Arrivals must be time by site and capacity must be site shaped.")
    if deadline_slots < 1:
        raise ValueError("deadline_slots must be positive.")
    if (rigid_arrivals < -1e-12).any() or (flexible_arrivals < -1e-12).any():
        raise ValueError("Work arrivals cannot be negative.")

    n, zones = rigid_arrivals.shape
    book = np.zeros((n + max_tail_slots, zones), dtype=np.float64)
    rigid_queue = np.zeros(zones, dtype=np.float64)
    flexible_deadlines = np.zeros((deadline_slots, zones), dtype=np.float64)
    total = rigid_arrivals + flexible_arrivals

    for t in range(n + max_tail_slots):
        if t:
            overdue = flexible_deadlines[0].copy()
            flexible_deadlines[:-1] = flexible_deadlines[1:]
            flexible_deadlines[-1] = 0.0
            flexible_deadlines[0] += overdue
        if t < n:
            rigid_queue += rigid_arrivals[t]
            flexible_deadlines[-1] += flexible_arrivals[t]

        rigid_service = np.minimum(rigid_queue, capacity)
        rigid_queue -= rigid_service
        remaining_capacity = np.maximum(capacity - rigid_service, 0.0)
        flexible_service = np.zeros(zones, dtype=np.float64)
        for due in range(deadline_slots):
            take = np.minimum(flexible_deadlines[due], remaining_capacity)
            flexible_deadlines[due] -= take
            remaining_capacity -= take
            flexible_service += take
            if not np.any(remaining_capacity > 1e-12):
                break
        book[t] = rigid_service + flexible_service

        if t >= n and not np.any(rigid_queue > 1e-10) and not np.any(flexible_deadlines > 1e-10):
            break
    return book, total, rigid_queue + flexible_deadlines.sum(axis=0)


def deadline_miss_mwh(
    book: np.ndarray,
    arrivals: np.ndarray,
    capacity_weights: np.ndarray,
    dt_hours: float,
    deadline_slots: int,
) -> float:
    """Return work still outstanding at each inclusive cohort deadline."""
    book = np.asarray(book, dtype=np.float64)
    arrivals = np.asarray(arrivals, dtype=np.float64)
    k = np.asarray(capacity_weights, dtype=np.float64)
    n = len(arrivals)
    served = np.cumsum(book, axis=0)
    released = np.cumsum(arrivals, axis=0)
    before = released - arrivals
    due_index = np.arange(n) + deadline_slots - 1
    late = np.maximum(released - np.maximum(before, served[due_index]), 0.0)
    return float(dt_hours * np.sum(late * k))


def closing_work_mwh(book: np.ndarray, n: int, capacity_weights: np.ndarray, dt_hours: float) -> float:
    return float(dt_hours * np.sum(np.asarray(book)[n:] * np.asarray(capacity_weights)[None, :]))
