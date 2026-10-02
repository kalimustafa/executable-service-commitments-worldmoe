"""Deterministic, no-training OOD mappings for four-row NYISO tables."""

from __future__ import annotations

from itertools import permutations

import torch

ZONE_ORDER = ("WEST", "CAPITL", "N.Y.C.", "LONGIL")


def neutral_mean(vector_table: torch.Tensor) -> torch.Tensor:
    if vector_table.ndim < 2 or vector_table.shape[0] != 4:
        raise ValueError("Expected a frozen four-row NYISO zone/site table.")
    return vector_table.detach().mean(dim=0)


def neutral_center_scale(center: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if center.shape != scale.shape or center.shape[0] != 4:
        raise ValueError("Expected matching center/scale arrays with four NYISO zone rows.")
    return center.detach().mean(dim=0), scale.detach().mean(dim=0)


def neutral_one_hot() -> torch.Tensor:
    """Average of the four frozen NYISO zone indicators."""
    return torch.full((4,), 0.25, dtype=torch.float32)


def cyclic_assignments() -> dict[str, tuple[str, ...]]:
    return {
        "P1": ("WEST", "CAPITL", "N.Y.C.", "LONGIL"),
        "P2": ("CAPITL", "N.Y.C.", "LONGIL", "WEST"),
        "P3": ("N.Y.C.", "LONGIL", "WEST", "CAPITL"),
        "P4": ("LONGIL", "WEST", "CAPITL", "N.Y.C."),
    }


def all_fixed_permutations() -> list[tuple[str, ...]]:
    """Sensitivity only; these assignments are never selected by performance."""
    return list(permutations(ZONE_ORDER))

