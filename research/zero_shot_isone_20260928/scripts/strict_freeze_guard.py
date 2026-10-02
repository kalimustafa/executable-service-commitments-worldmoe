"""Inference-only guard helpers. No optimizer/training API is imported here."""

from __future__ import annotations

from contextlib import contextmanager

import torch


def freeze_for_inference(module: torch.nn.Module, label: str = "module") -> torch.nn.Module:
    module.eval()
    module.requires_grad_(False)
    trainable = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"{label} has trainable parameters: {trainable[:10]}")
    dirty = [name for name, parameter in module.named_parameters() if parameter.grad is not None]
    if dirty:
        raise RuntimeError(f"{label} has existing gradients: {dirty[:10]}")
    return module


@contextmanager
def strict_inference():
    """Prevent autograd graph creation for the duration of one inference call."""
    with torch.inference_mode():
        yield

