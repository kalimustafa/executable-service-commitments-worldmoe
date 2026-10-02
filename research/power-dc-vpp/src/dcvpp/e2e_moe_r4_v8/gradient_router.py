"""Closed-form multi-task gradient routing for V8 shared parameters.

This is gradient surgery, not an operational mathematical solver.  The update
starts from one lexicographic primary gradient and sequentially removes any
component that would increase an active scientific constraint.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .config import PARAMETER_GROUPS


GradientTuple = tuple[Tensor | None, ...]


class GradientRoutingError(RuntimeError):
    """Parameter grouping or projected-gradient semantics are invalid."""


@dataclass(frozen=True)
class ResolvedParameterGroups:
    groups: Mapping[str, tuple[nn.Parameter, ...]]
    names: Mapping[str, tuple[str, ...]]

    def __getitem__(self, key: str) -> tuple[nn.Parameter, ...]:
        return self.groups[key]


def _entry_values(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        value = value.get("parameters", value.get("names"))
    if isinstance(value, (str, nn.Parameter)):
        return (value,)
    if not isinstance(value, Iterable):
        raise GradientRoutingError("each parameter group must be iterable")
    return value


def resolve_parameter_groups(model: nn.Module) -> ResolvedParameterGroups:
    """Resolve the model's six explicit, mutually exclusive parameter groups."""

    callback = getattr(model, "parameter_groups", None)
    if not callable(callback):
        raise GradientRoutingError("V8 model must expose parameter_groups()")
    raw = callback()
    if not isinstance(raw, Mapping) or set(raw) != set(PARAMETER_GROUPS):
        raise GradientRoutingError(
            f"parameter_groups() must return exactly {list(PARAMETER_GROUPS)}"
        )
    named = dict(model.named_parameters())
    by_id = {id(parameter): name for name, parameter in named.items()}
    groups: dict[str, tuple[nn.Parameter, ...]] = {}
    names: dict[str, tuple[str, ...]] = {}
    owners: dict[int, str] = {}
    for group in PARAMETER_GROUPS:
        parameters: list[nn.Parameter] = []
        group_names: list[str] = []
        for item in _entry_values(raw[group]):
            if isinstance(item, str):
                if item not in named:
                    raise GradientRoutingError(
                        f"group {group} names unknown parameter {item!r}"
                    )
                name, parameter = item, named[item]
            elif isinstance(item, nn.Parameter):
                if id(item) not in by_id:
                    raise GradientRoutingError(
                        f"group {group} contains a parameter outside the model"
                    )
                parameter, name = item, by_id[id(item)]
            else:
                raise GradientRoutingError(
                    f"group {group} contains a non-parameter entry"
                )
            previous = owners.get(id(parameter))
            if previous is not None:
                raise GradientRoutingError(
                    f"parameter {name} occurs in both {previous} and {group}"
                )
            owners[id(parameter)] = group
            parameters.append(parameter)
            group_names.append(name)
        groups[group] = tuple(parameters)
        names[group] = tuple(group_names)
    missing = set(named) - {name for values in names.values() for name in values}
    if missing:
        raise GradientRoutingError(
            f"parameter_groups() omitted model parameters: {sorted(missing)}"
        )
    for parameter in groups["frozen"]:
        parameter.requires_grad_(False)
    if any(not groups[name] for name in PARAMETER_GROUPS if name != "frozen"):
        raise GradientRoutingError("all five trainable V8 groups must be nonempty")
    return ResolvedParameterGroups(groups, names)


def _dot(left: GradientTuple, right: GradientTuple) -> Tensor:
    values = [
        a.float().mul(b.float()).sum()
        for a, b in zip(left, right, strict=True)
        if a is not None and b is not None
    ]
    if not values:
        return torch.tensor(0.0)
    result = values[0]
    for value in values[1:]:
        result = result + value.to(result)
    return result


def _add_scaled(
    left: GradientTuple, right: GradientTuple, scale: Tensor | float
) -> GradientTuple:
    output: list[Tensor | None] = []
    for a, b in zip(left, right, strict=True):
        if a is None and b is None:
            output.append(None)
        elif a is None:
            output.append(b * scale)  # type: ignore[operator]
        elif b is None:
            output.append(a)
        else:
            output.append(a + b.to(a) * scale)
    return tuple(output)


@dataclass(frozen=True)
class RoutedGradient:
    gradients: GradientTuple
    primary: str
    constraints: tuple[str, ...]
    projected_conflicts: tuple[str, ...]
    post_projection_dots: Mapping[str, float]


def route_shared_gradients(
    task_gradients: Mapping[str, GradientTuple],
    *,
    primary: str,
    constraints: Sequence[str],
    violations: Mapping[str, float] | None = None,
    passes: int = 2,
    epsilon: float = 1.0e-12,
) -> RoutedGradient:
    """Project a primary gradient against active task constraints.

    For gradient-descent update ``-g``, a nonnegative ``g dot c`` cannot
    increase the constraint to first order.  Conflicting negative components
    are removed analytically.  A positive violation adds a bounded repair
    component in the constraint-gradient direction.
    """

    if primary not in task_gradients:
        raise GradientRoutingError(f"missing primary gradient {primary!r}")
    if int(passes) < 1 or float(epsilon) <= 0.0:
        raise GradientRoutingError("projection passes/epsilon are invalid")
    width = len(task_gradients[primary])
    if width == 0 or any(len(value) != width for value in task_gradients.values()):
        raise GradientRoutingError("task gradient tuples must have equal nonzero width")
    ordered = tuple(str(name) for name in constraints)
    if any(name == primary or name not in task_gradients for name in ordered):
        raise GradientRoutingError("constraints must name non-primary task gradients")
    result = tuple(
        None if value is None else value.detach().clone()
        for value in task_gradients[primary]
    )
    if not any(value is not None for value in result):
        raise GradientRoutingError("primary task has no shared gradient")
    violation_map = {str(key): max(0.0, float(value)) for key, value in (violations or {}).items()}
    conflicts: list[str] = []
    for _ in range(int(passes)):
        for name in ordered:
            constraint = task_gradients[name]
            norm = _dot(constraint, constraint)
            if float(norm.detach().cpu()) <= float(epsilon):
                continue
            interaction = _dot(result, constraint)
            if float(interaction.detach().cpu()) < 0.0:
                result = _add_scaled(result, constraint, -interaction / (norm + epsilon))
                if name not in conflicts:
                    conflicts.append(name)
            violation = min(1.0, violation_map.get(name, 0.0))
            if violation > 0.0:
                result = _add_scaled(result, constraint, violation)
    dots = {
        name: float(_dot(result, task_gradients[name]).detach().cpu())
        for name in ordered
    }
    return RoutedGradient(
        result,
        str(primary),
        ordered,
        tuple(conflicts),
        dots,
    )


def assign_gradients(
    parameters: Sequence[nn.Parameter], gradients: GradientTuple
) -> None:
    if len(parameters) != len(gradients):
        raise GradientRoutingError("parameter/gradient widths differ")
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = None if gradient is None else gradient.detach().to(parameter)


__all__ = [
    "GradientRoutingError",
    "GradientTuple",
    "ResolvedParameterGroups",
    "RoutedGradient",
    "assign_gradients",
    "resolve_parameter_groups",
    "route_shared_gradients",
]
