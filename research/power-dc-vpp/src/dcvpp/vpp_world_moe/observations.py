"""Physics-aware observation queries from unified WorldMoE latent states."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import WorldMoEConfig
from .packets import (
    ENTITIES_PER_NODE,
    NODES,
    WorldConstraintPacket,
    WorldObservationPacket,
    WorldStatePacket,
    close_price_identity,
)


QUANTILE_LEVELS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def _required_train_probability(config: WorldMoEConfig, name: str) -> float:
    raw = getattr(config, name, None)
    if raw is None:
        raise ValueError(
            f"support atom head requires train-only resolved probability {name}"
        )
    value = float(raw)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly inside (0,1)")
    return value


def _pv_night_positive_probability(config: WorldMoEConfig) -> float:
    """Resolve the train-only night support prior without hiding lineage drift."""

    positive_raw = getattr(
        config, "support_atom_pv_night_positive_train_prevalence", None
    )
    zero_raw = getattr(config, "support_atom_pv_night_zero_train_prevalence", None)
    if positive_raw is None and zero_raw is None:
        raise ValueError(
            "mixed night PV support requires a train-only night positive or zero "
            "prevalence"
        )
    if positive_raw is not None:
        positive = float(positive_raw)
        if not math.isfinite(positive) or not 0.0 < positive < 1.0:
            raise ValueError(
                "support_atom_pv_night_positive_train_prevalence must lie in (0,1)"
            )
        zero = 1.0 - positive
        if zero_raw is not None and not math.isclose(
            zero, float(zero_raw), rel_tol=0.0, abs_tol=1.0e-10
        ):
            raise ValueError("night PV positive and zero train prevalences disagree")
        return positive
    zero = float(zero_raw)
    if not math.isfinite(zero) or not 0.0 < zero < 1.0:
        raise ValueError(
            "support_atom_pv_night_zero_train_prevalence must lie in (0,1)"
        )
    return 1.0 - zero


def _initialize_binary_atom_head(head: nn.Sequential, probability: float) -> None:
    projection = head[-1]
    if not isinstance(projection, nn.Linear) or projection.out_features != 1:
        raise TypeError("binary atom head must end in a one-logit Linear")
    nn.init.normal_(projection.weight, mean=0.0, std=1.0e-4)
    nn.init.constant_(projection.bias, math.log(probability / (1.0 - probability)))


def _initialize_categorical_atom_head(
    head: nn.Sequential,
    first_probability: float,
    second_probability: float,
) -> None:
    projection = head[-1]
    if not isinstance(projection, nn.Linear) or projection.out_features != 2:
        raise TypeError("categorical atom head must end in a two-logit Linear")
    interior = 1.0 - float(first_probability) - float(second_probability)
    if interior <= 0.0:
        raise ValueError("categorical atom probabilities leave no interior mass")
    nn.init.normal_(projection.weight, mean=0.0, std=1.0e-4)
    with torch.no_grad():
        projection.bias[0] = math.log(float(first_probability) / interior)
        projection.bias[1] = math.log(float(second_probability) / interior)


def _straight_through_binary(probability: Tensor, uniform: Tensor) -> Tensor:
    if probability.shape != uniform.shape:
        raise ValueError("binary support probability and uniform shapes differ")
    hard = (uniform < probability).to(probability.dtype)
    return hard + (probability - probability.detach())


def _straight_through_three_way(
    probabilities: Tensor,
    uniform: Tensor,
) -> Tensor:
    if probabilities.shape[:-1] != uniform.shape or probabilities.shape[-1] != 3:
        raise ValueError("three-way support probability/uniform shapes differ")
    first = uniform < probabilities[..., 0]
    second = (~first) & (
        uniform < probabilities[..., 0] + probabilities[..., 1]
    )
    hard = torch.stack((first, second, ~(first | second)), dim=-1).to(
        probabilities.dtype
    )
    return hard + (probabilities - probabilities.detach())


class MonotoneQuantileHead(nn.Module):
    """Decode ordered quantiles using one base and positive increments.

    The final axis is monotone by construction.  Optional positive/bounded
    transforms are monotone maps and therefore preserve that ordering.
    """

    def __init__(self, width: int, output_channels: int) -> None:
        super().__init__()
        self.output_channels = int(output_channels)
        self.projection = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, self.output_channels * len(QUANTILE_LEVELS)),
        )

    def initialize_channel_near_zero(self, channel: int) -> None:
        """Initialize one residual channel at the causal anchor.

        Positive-increment quantile parameterisations otherwise start with
        roughly ``softplus(0)`` between adjacent quantiles.  For a bounded
        anchor residual that would discard the anchor before the first update.
        """

        index = int(channel)
        if index < 0 or index >= self.output_channels:
            raise ValueError("quantile channel index is out of range")
        projection = self.projection[-1]
        if not isinstance(projection, nn.Linear):
            raise TypeError("quantile projection must end in a linear layer")
        width = len(QUANTILE_LEVELS)
        start = index * width
        with torch.no_grad():
            projection.weight[start : start + width].zero_()
            projection.bias[start] = 0.0
            projection.bias[start + 1 : start + width].fill_(-8.0)

    def forward(
        self,
        value: Tensor,
        *,
        transform: str = "identity",
        lower: Tensor | None = None,
        upper: Tensor | None = None,
    ) -> Tensor:
        raw = (
            self.projection(value)
            .float()
            .reshape(*value.shape[:-1], self.output_channels, len(QUANTILE_LEVELS))
        )
        base = raw[..., :1]
        increments = F.softplus(raw[..., 1:]).clamp_min(1.0e-6)
        ordered = torch.cat((base, base + increments.cumsum(dim=-1)), dim=-1)
        if transform == "positive":
            return F.softplus(ordered)
        if transform == "bounded":
            if lower is None or upper is None:
                raise ValueError("bounded quantiles require lower and upper tensors")
            return lower + torch.sigmoid(ordered) * (upper - lower)
        if transform != "identity":
            raise ValueError(f"unknown monotone quantile transform: {transform}")
        return ordered


def _expand_prefix(value: Tensor, prefix_dims: int) -> Tensor:
    result = value
    for _ in range(prefix_dims - 1):
        result = result.unsqueeze(1)
    return result


def _broadcast_price_anchor(value: Tensor, target: Tensor) -> Tensor:
    """Broadcast raw ``[...,T,4]`` anchors across optional path dimensions."""

    anchor = torch.as_tensor(value, device=target.device, dtype=torch.float32)
    if anchor.ndim < 2 or anchor.shape[-2:] != target.shape[-2:]:
        raise ValueError("DA anchor must end in the decoded [time,4] price dimensions")
    anchor_prefix = tuple(anchor.shape[:-2])
    target_prefix = tuple(target.shape[:-2])
    if (
        len(anchor_prefix) > len(target_prefix)
        or tuple(target_prefix[: len(anchor_prefix)]) != anchor_prefix
    ):
        raise ValueError("DA anchor batch prefix is incompatible with decoded prices")
    shape = (
        *anchor_prefix,
        *((1,) * (len(target_prefix) - len(anchor_prefix))),
        *anchor.shape[-2:],
    )
    return anchor.reshape(shape).expand_as(target)


def _broadcast_node_cap(value: Tensor, target: Tensor) -> Tensor:
    """Broadcast scalar/[4]/[batch,4] raw-unit caps over time and paths."""

    cap = torch.as_tensor(value, device=target.device, dtype=torch.float32)
    if cap.ndim == 0:
        result = cap
    elif cap.shape == (NODES,):
        result = cap.reshape(*((1,) * (target.ndim - 1)), NODES)
    elif cap.ndim >= 2 and cap.shape[-1] == NODES:
        cap_prefix = tuple(cap.shape[:-1])
        target_prefix = tuple(target.shape[:-2])
        if (
            len(cap_prefix) > len(target_prefix)
            or tuple(target_prefix[: len(cap_prefix)]) != cap_prefix
        ):
            raise ValueError("DA residual cap batch prefix is incompatible")
        result = cap.reshape(
            *cap_prefix,
            *((1,) * (len(target_prefix) - len(cap_prefix) + 1)),
            NODES,
        )
    else:
        raise ValueError("DA residual cap must be scalar, [4], or [batch,4]")
    result = torch.broadcast_to(result, target.shape)
    if not bool(torch.isfinite(result).all()) or bool((result <= 0).any()):
        raise ValueError("DA residual cap must be finite and positive")
    return result


def _lindley_queue_and_service(
    arrivals: Tensor,
    raw_service: Tensor,
    initial_queue: Tensor,
) -> tuple[Tensor, Tensor]:
    """Evaluate the nonnegative queue recurrence with two parallel scans.

    For net input ``x_t = arrivals_t - raw_service_t``, the sequential
    recurrence is the Lindley equation ``q_t = max(0, q_{t-1} + x_t)``.
    Its reflected cumulative-sum form is

    ``q_t = z_t - min(0, min_{k<=t} z_k)``,
    ``z_t = q_0 + sum_{k<=t} x_k``.

    ``cumsum`` and ``cummin`` therefore replace the Python-level time loop
    while preserving every optional batch/path prefix.  Inputs arrive here
    after the observation heads have promoted their public quantities to
    FP32.  Service retains its original definition rather than being inferred
    from a rounded queue difference.
    """

    unconstrained_queue = initial_queue.unsqueeze(-2) + torch.cumsum(
        arrivals - raw_service,
        dim=-2,
    )
    running_floor = torch.minimum(
        torch.cummin(unconstrained_queue, dim=-2).values,
        torch.zeros_like(unconstrained_queue),
    )
    queue_after_service = unconstrained_queue - running_floor
    queue_sequence = torch.cat(
        (initial_queue.unsqueeze(-2), queue_after_service),
        dim=-2,
    )
    available = queue_sequence[..., :-1, :] + arrivals
    service = torch.minimum(raw_service, available)
    return service, queue_sequence


class WorldObservationQueries(nn.Module):
    """Joint price, queue, bounded PV and physics-derived PUE queries."""

    def __init__(self, config: WorldMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.analytic_mixed_support_enabled = bool(
            getattr(config, "analytic_mixed_support_enabled", False)
        )
        if self.analytic_mixed_support_enabled and not bool(
            getattr(config, "support_atom_enabled", False)
        ):
            raise ValueError(
                "analytic mixed support requires support_atom_enabled"
            )
        width = config.d_model
        self.node_embedding = nn.Parameter(torch.randn(NODES, width) / width**0.5)
        self.price_query = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2)
        )
        self.arrival_query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.service_query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.soc_query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.pv_query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.cooling_ratio_query = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 1)
        )
        # DA and RT are the only trainable price quantile outputs.  DART is
        # derived from their coupled paths below for the legacy no-anchor path.
        # In R5-V2.1 channel zero is a bounded DA-anchor residual and channel
        # one is neural DART; RT is closed deterministically as DA-DART.
        self.price_quantile_query = MonotoneQuantileHead(width, 2)
        self.price_quantile_query.initialize_channel_near_zero(0)
        self.workload_volume_quantile_query = MonotoneQuantileHead(width, 1)
        self.pv_quantile_query = MonotoneQuantileHead(width, 1)
        self.pue_quantile_query = MonotoneQuantileHead(width, 1)
        self.workload_count_query = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2)
        )
        # Queue is a zero-inflated state target.  Keep its occurrence and
        # positive severity explicit instead of asking the physical recurrence
        # to serve as a point reconstruction head.  The recurrence remains the
        # operational queue used by planning; these heads are the W0 state
        # representation query.
        self.queue_occurrence_query = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        self.queue_positive_severity_query = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        # A fresh R5-V2.1 lineage starts exactly at its causal DA anchor.
        final_price = self.price_query[-1]
        if isinstance(final_price, nn.Linear):
            with torch.no_grad():
                final_price.weight[0].zero_()
                final_price.bias[0].zero_()
        # R5-V2.4 registers these heads only in the new lineage.  A legacy
        # support-disabled configuration therefore retains its exact
        # state-dict keyset and seeded initialization stream.
        if bool(getattr(config, "support_atom_enabled", False)):
            self.workload_zero_query = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, 1)
            )
            self.pv_atom_query = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, 2)
            )
            # A train-only audit found no non-structural PUE-max events: max
            # PUE is exactly induced by the workload-zero hurdle.  The only
            # residual PUE atom is therefore the physical minimum.
            self.pue_atom_query = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, 1)
            )
            workload_zero = _required_train_probability(
                config, "support_atom_workload_zero_train_prevalence"
            )
            pv_daylight_zero = _required_train_probability(
                config, "support_atom_pv_daylight_zero_prevalence"
            )
            pv_daylight_capacity = _required_train_probability(
                config, "support_atom_pv_daylight_capacity_prevalence"
            )
            pue_residual_minimum = _required_train_probability(
                config, "support_atom_pue_residual_minimum_prevalence"
            )
            _initialize_binary_atom_head(self.workload_zero_query, workload_zero)
            _initialize_categorical_atom_head(
                self.pv_atom_query,
                pv_daylight_zero,
                pv_daylight_capacity,
            )
            _initialize_binary_atom_head(
                self.pue_atom_query, pue_residual_minimum
            )
            if self.analytic_mixed_support_enabled:
                mode = str(
                    getattr(config, "pv_night_support_mode", "")
                ).strip().lower()
                if mode != "mixed_zero_rare_positive":
                    raise ValueError(
                        "analytic mixed support requires PV night mode "
                        "mixed_zero_rare_positive"
                    )
                # Night support is not deterministic in the observed train
                # distribution.  A separate hurdle prevents the overwhelmingly
                # common zero mass from erasing the rare strictly-positive
                # interior branch.
                self.pv_night_positive_query = nn.Sequential(
                    nn.LayerNorm(width), nn.Linear(width, 1)
                )
                _initialize_binary_atom_head(
                    self.pv_night_positive_query,
                    _pv_night_positive_probability(config),
                )
                # These heads parameterize conditional interior distributions.
                # They are intentionally distinct from the W0 unconditional
                # quantile heads, whose lower quantiles may legitimately sit on
                # structural atoms.
                self.workload_interior_quantile_query = MonotoneQuantileHead(
                    width, 1
                )
                self.pv_interior_quantile_query = MonotoneQuantileHead(width, 1)
                self.pue_interior_quantile_query = MonotoneQuantileHead(width, 1)

    def _decode(
        self,
        market: Tensor,
        battery_token: Tensor,
        arrival_token: Tensor,
        queue_token: Tensor,
        pv_token: Tensor,
        cooling_token: Tensor,
        compute_token: Tensor,
        constraints: WorldConstraintPacket,
        *,
        da_anchor: Tensor | None = None,
        da_residual_cap: Tensor | None = None,
    ) -> WorldObservationPacket:
        del queue_token
        prefix = market.shape[:-3]
        prefix_dims = len(prefix)
        # Neural heads may run under BF16 autocast.  Public quantities and
        # deterministic state recurrences are always promoted to FP32.
        price = self.price_query(market).float()
        if da_anchor is None:
            da, rt, dart = close_price_identity(price[..., 0], price[..., 1])
        else:
            if da_residual_cap is None:
                raise ValueError("a DA anchor requires a train-frozen residual cap")
            raw_da_residual = price[..., 0]
            anchor = _broadcast_price_anchor(da_anchor, raw_da_residual)
            cap = _broadcast_node_cap(da_residual_cap, raw_da_residual)
            da = anchor + cap * torch.tanh(raw_da_residual.float())
            neural_dart = price[..., 1]
            rt = da - neural_dart
            # Recompute DART last so (DA-RT)-DART is exactly representable in
            # the forward dtype rather than relying on two independent heads.
            dart = da - rt

        arrivals = F.softplus(self.arrival_query(arrival_token).squeeze(-1).float())
        compute_capacity = _expand_prefix(
            constraints.compute_capacity.float(), prefix_dims
        ).unsqueeze(-2)
        raw_service = (
            torch.sigmoid(self.service_query(compute_token).squeeze(-1).float())
            * compute_capacity
        )
        queue = _expand_prefix(constraints.initial_queue.float(), prefix_dims)
        queue = queue.expand(*prefix, NODES)
        service, queue_sequence = _lindley_queue_and_service(
            arrivals,
            raw_service,
            queue,
        )

        pv_capacity = _expand_prefix(
            constraints.pv_capacity.float(), prefix_dims
        ).unsqueeze(-2)
        pv = torch.sigmoid(self.pv_query(pv_token).squeeze(-1).float()) * pv_capacity
        idle = _expand_prefix(constraints.idle_power.float(), prefix_dims).unsqueeze(-2)
        work_power = _expand_prefix(
            constraints.work_power_per_unit.float(), prefix_dims
        ).unsqueeze(-2)
        it_power = idle + work_power * service

        pue_min = _expand_prefix(constraints.pue_min.float(), prefix_dims).unsqueeze(-2)
        pue_max = _expand_prefix(constraints.pue_max.float(), prefix_dims).unsqueeze(-2)
        ratio_fraction = torch.sigmoid(
            self.cooling_ratio_query(cooling_token).squeeze(-1).float()
        )
        cooling_ratio = (pue_min - 1.0) + ratio_fraction * (pue_max - pue_min)
        cooling_power = it_power * cooling_ratio
        facility_power = it_power + cooling_power
        pue = facility_power / it_power.clamp_min(1.0e-6)
        # Observation SOC follows the corpus state contract and is a fraction
        # in [0,1].  The action packet separately carries physical MWh states.
        soc = torch.sigmoid(self.soc_query(battery_token).squeeze(-1).float())
        aggregate_workload = arrivals.sum(dim=-1, keepdim=True)
        public = torch.cat((da, rt, dart, aggregate_workload, pv, pue), dim=-1)
        return WorldObservationPacket(
            da=da,
            rt=rt,
            dart=dart,
            arrivals=arrivals,
            service=service,
            queue=queue_sequence,
            pv=pv,
            it_power=it_power,
            cooling_power=cooling_power,
            facility_power=facility_power,
            pue=pue,
            soc=soc,
            public_21=public,
        ).validate()

    def _decode_stochastic_support(
        self,
        market: Tensor,
        battery_token: Tensor,
        arrival_token: Tensor,
        queue_token: Tensor,
        pv_token: Tensor,
        cooling_token: Tensor,
        compute_token: Tensor,
        constraints: WorldConstraintPacket,
        support_uniforms: Mapping[str, Tensor],
    ) -> tuple[WorldObservationPacket, dict[str, Tensor]]:
        """Decode stochastic paths with explicit physical support atoms.

        This method is intentionally separate from :meth:`_decode`: W0/W1
        and legacy checkpoints keep their original continuous observation
        path byte-for-byte, while U0/P1 can opt into the R5-V2.4 mixed
        discrete/continuous support.  The categorical draws use caller-owned
        uniforms so the dynamics generator remains the single RNG authority.
        """

        del queue_token
        required_uniforms = {"workload_zero", "pv_atom", "pue_min"}
        if set(support_uniforms) != required_uniforms:
            raise ValueError(
                "support uniforms must contain workload_zero, pv_atom, and pue_min"
            )
        if constraints.pv_support_mask is None:
            raise ValueError("support-atom rollout requires a causal pv_support_mask")

        prefix = market.shape[:-3]
        prefix_dims = len(prefix)
        price = self.price_query(market).float()
        da, rt, dart = close_price_identity(price[..., 0], price[..., 1])

        positive_arrivals = F.softplus(
            self.arrival_query(arrival_token).squeeze(-1).float()
        )
        workload_logits = self.workload_zero_query(
            arrival_token.mean(dim=-2)
        ).float()
        workload_zero_probability = torch.sigmoid(workload_logits)
        workload_zero = _straight_through_binary(
            workload_zero_probability,
            support_uniforms["workload_zero"].to(
                device=workload_zero_probability.device,
                dtype=workload_zero_probability.dtype,
            ),
        )
        arrivals = positive_arrivals * (1.0 - workload_zero)

        compute_capacity = _expand_prefix(
            constraints.compute_capacity.float(), prefix_dims
        ).unsqueeze(-2)
        raw_service = (
            torch.sigmoid(self.service_query(compute_token).squeeze(-1).float())
            * compute_capacity
        )
        initial_queue = _expand_prefix(constraints.initial_queue.float(), prefix_dims)
        initial_queue = initial_queue.expand(*prefix, NODES)
        service, queue_sequence = _lindley_queue_and_service(
            arrivals,
            raw_service,
            initial_queue,
        )

        pv_capacity = _expand_prefix(
            constraints.pv_capacity.float(), prefix_dims
        ).unsqueeze(-2)
        positive_pv_fraction = torch.sigmoid(
            self.pv_query(pv_token).squeeze(-1).float()
        )
        if self.analytic_mixed_support_enabled:
            positive_pv_fraction = positive_pv_fraction.clamp(
                1.0e-6, 1.0 - 1.0e-6
            )
        positive_pv = positive_pv_fraction * pv_capacity
        pv_atom_logits = self.pv_atom_query(pv_token).float()
        pv_atom_probabilities = torch.softmax(
            torch.cat(
                (pv_atom_logits, torch.zeros_like(pv_atom_logits[..., :1])),
                dim=-1,
            ),
            dim=-1,
        )
        pv_atom = _straight_through_three_way(
            pv_atom_probabilities,
            support_uniforms["pv_atom"].to(
                device=pv_atom_probabilities.device,
                dtype=pv_atom_probabilities.dtype,
            ),
        )
        daylight = constraints.pv_support_mask.to(device=positive_pv.device)
        if daylight.shape[-2:] != positive_pv.shape[-2:]:
            raise ValueError("causal pv_support_mask differs from rollout horizon")
        while daylight.ndim < positive_pv.ndim:
            daylight = daylight.unsqueeze(1)
        daylight = daylight.expand_as(positive_pv)
        daylight_float = daylight.to(positive_pv.dtype)
        pv_zero_draw = pv_atom[..., 0]
        pv_capacity_draw = pv_atom[..., 1]
        daylight_pv = (
            pv_capacity_draw * pv_capacity
            + (1.0 - pv_zero_draw - pv_capacity_draw) * positive_pv
        )
        if self.analytic_mixed_support_enabled:
            night_positive_probability = torch.sigmoid(
                self.pv_night_positive_query(pv_token).squeeze(-1).float()
            )
            night_positive_draw = _straight_through_binary(
                night_positive_probability,
                support_uniforms["pv_atom"].to(
                    device=night_positive_probability.device,
                    dtype=night_positive_probability.dtype,
                ),
            )
            night_pv = night_positive_draw * positive_pv
            pv = daylight_float * daylight_pv + (1.0 - daylight_float) * night_pv
            pv_zero_probability = (
                daylight_float * pv_atom_probabilities[..., 0]
                + (1.0 - daylight_float) * (1.0 - night_positive_probability)
            )
        else:
            # R5-V2.4 compatibility: known night was a deterministic zero.
            pv = daylight_float * daylight_pv
            pv_zero_probability = (1.0 - daylight_float) + (
                daylight_float * pv_atom_probabilities[..., 0]
            )
        pv_capacity_probability = daylight_float * pv_atom_probabilities[..., 1]

        idle = _expand_prefix(constraints.idle_power.float(), prefix_dims).unsqueeze(-2)
        work_power = _expand_prefix(
            constraints.work_power_per_unit.float(), prefix_dims
        ).unsqueeze(-2)
        it_power = idle + work_power * service

        pue_min = _expand_prefix(constraints.pue_min.float(), prefix_dims).unsqueeze(-2)
        pue_max = _expand_prefix(constraints.pue_max.float(), prefix_dims).unsqueeze(-2)
        utilization = (service / compute_capacity.clamp_min(1.0e-6)).clamp(0.0, 1.0)
        cooling_shape = 0.25 + F.softplus(
            self.cooling_ratio_query(cooling_token).squeeze(-1).float()
        )
        interior_fraction = (1.0 - utilization).pow(cooling_shape)
        # Non-atom samples must remain strictly inside the physical interval;
        # otherwise zero utilization would silently create an unmodelled PUE
        # max atom in addition to the workload-zero structural event.
        interior_fraction = interior_fraction.clamp(1.0e-6, 1.0 - 1.0e-6)
        pue_interior = pue_min + interior_fraction * (pue_max - pue_min)
        pue_min_probability_conditional = torch.sigmoid(
            self.pue_atom_query(cooling_token).squeeze(-1).float()
        )
        pue_min_draw = _straight_through_binary(
            pue_min_probability_conditional,
            support_uniforms["pue_min"].to(
                device=pue_min_probability_conditional.device,
                dtype=pue_min_probability_conditional.dtype,
            ),
        )
        pue_nonzero = (
            pue_min_draw * pue_min + (1.0 - pue_min_draw) * pue_interior
        )
        workload_zero_nodes = workload_zero.expand(*workload_zero.shape[:-1], NODES)
        workload_zero_probability_nodes = workload_zero_probability.expand(
            *workload_zero_probability.shape[:-1], NODES
        )
        pue = (
            workload_zero_nodes * pue_max
            + (1.0 - workload_zero_nodes) * pue_nonzero
        )
        pue_min_probability = (
            (1.0 - workload_zero_probability_nodes)
            * pue_min_probability_conditional
        )
        # The train-only support audit found no residual max event.  PUE-max
        # probability is therefore exactly the VPP-wide workload-zero mass.
        pue_max_probability = workload_zero_probability_nodes
        cooling_power = it_power * (pue - 1.0)
        facility_power = it_power + cooling_power

        soc = torch.sigmoid(self.soc_query(battery_token).squeeze(-1).float())
        aggregate_workload = arrivals.sum(dim=-1, keepdim=True)
        public = torch.cat((da, rt, dart, aggregate_workload, pv, pue), dim=-1)
        observation = WorldObservationPacket(
            da=da,
            rt=rt,
            dart=dart,
            arrivals=arrivals,
            service=service,
            queue=queue_sequence,
            pv=pv,
            it_power=it_power,
            cooling_power=cooling_power,
            facility_power=facility_power,
            pue=pue,
            soc=soc,
            public_21=public,
        ).validate()
        support_probabilities = {
            "workload_zero": workload_zero_probability,
            "pv_zero": pv_zero_probability,
            "pv_capacity": pv_capacity_probability,
            "pue_min": pue_min_probability,
            "pue_max": pue_max_probability,
        }
        return observation, support_probabilities

    def from_state(
        self,
        state: WorldStatePacket,
        constraints: WorldConstraintPacket,
        *,
        da_anchor: Tensor | None = None,
        da_residual_cap: Tensor | None = None,
    ) -> WorldObservationPacket:
        batch, time, _, width = state.slots.shape
        local = state.slots[:, :, : NODES * ENTITIES_PER_NODE].reshape(
            batch, time, NODES, ENTITIES_PER_NODE, width
        )
        return self._decode(
            local[..., 0, :],
            local[..., 1, :],
            local[..., 4, :],
            local[..., 5, :],
            local[..., 2, :],
            local[..., 3, :],
            local[..., 6, :],
            constraints,
            da_anchor=da_anchor,
            da_residual_cap=da_residual_cap,
        )

    def from_latent_timeline(
        self,
        latent: Tensor,
        constraints: WorldConstraintPacket,
        *,
        da_anchor: Tensor | None = None,
        da_residual_cap: Tensor | None = None,
    ) -> WorldObservationPacket:
        # latent is [B,T,D] or [B,P,T,D].  Node embeddings recover typed
        # node-specific query states without materializing forty rollout slots.
        node = self.node_embedding.to(latent.dtype)
        typed = latent.unsqueeze(-2) + node
        return self._decode(
            typed,
            typed,
            typed,
            typed,
            typed,
            typed,
            typed,
            constraints,
            da_anchor=da_anchor,
            da_residual_cap=da_residual_cap,
        )

    def from_latent_timeline_with_support(
        self,
        latent: Tensor,
        constraints: WorldConstraintPacket,
        support_uniforms: Mapping[str, Tensor],
    ) -> tuple[WorldObservationPacket, dict[str, Tensor]]:
        """Decode U0/P1 paths with explicit mixed-support observations."""

        if not bool(getattr(self.config, "support_atom_enabled", False)):
            raise ValueError("support decoding requires support_atom_enabled")
        if latent.ndim != 4 or latent.shape[-1] != self.config.d_model:
            raise ValueError("support latent timeline must be [B,P,T,D]")
        typed = latent.unsqueeze(-2) + self.node_embedding.to(latent.dtype)
        return self._decode_stochastic_support(
            typed,
            typed,
            typed,
            typed,
            typed,
            typed,
            typed,
            constraints,
            support_uniforms,
        )

    def _mixed_distribution_decode(
        self,
        market: Tensor,
        arrival_token: Tensor,
        pv_token: Tensor,
        cooling_token: Tensor,
        constraints: WorldConstraintPacket,
        *,
        da_anchor: Tensor | None = None,
        da_residual_cap: Tensor | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor] | tuple[str, ...]]:
        """Decode an analytic mixed marginal distribution.

        Boundary occurrence probabilities and conditional-interior quantiles
        are distribution parameters, not empirical summaries of sampled
        scenarios.  Scenario paths remain available to joint scores, while
        marginal calibration can therefore be evaluated without a minimum
        number of rare component draws.
        """

        if not self.analytic_mixed_support_enabled:
            raise ValueError(
                "analytic distribution decoding requires "
                "analytic_mixed_support_enabled"
            )
        if constraints.pv_support_mask is None:
            raise ValueError(
                "analytic mixed distribution requires a causal pv_support_mask"
            )
        prefix_dims = len(market.shape[:-3])
        legacy_quantiles = self._quantile_decode(
            market,
            arrival_token,
            pv_token,
            cooling_token,
            constraints,
            da_anchor=da_anchor,
            da_residual_cap=da_residual_cap,
        )

        workload_token = arrival_token.mean(dim=-2)
        workload_zero = torch.sigmoid(
            self.workload_zero_query(workload_token).float()
        )
        workload_interior = self.workload_interior_quantile_query(
            workload_token, transform="positive"
        )

        pv_logits = self.pv_atom_query(pv_token).float()
        daylight_probabilities = torch.softmax(
            torch.cat((pv_logits, torch.zeros_like(pv_logits[..., :1])), dim=-1),
            dim=-1,
        )
        daylight = constraints.pv_support_mask.to(device=pv_token.device)
        expected_pv_shape = pv_token.shape[:-1]
        if daylight.shape[-2:] != expected_pv_shape[-2:]:
            raise ValueError("causal pv_support_mask differs from distribution horizon")
        while daylight.ndim < len(expected_pv_shape):
            daylight = daylight.unsqueeze(1)
        daylight = daylight.expand(expected_pv_shape)
        daylight_float = daylight.to(torch.float32)
        night_positive = torch.sigmoid(
            self.pv_night_positive_query(pv_token).squeeze(-1).float()
        )
        pv_zero = (
            daylight_float * daylight_probabilities[..., 0]
            + (1.0 - daylight_float) * (1.0 - night_positive)
        )
        pv_capacity_probability = (
            daylight_float * daylight_probabilities[..., 1]
        )
        pv_capacity = (
            _expand_prefix(constraints.pv_capacity.float(), prefix_dims)
            .unsqueeze(-2)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        pv_epsilon = (pv_capacity * 1.0e-6).clamp_min(1.0e-8)
        pv_interior = self.pv_interior_quantile_query(
            pv_token,
            transform="bounded",
            lower=pv_epsilon,
            upper=pv_capacity - pv_epsilon,
        ).squeeze(-2)

        pue_min = (
            _expand_prefix(constraints.pue_min.float(), prefix_dims)
            .unsqueeze(-2)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        pue_max = (
            _expand_prefix(constraints.pue_max.float(), prefix_dims)
            .unsqueeze(-2)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        pue_epsilon = ((pue_max - pue_min) * 1.0e-6).clamp_min(1.0e-8)
        pue_interior = self.pue_interior_quantile_query(
            cooling_token,
            transform="bounded",
            lower=pue_min + pue_epsilon,
            upper=pue_max - pue_epsilon,
        ).squeeze(-2)
        pue_min_conditional = torch.sigmoid(
            self.pue_atom_query(cooling_token).squeeze(-1).float()
        )
        workload_zero_nodes = workload_zero.expand(
            *workload_zero.shape[:-1], NODES
        )
        pue_min_probability = (1.0 - workload_zero_nodes) * pue_min_conditional
        pue_max_probability = workload_zero_nodes

        return {
            "quantile_levels": torch.tensor(
                QUANTILE_LEVELS, device=market.device, dtype=torch.float32
            ),
            "quantiles": {
                "price": legacy_quantiles["price"],
                "workload": workload_interior,
                "pv": pv_interior,
                "pue": pue_interior,
            },
            "atom_probabilities": {
                "workload_zero": workload_zero,
                "pv_zero": pv_zero,
                "pv_capacity": pv_capacity_probability,
                "pue_min": pue_min_probability,
                "pue_max": pue_max_probability,
            },
            "conditional_interior_families": ("workload", "pv", "pue"),
            "joint_price_paths": legacy_quantiles["joint_price_paths"],
        }

    def mixed_distribution_from_state(
        self,
        state: WorldStatePacket,
        constraints: WorldConstraintPacket,
        *,
        da_anchor: Tensor | None = None,
        da_residual_cap: Tensor | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor] | tuple[str, ...]]:
        """Return path-free mixed distribution parameters from a world state."""

        batch, time, _, width = state.slots.shape
        local = state.slots[:, :, : NODES * ENTITIES_PER_NODE].reshape(
            batch, time, NODES, ENTITIES_PER_NODE, width
        )
        return self._mixed_distribution_decode(
            local[..., 0, :],
            local[..., 4, :],
            local[..., 2, :],
            local[..., 3, :],
            constraints,
            da_anchor=da_anchor,
            da_residual_cap=da_residual_cap,
        )

    def mixed_distribution_from_latent_timeline(
        self,
        latent: Tensor,
        constraints: WorldConstraintPacket,
        *,
        da_anchor: Tensor | None = None,
        da_residual_cap: Tensor | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor] | tuple[str, ...]]:
        """Return path-free mixed distribution parameters from ``[B,T,D]``."""

        if latent.ndim != 3 or latent.shape[-1] != self.config.d_model:
            raise ValueError("analytic distribution latent timeline must be [B,T,D]")
        typed = latent.unsqueeze(-2) + self.node_embedding.to(latent.dtype)
        return self._mixed_distribution_decode(
            typed,
            typed,
            typed,
            typed,
            constraints,
            da_anchor=da_anchor,
            da_residual_cap=da_residual_cap,
        )

    def _quantile_decode(
        self,
        market: Tensor,
        arrival_token: Tensor,
        pv_token: Tensor,
        cooling_token: Tensor,
        constraints: WorldConstraintPacket,
        *,
        da_anchor: Tensor | None = None,
        da_residual_cap: Tensor | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        prefix_dims = len(market.shape[:-3])
        price_paths = self.price_quantile_query(market).float()
        if da_anchor is None:
            # R5-V2 compatibility: channels are absolute DA and RT support
            # paths, and DART is derived from their difference.
            da = price_paths[..., 0, :]
            rt_paths = price_paths[..., 1, :]
            dart_paths = da - rt_paths
            rt = rt_paths
            dart = torch.sort(dart_paths, dim=-1).values
        else:
            if da_residual_cap is None:
                raise ValueError("a DA anchor requires a train-frozen residual cap")
            raw_da_residual = price_paths[..., 0, :]
            anchor_target = raw_da_residual[..., 0]
            anchor = _broadcast_price_anchor(da_anchor, anchor_target).unsqueeze(-1)
            cap = _broadcast_node_cap(da_residual_cap, anchor_target).unsqueeze(-1)
            da = anchor + cap * torch.tanh(raw_da_residual)
            neural_dart = price_paths[..., 1, :]
            rt_paths = da - neural_dart
            dart_paths = da - rt_paths
            # Marginal quantiles remain ordered for pinball scoring; the joint
            # paths below retain their coupling and exact price identity.
            rt = torch.sort(rt_paths, dim=-1).values
            dart = torch.sort(dart_paths, dim=-1).values

        workload_by_node = self.workload_volume_quantile_query(
            arrival_token, transform="positive"
        ).squeeze(-2)
        workload = workload_by_node.sum(dim=-2, keepdim=True)

        pv_capacity = (
            _expand_prefix(constraints.pv_capacity.float(), prefix_dims)
            .unsqueeze(-2)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        pv = self.pv_quantile_query(
            pv_token,
            transform="bounded",
            lower=torch.zeros_like(pv_capacity),
            upper=pv_capacity,
        ).squeeze(-2)

        pue_min = (
            _expand_prefix(constraints.pue_min.float(), prefix_dims)
            .unsqueeze(-2)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        pue_max = (
            _expand_prefix(constraints.pue_max.float(), prefix_dims)
            .unsqueeze(-2)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        pue = self.pue_quantile_query(
            cooling_token,
            transform="bounded",
            lower=pue_min,
            upper=pue_max,
        ).squeeze(-2)
        price = torch.cat((da, rt, dart), dim=-2)
        return {
            "price": price,
            "workload": workload,
            "pv": pv,
            "pue": pue,
            "da": da,
            "rt": rt,
            "dart": dart,
            "joint_price_paths": {"da": da, "rt": rt_paths, "dart": dart_paths},
        }

    def quantiles_from_state(
        self,
        state: WorldStatePacket,
        constraints: WorldConstraintPacket,
        *,
        da_anchor: Tensor | None = None,
        da_residual_cap: Tensor | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        batch, time, _, width = state.slots.shape
        local = state.slots[:, :, : NODES * ENTITIES_PER_NODE].reshape(
            batch, time, NODES, ENTITIES_PER_NODE, width
        )
        return self._quantile_decode(
            local[..., 0, :],
            local[..., 4, :],
            local[..., 2, :],
            local[..., 3, :],
            constraints,
            da_anchor=da_anchor,
            da_residual_cap=da_residual_cap,
        )

    def quantiles_from_latent_timeline(
        self,
        latent: Tensor,
        constraints: WorldConstraintPacket,
        *,
        da_anchor: Tensor | None = None,
        da_residual_cap: Tensor | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        typed = latent.unsqueeze(-2) + self.node_embedding.to(latent.dtype)
        return self._quantile_decode(
            typed,
            typed,
            typed,
            typed,
            constraints,
            da_anchor=da_anchor,
            da_residual_cap=da_residual_cap,
        )

    def count_distribution_from_state(
        self,
        state: WorldStatePacket,
        label_mask: Tensor | None,
    ) -> dict[str, Tensor] | None:
        """Return NB parameters only when a genuine count-label mask is present."""

        if (
            label_mask is None
            or label_mask.dtype is not torch.bool
            or not bool(label_mask.any())
        ):
            return None
        batch, time, _, width = state.slots.shape
        local = state.slots[:, :, : NODES * ENTITIES_PER_NODE].reshape(
            batch, time, NODES, ENTITIES_PER_NODE, width
        )
        count_token = local[..., 4, :].mean(dim=-2)
        raw = self.workload_count_query(count_token).float()
        mean = F.softplus(raw[..., :1]).clamp_min(1.0e-6)
        total_count = F.softplus(raw[..., 1:2]).clamp_min(1.0e-4)
        logits = torch.log(mean / total_count)
        return {
            "mean": mean,
            "total_count": total_count,
            "logits": logits,
            "valid_mask": label_mask,
        }

    def queue_distribution_from_state(
        self,
        state: WorldStatePacket,
    ) -> dict[str, Tensor]:
        """Decode a hurdle distribution from the typed queue entity token.

        ``occurrence_logits`` parameterizes the proper Bernoulli occurrence
        score. ``positive_severity`` is trained only where the frozen teacher
        trace is positive. ``mean`` is a convenient nonnegative point view;
        it is not used to collapse the two training objectives back together.
        """

        batch, time, _, width = state.slots.shape
        local = state.slots[:, :, : NODES * ENTITIES_PER_NODE].reshape(
            batch, time, NODES, ENTITIES_PER_NODE, width
        )
        queue_token = local[..., 5, :]
        occurrence_logits = self.queue_occurrence_query(queue_token).squeeze(-1).float()
        positive_severity = F.softplus(
            self.queue_positive_severity_query(queue_token).squeeze(-1).float()
        )
        occurrence_probability = torch.sigmoid(occurrence_logits)
        return {
            "occurrence_logits": occurrence_logits,
            "occurrence_probability": occurrence_probability,
            "positive_severity": positive_severity,
            "mean": occurrence_probability * positive_severity,
        }

    def queue_distribution_from_latent_timeline(
        self,
        latent: Tensor,
    ) -> dict[str, Tensor]:
        """Decode independent W1 queue endpoints with the W0 hurdle heads.

        A multi-horizon latent endpoint is already the state *after* its
        action-conditioned transition.  Replaying the operational queue
        recurrence from the issue-time queue for each shifted horizon gives
        h4/h16 the wrong initial state and creates clamp dead zones.  The typed
        hurdle heads instead provide a differentiable Bernoulli occurrence and
        positive-only severity at every endpoint, using exactly the heads that
        W0 trained.
        """

        if latent.ndim not in (3, 4) or latent.shape[-1] != self.config.d_model:
            raise ValueError("latent queue timeline must be [B,T,D] or [B,P,T,D]")
        queue_token = latent.unsqueeze(-2) + self.node_embedding.to(latent.dtype)
        occurrence_logits = self.queue_occurrence_query(queue_token).squeeze(-1).float()
        positive_severity = F.softplus(
            self.queue_positive_severity_query(queue_token).squeeze(-1).float()
        )
        occurrence_probability = torch.sigmoid(occurrence_logits)
        return {
            "occurrence_logits": occurrence_logits,
            "occurrence_probability": occurrence_probability,
            "positive_severity": positive_severity,
            "mean": occurrence_probability * positive_severity,
        }

    def soc_from_latent_timeline(self, latent: Tensor) -> Tensor:
        """Decode normalized endpoint SOC without replaying a shifted queue path."""

        if latent.ndim not in (3, 4) or latent.shape[-1] != self.config.d_model:
            raise ValueError("latent SOC timeline must be [B,T,D] or [B,P,T,D]")
        battery_token = latent.unsqueeze(-2) + self.node_embedding.to(latent.dtype)
        return torch.sigmoid(self.soc_query(battery_token).squeeze(-1).float())


__all__ = ["MonotoneQuantileHead", "QUANTILE_LEVELS", "WorldObservationQueries"]
