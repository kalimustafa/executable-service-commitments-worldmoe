from __future__ import annotations

import math

import torch
from torch import nn


class TinyDGPTE(nn.Module):
    """A real, compact graph-patch Transformer used only for the demo smoke run."""

    def __init__(self, history: int, horizon: int, sites: int, quantiles: int = 5, width: int = 32) -> None:
        super().__init__()
        self.history = history
        self.horizon = horizon
        self.sites = sites
        self.quantiles = quantiles
        self.patch = nn.Conv1d(1, width, kernel_size=16, stride=8)
        layer = nn.TransformerEncoderLayer(width, nhead=4, dim_feedforward=64, dropout=0.05, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.graph_gate = nn.Parameter(torch.tensor(0.5))
        self.head = nn.Linear(width, horizon * quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch, history, sites
        batch, _, sites = x.shape
        tokens = self.patch(x.transpose(1, 2).reshape(batch * sites, 1, self.history)).transpose(1, 2)
        encoded = self.encoder(tokens).mean(dim=1).reshape(batch, sites, -1)
        adjacency = torch.full((sites, sites), 1.0 / sites, dtype=x.dtype, device=x.device)
        mixed = adjacency @ encoded
        gate = torch.sigmoid(self.graph_gate)
        encoded = gate * mixed + (1.0 - gate) * encoded
        return self.head(encoded).reshape(batch, sites, self.horizon, self.quantiles)


class DynamicGraphAttention(nn.Module):
    """Multi-head site attention constrained by a row-stochastic graph."""

    def __init__(self, width: int = 128, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("graph-attention width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, width),
        )
        self.norm2 = nn.LayerNorm(width)

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or adjacency.ndim != 3:
            raise ValueError("graph attention expects values=(B,S,D), adjacency=(B,S,S)")
        batch, sites, width = values.shape
        if adjacency.shape != (batch, sites, sites):
            raise ValueError("adjacency shape does not match graph-attention values")
        q = self.query(values).reshape(batch, sites, self.heads, self.head_width).transpose(1, 2)
        k = self.key(values).reshape(batch, sites, self.heads, self.head_width).transpose(1, 2)
        v = self.value(values).reshape(batch, sites, self.heads, self.head_width).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_width)
        scores = scores + torch.log(adjacency.clamp_min(1e-8)).unsqueeze(1)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(batch, sites, width)
        hidden = self.norm1(values + self.output(attended))
        return self.norm2(hidden + self.feed_forward(hidden))


class DGPTE(nn.Module):
    """Paper DGPT-E: dynamic-graph, patch-token, quantile Transformer.

    The paper architecture is deliberately fixed at width 128, three
    Transformer blocks, four heads, feed-forward width 256 and dropout 0.1.
    ``history`` and ``horizon`` remain parameters so shape and checkpoint tests
    can use small tensors; paper execution validates them as 672 and 96.

    Inputs may be ``(B,T,S)`` (price only) or ``(B,T,S,F)``.  Dynamic edges are
    computed exclusively from the primary historical channel supplied to the
    model, so a caller cannot accidentally construct a test-future graph.
    """

    def __init__(
        self,
        *,
        history: int = 672,
        horizon: int = 96,
        sites: int = 4,
        input_features: int = 1,
        future_features: int = 0,
        quantiles: int = 5,
        width: int = 128,
        patch_length: int = 16,
        stride: int = 8,
        transformer_blocks: int = 3,
        heads: int = 4,
        feed_forward_width: int = 256,
        dropout: float = 0.1,
        graph_layers: int = 1,
        graph_mode: str = "dynamic",
        geographic_adjacency: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if history < patch_length:
            raise ValueError("history must be at least one patch")
        if width != 128 or transformer_blocks != 3 or heads != 4:
            raise ValueError("paper DGPT-E fixes width=128, blocks=3, heads=4")
        if feed_forward_width != 256 or not math.isclose(dropout, 0.1):
            raise ValueError("paper DGPT-E fixes feed-forward width=256 and dropout=0.1")
        if graph_layers not in {1, 2}:
            raise ValueError("paper DGPT-E supports one or two graph-attention layers")
        if graph_mode not in {"dynamic", "static", "none"}:
            raise ValueError("graph_mode must be dynamic, static, or none")
        self.history = int(history)
        self.horizon = int(horizon)
        self.sites = int(sites)
        self.input_features = int(input_features)
        self.future_features = int(future_features)
        self.quantiles = int(quantiles)
        self.patch_length = int(patch_length)
        self.stride = int(stride)
        self.graph_mode = graph_mode
        self.patch = nn.Conv1d(
            self.input_features, width, kernel_size=patch_length, stride=stride
        )
        patch_count = (history - patch_length) // stride + 1
        self.position = nn.Parameter(torch.zeros(1, patch_count, width))
        nn.init.trunc_normal_(self.position, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            width,
            nhead=heads,
            dim_feedforward=feed_forward_width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_blocks)
        self.graph_mix_logit = nn.Parameter(torch.tensor(0.0))
        self.graph_layers = nn.ModuleList(
            [DynamicGraphAttention(width, heads=heads, dropout=dropout) for _ in range(graph_layers)]
        )
        if geographic_adjacency is None:
            geographic_adjacency = torch.full((sites, sites), 1.0 / sites)
        if geographic_adjacency.shape != (sites, sites):
            raise ValueError("geographic adjacency must have shape (sites, sites)")
        geographic_adjacency = geographic_adjacency.to(dtype=torch.float32)
        geographic_adjacency = geographic_adjacency.clamp_min(0)
        geographic_adjacency = geographic_adjacency / geographic_adjacency.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        self.register_buffer("geographic_adjacency", geographic_adjacency)
        self.future_projection = (
            nn.Sequential(
                nn.Linear(future_features, width),
                nn.GELU(),
                nn.Linear(width, width),
            )
            if future_features
            else None
        )
        self.head = nn.Linear(width, horizon * quantiles)

    @property
    def architecture_contract(self) -> dict[str, int | float | str]:
        return {
            "history": self.history,
            "horizon": self.horizon,
            "width": 128,
            "transformer_blocks": 3,
            "heads": 4,
            "feed_forward_width": 256,
            "dropout": 0.1,
            "patch_length": self.patch_length,
            "stride": self.stride,
            "graph_layers": len(self.graph_layers),
            "graph_mode": self.graph_mode,
        }

    def dynamic_adjacency(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim == 4:
            primary = history[..., 0]
        elif history.ndim == 3:
            primary = history
        else:
            raise ValueError("DGPT-E history must have three or four dimensions")
        if primary.shape[1:] != (self.history, self.sites):
            raise ValueError("DGPT-E history shape does not match its architecture")
        centered = primary - primary.mean(dim=1, keepdim=True)
        covariance = torch.einsum("bts,btu->bsu", centered, centered) / max(
            1, primary.shape[1] - 1
        )
        scale = torch.sqrt(torch.diagonal(covariance, dim1=-2, dim2=-1).clamp_min(1e-8))
        correlation = covariance / (
            scale.unsqueeze(-1) * scale.unsqueeze(-2)
        ).clamp_min(1e-8)
        historical = correlation.abs().clamp_min(0)
        historical = historical / historical.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        geographic = self.geographic_adjacency.to(history).expand(primary.shape[0], -1, -1)
        if self.graph_mode == "static":
            adjacency = geographic
        elif self.graph_mode == "none":
            adjacency = torch.eye(self.sites, dtype=history.dtype, device=history.device).expand(
                primary.shape[0], -1, -1
            )
        else:
            mix = torch.sigmoid(self.graph_mix_logit)
            adjacency = mix * geographic + (1.0 - mix) * historical
        if not torch.isfinite(adjacency).all():
            raise ValueError("dynamic graph contains non-finite values")
        return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(
        self,
        history: torch.Tensor,
        future_covariates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history.ndim == 3:
            history = history.unsqueeze(-1)
        expected = (self.history, self.sites, self.input_features)
        if history.ndim != 4 or history.shape[1:] != expected:
            raise ValueError(
                f"DGPT-E expects (batch,{self.history},{self.sites},{self.input_features}), "
                f"got {tuple(history.shape)}"
            )
        batch = history.shape[0]
        site_first = history.permute(0, 2, 3, 1).reshape(
            batch * self.sites, self.input_features, self.history
        )
        tokens = self.patch(site_first).transpose(1, 2) + self.position
        encoded = self.encoder(tokens).mean(dim=1).reshape(batch, self.sites, -1)
        adjacency = self.dynamic_adjacency(history)
        for layer in self.graph_layers:
            encoded = layer(encoded, adjacency)
        if self.future_projection is not None:
            if future_covariates is None or future_covariates.shape != (
                batch,
                self.horizon,
                self.sites,
                self.future_features,
            ):
                raise ValueError("future covariates do not match the configured DGPT-E shape")
            future_context = self.future_projection(future_covariates).mean(dim=1)
            encoded = encoded + future_context
        return self.head(encoded).reshape(
            batch, self.sites, self.horizon, self.quantiles
        )
