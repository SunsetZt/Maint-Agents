"""Shared Transformer architecture for supervised RUL quantile regression."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from industrial_ops_agent.predictive_maintenance.rul_dataset_contract import (
    CONTRACT_VERSION,
    SIGNAL_ORDER,
)


@dataclass(frozen=True, slots=True)
class RulModelSpec:
    architecture_revision: str
    sequence_contract_version: str
    signal_order: tuple[str, ...]
    max_sequence_length: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    quantiles: tuple[float, float, float]

    @classmethod
    def from_document(cls, value: dict[str, Any]) -> RulModelSpec:
        try:
            quantiles = tuple(float(item) for item in value["quantiles"])
            if len(quantiles) != 3:
                raise ValueError
            spec = cls(
                architecture_revision=str(value["architecture_revision"]),
                sequence_contract_version=str(value["sequence_contract_version"]),
                signal_order=tuple(str(item) for item in value["signal_order"]),
                max_sequence_length=int(value["max_sequence_length"]),
                d_model=int(value["d_model"]),
                nhead=int(value["nhead"]),
                num_layers=int(value["num_layers"]),
                dim_feedforward=int(value["dim_feedforward"]),
                dropout=float(value["dropout"]),
                quantiles=(quantiles[0], quantiles[1], quantiles[2]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("rul_model_config_is_invalid") from exc
        spec.validate()
        return spec

    def validate(self) -> None:
        if (
            self.architecture_revision != "rul-transformer-v1"
            or self.sequence_contract_version != CONTRACT_VERSION
            or self.signal_order != SIGNAL_ORDER
            or self.quantiles != (0.1, 0.5, 0.9)
            or not 3 <= self.max_sequence_length <= 4096
            or not 16 <= self.d_model <= 1024
            or not 1 <= self.nhead <= 32
            or self.d_model % self.nhead
            or not 1 <= self.num_layers <= 24
            or not self.d_model <= self.dim_feedforward <= 8192
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("rul_model_config_is_incompatible")


def build_rul_transformer(torch: Any, spec: RulModelSpec) -> Any:
    """Build the same quantile model graph for training and independent evaluation."""

    spec.validate()
    nn = torch.nn

    class IndustrialRulTransformer(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(len(spec.signal_order), spec.d_model)
            self.position = nn.Parameter(torch.zeros(1, spec.max_sequence_length, spec.d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=spec.d_model,
                nhead=spec.nhead,
                dim_feedforward=spec.dim_feedforward,
                dropout=spec.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=spec.num_layers)
            self.norm = nn.LayerNorm(spec.d_model)
            self.quantile_head = nn.Linear(spec.d_model, len(spec.quantiles))

        def forward(self, values: Any, padding_mask: Any) -> Any:
            encoded = self.input_projection(values)
            encoded = encoded + self.position[:, : values.shape[1], :]
            encoded = self.norm(self.encoder(encoded, src_key_padding_mask=padding_mask))
            valid = (~padding_mask).unsqueeze(-1)
            pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            raw = self.quantile_head(pooled)
            median = nn.functional.softplus(raw[:, 1])
            lower = (median - nn.functional.softplus(raw[:, 0])).clamp_min(0)
            upper = median + nn.functional.softplus(raw[:, 2])
            return torch.stack((lower, median, upper), dim=1)

    return IndustrialRulTransformer()
