"""Shared architecture contract for telemetry Transformer training and evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from industrial_ops_agent.predictive_maintenance.dataset_contract import (
    CONTRACT_VERSION,
    SIGNAL_ORDER,
)


@dataclass(frozen=True, slots=True)
class TimeseriesModelSpec:
    architecture_revision: str
    sequence_contract_version: str
    signal_order: tuple[str, ...]
    max_sequence_length: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float

    @classmethod
    def from_document(cls, value: dict[str, Any]) -> TimeseriesModelSpec:
        try:
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
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("timeseries_model_config_is_invalid") from exc
        spec.validate()
        return spec

    def validate(self) -> None:
        if (
            self.architecture_revision != "timeseries-transformer-v1"
            or self.sequence_contract_version != CONTRACT_VERSION
            or self.signal_order != SIGNAL_ORDER
            or not 3 <= self.max_sequence_length <= 4096
            or not 16 <= self.d_model <= 1024
            or not 1 <= self.nhead <= 32
            or self.d_model % self.nhead
            or not 1 <= self.num_layers <= 24
            or not self.d_model <= self.dim_feedforward <= 8192
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("timeseries_model_config_is_incompatible")


def build_timeseries_transformer(torch: Any, spec: TimeseriesModelSpec) -> Any:
    """Build the exact same module graph for train and independent evaluation."""

    spec.validate()
    nn = torch.nn

    class TelemetryTransformer(nn.Module):  # type: ignore[misc, name-defined]
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
            self.reconstruction_head = nn.Linear(spec.d_model, len(spec.signal_order))

        def forward(self, values: Any, padding_mask: Any) -> Any:
            encoded = self.input_projection(values)
            encoded = encoded + self.position[:, : values.shape[1], :]
            encoded = self.encoder(encoded, src_key_padding_mask=padding_mask)
            return self.reconstruction_head(self.norm(encoded))

    return TelemetryTransformer()
