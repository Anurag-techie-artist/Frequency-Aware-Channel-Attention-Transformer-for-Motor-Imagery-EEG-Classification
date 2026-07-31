"""
EEG Classifier Orchestrator Module.

Accepts FATEOutput dataclass, tuple returned from FATE, or raw CLS embedding tensor
(B, d_model) and computes class logits (B, num_classes) and PredictionOutput.
"""

import os
import time
import yaml
import logging
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Union

import torch
import torch.nn as nn

from models.classifier.classification_head import (
    ClassificationHead,
    ClassificationHeadConfig,
)
from models.transformer.frequency_aware_transformer import FATEOutput

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassifierMetadata:
    """Immutable metadata container for classifier execution metrics."""

    input_type: str
    d_model: int
    num_classes: int
    hidden_dim: int
    batch_size: int
    execution_time_ms: float


@dataclass(frozen=True)
class PredictionOutput:
    """Immutable container holding raw logits, softmax probabilities, predicted class, and metadata."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_class: torch.Tensor
    metadata: ClassifierMetadata


@dataclass
class ClassifierConfig:
    """Configuration options for EEGClassifier."""

    enabled: bool = True
    num_classes: int = 4
    hidden_dim: int = 256
    dropout: float = 0.3
    activation: str = "gelu"
    classifier_type: str = "mlp"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClassifierConfig":
        """Build config from dictionary."""
        return cls(
            enabled=bool(d.get("enabled", True)),
            num_classes=int(d.get("num_classes", 4)),
            hidden_dim=int(d.get("hidden_dim", 256)),
            dropout=float(d.get("dropout", 0.3)),
            activation=str(d.get("activation", "gelu")),
            classifier_type=str(d.get("classifier_type", "mlp")),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ClassifierConfig":
        """Load configuration from YAML file."""
        if not os.path.exists(yaml_path):
            logger.warning(
                f"Config file {yaml_path} not found. Using default Classifier configuration."
            )
            return cls()

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        model_dict = data.get("model", {}) if "model" in data else data
        clf_dict = model_dict.get("classifier", model_dict)
        return cls.from_dict(clf_dict)


class EEGClassifier(nn.Module):
    """
    EEG Classifier Module.

    Flexible classifier wrapper accepting FATEOutput, tuple output, or raw CLS embedding tensor,
    decoupling input aggregation strategies from downstream prediction logic.
    """

    def __init__(
        self,
        config: Optional[ClassifierConfig] = None,
        d_model: int = 128,
        num_classes: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        dropout: Optional[float] = None,
        activation: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        super().__init__()
        base_cfg = config or ClassifierConfig()

        self.config = ClassifierConfig(
            enabled=enabled if enabled is not None else base_cfg.enabled,
            num_classes=num_classes if num_classes is not None else base_cfg.num_classes,
            hidden_dim=hidden_dim if hidden_dim is not None else base_cfg.hidden_dim,
            dropout=dropout if dropout is not None else base_cfg.dropout,
            activation=activation if activation is not None else base_cfg.activation,
            classifier_type=base_cfg.classifier_type,
        )

        self.d_model = d_model
        head_cfg = ClassificationHeadConfig(
            d_model=d_model,
            num_classes=self.config.num_classes,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
            activation=self.config.activation,
        )
        self.head = ClassificationHead(config=head_cfg)

    def forward(
        self,
        input_data: Union[torch.Tensor, FATEOutput, Tuple[Any, ...]],
        return_metadata: bool = False,
    ) -> Union[torch.Tensor, PredictionOutput]:
        """
        Forward pass converting FATEOutput, Tuple, or CLS embedding tensor into class logits.

        Args:
            input_data: FATEOutput instance, tuple (features, FATEOutput), or torch.Tensor (B, d_model)
            return_metadata: If True, returns PredictionOutput dataclass directly

        Returns:
            Logits tensor of shape (B, num_classes) if return_metadata is False,
            otherwise PredictionOutput dataclass.
        """
        start_time = time.perf_counter()

        # Decouple input type: handle FATEOutput, tuple output, or raw Tensor
        if isinstance(input_data, FATEOutput):
            cls_embedding = input_data.cls_embedding
            input_type_str = "FATEOutput"
        elif isinstance(input_data, tuple):
            cls_embedding = None
            input_type_str = "Tuple"
            for item in input_data:
                if isinstance(item, FATEOutput):
                    cls_embedding = item.cls_embedding
                    input_type_str = "Tuple(FATEOutput)"
                    break
            if cls_embedding is None:
                for item in input_data:
                    if isinstance(item, torch.Tensor):
                        cls_embedding = item
                        input_type_str = "Tuple(Tensor)"
                        break
            if cls_embedding is None:
                raise TypeError("Could not extract FATEOutput or Tensor from input tuple")
        elif isinstance(input_data, torch.Tensor):
            cls_embedding = input_data
            input_type_str = "Tensor"
        else:
            raise TypeError(
                f"EEGClassifier expected FATEOutput, tuple, or torch.Tensor, got {type(input_data)}"
            )

        # Compute raw class logits
        logits = self.head(cls_embedding)

        if not return_metadata:
            return logits

        exec_time = (time.perf_counter() - start_time) * 1000.0
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)

        batch_size = logits.size(0) if logits.dim() > 1 else 1
        meta = ClassifierMetadata(
            input_type=input_type_str,
            d_model=self.d_model,
            num_classes=self.config.num_classes,
            hidden_dim=self.config.hidden_dim,
            batch_size=batch_size,
            execution_time_ms=exec_time,
        )

        return PredictionOutput(
            logits=logits,
            probabilities=probabilities,
            predicted_class=predicted_class,
            metadata=meta,
        )
