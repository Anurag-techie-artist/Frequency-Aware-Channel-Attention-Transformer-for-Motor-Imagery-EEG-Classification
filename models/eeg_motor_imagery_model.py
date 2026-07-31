"""
End-to-End EEGMotorImageryModel Assembly.

Assembles Adaptive Channel Attention (ACA), Frequency-Aware Transformer Encoder (FATE),
and EEGClassifier into a unified PyTorch neural network model.

Pipeline Flow:
    X (B, F, C, S)
          │
          ▼
    AdaptiveChannelAttention (ACA)
          │
          ▼
    FrequencyAwareTransformer (FATE)
          │
          ▼
    EEGClassifier (MLP Head)
          │
          ▼
    PredictionOutput / Logits (B, num_classes)
"""

import logging
from typing import Dict, Any, Optional, Union, Tuple

import torch
import torch.nn as nn

from models.attention import ACA, AdaptiveChannelAttentionConfig
from models.transformer import FATE, FrequencyAwareTransformerConfig
from models.classifier import EEGClassifier, ClassifierConfig, PredictionOutput

logger = logging.getLogger(__name__)


class EEGMotorImageryModel(nn.Module):
    """
    End-to-End Motor Imagery EEG Classification Model.

    Combines ACA channel attention, FATE transformer encoder, and MLP classifier head.
    """

    def __init__(
        self,
        attention_config: Optional[AdaptiveChannelAttentionConfig] = None,
        transformer_config: Optional[FrequencyAwareTransformerConfig] = None,
        classifier_config: Optional[ClassifierConfig] = None,
        num_channels: int = 133,
        num_bands: int = 4,
        d_model: int = 128,
        num_classes: int = 4,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_bands = num_bands
        self.d_model = d_model
        self.num_classes = num_classes

        # Instantiate ACA attention module
        att_cfg = attention_config or AdaptiveChannelAttentionConfig()
        self.attention = ACA(
            config=att_cfg, num_channels=num_channels, num_bands=num_bands
        )

        # Instantiate FATE transformer encoder module
        fate_cfg = transformer_config or FrequencyAwareTransformerConfig()
        self.transformer = FATE(config=fate_cfg, d_model=d_model)

        # Instantiate EEGClassifier head module
        clf_cfg = classifier_config or ClassifierConfig()
        self.classifier = EEGClassifier(
            config=clf_cfg, d_model=d_model, num_classes=num_classes
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EEGMotorImageryModel":
        """
        Build EEGMotorImageryModel directly from configuration dictionary.

        Args:
            config: Master dictionary containing model/attention/transformer/classifier parameters.

        Returns:
            Instantiated EEGMotorImageryModel module.
        """
        model_dict = config.get("model", config)

        att_cfg = AdaptiveChannelAttentionConfig.from_dict(
            model_dict.get("attention", {})
        )
        fate_cfg = FrequencyAwareTransformerConfig.from_dict(
            model_dict.get("transformer", {})
        )
        clf_cfg = ClassifierConfig.from_dict(model_dict.get("classifier", {}))

        d_model = fate_cfg.transformer.d_model
        num_classes = clf_cfg.num_classes

        return cls(
            attention_config=att_cfg,
            transformer_config=fate_cfg,
            classifier_config=clf_cfg,
            d_model=d_model,
            num_classes=num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_metadata: bool = False,
    ) -> Union[torch.Tensor, PredictionOutput]:
        """
        Forward pass for EEGMotorImageryModel.

        Args:
            x: Input multi-band EEG tensor of shape (B, F, C, S) or (F, C, S)
            return_metadata: If True, returns PredictionOutput dataclass directly

        Returns:
            Logits tensor of shape (B, num_classes) if return_metadata is False,
            otherwise PredictionOutput dataclass.
        """
        # 1. Adaptive Channel Attention (B, F, C, S) -> (B, F, C, S)
        x_att = self.attention(x)

        # 2. Frequency-Aware Transformer Encoder (B, F, C, S) -> FATEOutput
        _, fate_out = self.transformer(x_att, return_metadata=True)

        # 3. Classifier Head -> Logits (B, num_classes) or PredictionOutput
        prediction = self.classifier(fate_out, return_metadata=return_metadata)

        return prediction
