"""
InferenceEngine Module for EEGMotorImageryModel Evaluation.

Runs forward inference in model.eval() and torch.no_grad() mode, returning logits,
probabilities, predicted class indices, extracted CLS embeddings, and ACA attention weights.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.classifier import PredictionOutput
from models.attention.adaptive_channel_attention import AttentionOutput


@dataclass(frozen=True)
class InferenceResults:
    """Immutable container holding batch inference tensors."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_class: torch.Tensor
    targets: torch.Tensor
    cls_embeddings: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None


class InferenceEngine:
    """Model inference engine running forward passes without gradient tracking."""

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()

    def run_inference(self, dataloader: DataLoader) -> InferenceResults:
        """
        Execute forward inference loop over DataLoader.

        Args:
            dataloader: PyTorch DataLoader yielding (x_batch, y_batch)

        Returns:
            InferenceResults containing concatenated tensors across batches
        """
        all_logits = []
        all_probs = []
        all_preds = []
        all_targets = []
        all_cls = []
        all_att = []

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Extract ACA attention weights if available
                if hasattr(self.model, "attention"):
                    att_res = self.model.attention(x_batch, return_attention=True)
                    if isinstance(att_res, tuple):
                        x_att, att_out = att_res
                    elif isinstance(att_res, AttentionOutput):
                        x_att, att_out = att_res.features, att_res
                    else:
                        x_att, att_out = att_res, None

                    if att_out is not None and hasattr(att_out, "attention_weights"):
                        all_att.append(att_out.attention_weights.cpu())
                else:
                    x_att = x_batch

                # Extract Transformer CLS embedding if available
                if hasattr(self.model, "transformer"):
                    _, fate_out = self.model.transformer(x_att, return_metadata=True)
                    cls_emb = fate_out.cls_embedding
                    pred = self.model.classifier(fate_out, return_metadata=True)
                else:
                    pred = self.model(x_batch, return_metadata=True)
                    cls_emb = x_batch.mean(dim=(-2, -1)) if x_batch.dim() >= 3 else x_batch

                all_logits.append(pred.logits.cpu())
                all_probs.append(pred.probabilities.cpu())
                all_preds.append(pred.predicted_class.cpu())
                all_targets.append(y_batch.cpu())
                all_cls.append(cls_emb.cpu())

        cat_logits = torch.cat(all_logits, dim=0)
        cat_probs = torch.cat(all_probs, dim=0)
        cat_preds = torch.cat(all_preds, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        cat_cls = torch.cat(all_cls, dim=0)
        cat_att = torch.cat(all_att, dim=0) if len(all_att) > 0 else None

        return InferenceResults(
            logits=cat_logits,
            probabilities=cat_probs,
            predicted_class=cat_preds,
            targets=cat_targets,
            cls_embeddings=cat_cls,
            attention_weights=cat_att,
        )
