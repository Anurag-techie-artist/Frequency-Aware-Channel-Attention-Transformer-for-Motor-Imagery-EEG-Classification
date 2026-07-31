"""
Unit Tests for ClassificationHead Module (Phase 6).

Tests:
1. test_output_shape: Verify (B, d_model) -> (B, num_classes) for 1D and 2D tensors.
2. test_deterministic_eval: Verify model.eval() produces identical logits.
3. test_serialization: Verify torch.save() & torch.load() parity.
4. test_gradient_flow: Verify loss backward computes non-zero gradients.
5. test_device_compatibility: CPU & CUDA placement.
"""

import os
import sys
import tempfile
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.classifier import ClassificationHead, ClassificationHeadConfig


class TestClassificationHead(unittest.TestCase):
    """Test suite for ClassificationHead PyTorch module."""

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.d_model = 128
        self.num_classes = 4

        self.sample_cls = torch.randn(self.batch_size, self.d_model)

    def test_output_shape(self):
        """Test output shape (B, num_classes)."""
        head = ClassificationHead(d_model=self.d_model, num_classes=self.num_classes)

        logits_2d = head(self.sample_cls)
        self.assertEqual(logits_2d.shape, (self.batch_size, self.num_classes))

        sample_1d = self.sample_cls[0]
        logits_1d = head(sample_1d)
        self.assertEqual(logits_1d.shape, (self.num_classes,))

    def test_deterministic_eval(self):
        """Test deterministic logits in eval mode."""
        head = ClassificationHead(d_model=self.d_model, dropout=0.3)
        head.eval()

        l1 = head(self.sample_cls)
        l2 = head(self.sample_cls)
        self.assertTrue(torch.equal(l1, l2))

    def test_serialization(self):
        """Test state_dict serialization parity."""
        head = ClassificationHead(d_model=self.d_model, num_classes=self.num_classes)
        head.eval()
        orig_logits = head(self.sample_cls)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt = os.path.join(tmp_dir, "head.pt")
            torch.save(head.state_dict(), ckpt)

            loaded = ClassificationHead(d_model=self.d_model, num_classes=self.num_classes)
            loaded.load_state_dict(torch.load(ckpt))
            loaded.eval()

            loaded_logits = loaded(self.sample_cls)
            self.assertTrue(torch.equal(orig_logits, loaded_logits))

    def test_gradient_flow(self):
        """Test backward pass produces non-zero gradients."""
        head = ClassificationHead(d_model=self.d_model, num_classes=self.num_classes)
        x = self.sample_cls.clone().requires_grad_(True)

        logits = head(x)
        loss = logits.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertFalse(torch.all(x.grad == 0))

        for param in head.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.isnan(param.grad).any())

    def test_device_compatibility(self):
        """Test CPU and CUDA compatibility."""
        head = ClassificationHead(d_model=self.d_model)
        out_cpu = head(self.sample_cls)
        self.assertEqual(out_cpu.device.type, "cpu")

        if torch.cuda.is_available():
            head_cuda = ClassificationHead(d_model=self.d_model).cuda()
            out_cuda = head_cuda(self.sample_cls.cuda())
            self.assertEqual(out_cuda.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
