# tests/test_model.py
"""Unit tests for StudentPolicy model."""
import pytest
import torch
import torch.nn as nn
from model import StudentPolicy


class TestStudentPolicyArchitecture:
    """Tests for model architecture and initialization."""

    def test_single_frame_input(self):
        """Model accepts single-frame input (3 channels)."""
        model = StudentPolicy(in_ch=3)
        x = torch.randn(2, 3, 260, 260)
        out = model(x)
        assert out.shape == (2, 3), f"Expected (2, 3), got {out.shape}"

    def test_temporal_stack_input(self):
        """Model accepts 3-frame temporal stack (9 channels)."""
        model = StudentPolicy(in_ch=9)
        x = torch.randn(2, 9, 260, 260)
        out = model(x)
        assert out.shape == (2, 3), f"Expected (2, 3), got {out.shape}"

    def test_aux_tlight_output(self):
        """Model returns traffic light logits when aux_tlight=True."""
        model = StudentPolicy(in_ch=3, aux_tlight=True)
        x = torch.randn(2, 3, 260, 260)
        out, tlogits = model(x)
        assert out.shape == (2, 3)
        assert tlogits.shape == (2, 4), f"Expected (2, 4) for traffic light classes, got {tlogits.shape}"

    def test_parameter_count(self):
        """Model has expected parameter count (sanity check for EfficientNet-B2)."""
        model = StudentPolicy(in_ch=9)
        total = sum(p.numel() for p in model.parameters())
        # EfficientNet-B2 has ~9M params, heads add ~2M more
        assert 8_000_000 < total < 15_000_000, f"Unexpected param count: {total:,}"


class TestStudentPolicyOutputRanges:
    """Tests for output value constraints."""

    @pytest.fixture
    def model(self):
        return StudentPolicy(in_ch=3).eval()

    def test_steering_range(self, model):
        """Steering output is in [-1, 1] (tanh activation)."""
        x = torch.randn(10, 3, 260, 260)
        with torch.no_grad():
            out = model(x)
        steer = out[:, 0]
        assert steer.min() >= -1.0, f"Steering below -1: {steer.min()}"
        assert steer.max() <= 1.0, f"Steering above 1: {steer.max()}"

    def test_throttle_range(self, model):
        """Throttle output is in [0, 1] (sigmoid activation)."""
        x = torch.randn(10, 3, 260, 260)
        with torch.no_grad():
            out = model(x)
        thr = out[:, 1]
        assert thr.min() >= 0.0, f"Throttle below 0: {thr.min()}"
        assert thr.max() <= 1.0, f"Throttle above 1: {thr.max()}"

    def test_brake_range(self, model):
        """Brake output is in [0, 1] (sigmoid activation)."""
        x = torch.randn(10, 3, 260, 260)
        with torch.no_grad():
            out = model(x)
        brk = out[:, 2]
        assert brk.min() >= 0.0, f"Brake below 0: {brk.min()}"
        assert brk.max() <= 1.0, f"Brake above 1: {brk.max()}"


class TestStudentPolicyGradients:
    """Tests for gradient flow (critical for training stability)."""

    def test_gradients_flow_to_backbone(self):
        """Gradients propagate back to the backbone (no dead layers)."""
        model = StudentPolicy(in_ch=3)
        x = torch.randn(2, 3, 260, 260)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Check backbone first conv has gradients
        first_conv = model.backbone[0][0]
        assert first_conv.weight.grad is not None, "No gradients in backbone first conv"
        assert first_conv.weight.grad.abs().sum() > 0, "Gradients are zero in backbone"

    def test_gradients_flow_to_steering_head(self):
        """Gradients propagate to steering head."""
        model = StudentPolicy(in_ch=3)
        x = torch.randn(2, 3, 260, 260)
        out = model(x)
        # Only backprop through steering
        loss = out[:, 0].sum()
        loss.backward()

        assert model.steer_fc1.weight.grad is not None
        assert model.steer_fc1.weight.grad.abs().sum() > 0

    def test_gradients_flow_to_pedal_head(self):
        """Gradients propagate to pedal head."""
        model = StudentPolicy(in_ch=3)
        x = torch.randn(2, 3, 260, 260)
        out = model(x)
        # Only backprop through throttle + brake
        loss = out[:, 1:].sum()
        loss.backward()

        # Check first layer of pedal_head Sequential
        pedal_fc1 = model.pedal_head[0]
        assert pedal_fc1.weight.grad is not None
        assert pedal_fc1.weight.grad.abs().sum() > 0

    def test_skip_connection_helps_gradient(self):
        """Skip connection in steering head improves gradient magnitude."""
        model = StudentPolicy(in_ch=3)
        x = torch.randn(2, 3, 260, 260)
        out = model(x)
        loss = out[:, 0].sum()
        loss.backward()

        # The shortcut should have meaningful gradients
        assert model.steer_shortcut.weight.grad is not None
        shortcut_grad_norm = model.steer_shortcut.weight.grad.norm().item()
        assert shortcut_grad_norm > 1e-6, f"Shortcut gradient too small: {shortcut_grad_norm}"


class TestStudentPolicyDeterminism:
    """Tests for reproducibility."""

    def test_eval_mode_deterministic(self):
        """Model produces same output in eval mode (no dropout randomness)."""
        model = StudentPolicy(in_ch=3).eval()
        x = torch.randn(2, 3, 260, 260)
        
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_train_mode_has_dropout(self):
        """Model produces different outputs in train mode (dropout active)."""
        model = StudentPolicy(in_ch=3).train()
        x = torch.randn(2, 3, 260, 260)
        
        torch.manual_seed(42)
        out1 = model(x)
        torch.manual_seed(123)
        out2 = model(x)
        
        # With dropout, outputs should differ (unless extremely unlucky)
        assert not torch.allclose(out1, out2), "Train mode should have randomness from dropout"


class TestTemporalWeightInit:
    """Tests for temporal frame weight initialization."""

    def test_temporal_weights_scaled(self):
        """Temporal conv weights are scaled by 1/num_frames."""
        model_single = StudentPolicy(in_ch=3)
        model_triple = StudentPolicy(in_ch=9)

        w_single = model_single.backbone[0][0].weight.data  # [out, 3, k, k]
        w_triple = model_triple.backbone[0][0].weight.data  # [out, 9, k, k]

        # Each 3-channel slice should be ~1/3 of original
        # (within tolerance due to potential minor variations)
        slice_0 = w_triple[:, 0:3, :, :]
        slice_1 = w_triple[:, 3:6, :, :]
        slice_2 = w_triple[:, 6:9, :, :]

        # All three slices should be approximately equal (same pretrained weights)
        assert torch.allclose(slice_0, slice_1, atol=1e-5), "Temporal slices should match"
        assert torch.allclose(slice_1, slice_2, atol=1e-5), "Temporal slices should match"

        # Each slice should be ~1/3 of the single-frame weights
        expected = w_single / 3.0
        assert torch.allclose(slice_0, expected, atol=1e-5), "Temporal weights should be scaled by 1/3"
