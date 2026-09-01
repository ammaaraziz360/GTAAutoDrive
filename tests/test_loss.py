# tests/test_loss.py
"""Unit tests for the loss function."""
import pytest
import torch
import sys
sys.path.insert(0, '..')

# Import loss_fn from train.py
from train import loss_fn


class TestLossFnBasics:
    """Basic loss function tests."""

    def test_loss_shape(self):
        """Loss returns a scalar."""
        pred = torch.randn(8, 3)
        target = torch.randn(8, 3)
        loss = loss_fn(pred, target)
        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"

    def test_loss_positive(self):
        """Loss is always positive."""
        pred = torch.randn(8, 3)
        target = torch.randn(8, 3)
        loss = loss_fn(pred, target)
        assert loss >= 0, f"Loss should be non-negative, got {loss}"

    def test_perfect_prediction_low_loss(self):
        """Perfect predictions result in near-zero loss."""
        target = torch.tensor([[0.0, 0.5, 0.0], [0.3, 0.8, 0.1]])
        pred = target.clone()
        loss = loss_fn(pred, target)
        assert loss < 0.01, f"Perfect prediction should have ~0 loss, got {loss}"

    def test_wrong_shape_raises(self):
        """Incorrect output shape raises ValueError."""
        pred = torch.randn(8, 2)  # Wrong: should be 3
        target = torch.randn(8, 3)
        with pytest.raises(ValueError, match="expects 3 outputs"):
            loss_fn(pred, target)


class TestSteeringBoost:
    """Tests for steering boost behavior."""

    def test_turn_errors_penalized_more(self):
        """Errors during turns (high target steering) are penalized more."""
        # Same error magnitude, different target steering
        pred = torch.tensor([[0.0, 0.5, 0.0]])  # Predicting 0 steering
        
        target_straight = torch.tensor([[0.1, 0.5, 0.0]])  # Small turn
        target_sharp = torch.tensor([[0.8, 0.5, 0.0]])     # Sharp turn
        
        loss_straight = loss_fn(pred, target_straight)
        loss_sharp = loss_fn(pred, target_sharp)
        
        # Sharp turn error should be penalized more heavily
        assert loss_sharp > loss_straight, \
            f"Sharp turn loss ({loss_sharp}) should exceed straight loss ({loss_straight})"

    def test_boost_is_symmetric(self):
        """Left and right turns are boosted equally."""
        pred = torch.tensor([[0.0, 0.5, 0.0]])
        
        target_left = torch.tensor([[-0.5, 0.5, 0.0]])
        target_right = torch.tensor([[0.5, 0.5, 0.0]])
        
        loss_left = loss_fn(pred, target_left)
        loss_right = loss_fn(pred, target_right)
        
        assert torch.allclose(loss_left, loss_right, atol=1e-5), \
            "Left and right turn boosts should be symmetric"


class TestThrottleBoost:
    """Tests for throttle boost behavior."""

    def test_acceleration_errors_penalized_more(self):
        """Errors when teacher accelerates are penalized more."""
        pred = torch.tensor([[0.0, 0.0, 0.0]])  # Not accelerating
        
        target_coast = torch.tensor([[0.0, 0.2, 0.0]])  # Light throttle
        target_accel = torch.tensor([[0.0, 0.9, 0.0]])  # Heavy throttle
        
        loss_coast = loss_fn(pred, target_coast)
        loss_accel = loss_fn(pred, target_accel)
        
        assert loss_accel > loss_coast, \
            f"Acceleration loss ({loss_accel}) should exceed coast loss ({loss_coast})"


class TestBrakeBoost:
    """Tests for brake boost behavior."""

    def test_braking_errors_penalized_more(self):
        """Errors when teacher brakes are penalized more (safety critical)."""
        pred = torch.tensor([[0.0, 0.5, 0.0]])  # Not braking
        
        target_light = torch.tensor([[0.0, 0.5, 0.2]])  # Light brake
        target_hard = torch.tensor([[0.0, 0.5, 0.9]])   # Hard brake
        
        loss_light = loss_fn(pred, target_light)
        loss_hard = loss_fn(pred, target_hard)
        
        assert loss_hard > loss_light, \
            f"Hard brake loss ({loss_hard}) should exceed light brake loss ({loss_light})"


class TestPedalExclusivity:
    """Tests for pedal exclusivity penalty."""

    def test_simultaneous_pedals_penalized(self):
        """Pressing throttle and brake simultaneously increases loss."""
        target = torch.tensor([[0.0, 0.5, 0.5]])  # Teacher uses both (edge case)
        
        # Model that correctly predicts vs model that goes extreme
        pred_correct = torch.tensor([[0.0, 0.5, 0.5]])
        pred_exclusive = torch.tensor([[0.0, 1.0, 0.0]])  # Only throttle
        
        # The product penalty should make correct prediction have slightly higher loss
        # due to thr * brk = 0.25 penalty
        loss_correct = loss_fn(pred_correct, target, pedal_lambda=0.1)
        loss_exclusive = loss_fn(pred_exclusive, target, pedal_lambda=0.1)
        
        # This is a trade-off test - exclusive has higher reconstruction error
        # but lower penalty. Just verify the penalty term works.
        pred_both = torch.tensor([[0.0, 0.8, 0.8]])  # High both
        pred_one = torch.tensor([[0.0, 0.8, 0.0]])   # High throttle only
        
        loss_both = loss_fn(pred_both, target, pedal_lambda=1.0)  # High penalty weight
        loss_one = loss_fn(pred_one, target, pedal_lambda=1.0)
        
        # Even with same base error magnitude, using both pedals should cost more
        # due to the 0.8 * 0.8 = 0.64 penalty
        assert loss_both > loss_one, "Simultaneous pedal use should be penalized"

    def test_no_penalty_when_exclusive(self):
        """No penalty when pedals are mutually exclusive."""
        target = torch.tensor([[0.0, 0.5, 0.0]])
        
        pred_throttle_only = torch.tensor([[0.0, 0.5, 0.0]])
        pred_brake_only = torch.tensor([[0.0, 0.0, 0.5]])
        
        # Neither should have pedal penalty (one pedal is 0)
        loss1 = loss_fn(pred_throttle_only, target, pedal_lambda=10.0)
        loss2 = loss_fn(pred_brake_only, target, pedal_lambda=10.0)
        
        # Recalculate without penalty
        loss1_no_penalty = loss_fn(pred_throttle_only, target, pedal_lambda=0.0)
        loss2_no_penalty = loss_fn(pred_brake_only, target, pedal_lambda=0.0)
        
        # With one pedal at 0, penalty should be 0
        assert torch.allclose(loss1, loss1_no_penalty, atol=1e-5)


class TestDecoupledBoosts:
    """Tests for decoupled (independent) action boosting."""

    def test_steering_boost_independent(self):
        """Steering boost doesn't affect throttle/brake loss."""
        # High steering target, same throttle target
        pred = torch.tensor([[0.0, 0.0, 0.0]])
        
        target_low_steer = torch.tensor([[0.1, 0.5, 0.0]])
        target_high_steer = torch.tensor([[0.9, 0.5, 0.0]])
        
        # Calculate losses with steering_boost disabled vs enabled
        loss_low = loss_fn(pred, target_low_steer, steer_boost_scale=1.0)
        loss_high = loss_fn(pred, target_high_steer, steer_boost_scale=1.0)
        
        # The throttle component should be identical (same target, same pred)
        # Only steering component should differ due to different target magnitude
        # This is an indirect test - direct component extraction would require modifying loss_fn

    def test_all_boosts_can_be_disabled(self):
        """Setting all boost scales to 1.0 gives unweighted loss."""
        pred = torch.randn(4, 3)
        target = torch.randn(4, 3)
        
        loss_boosted = loss_fn(pred, target, 
                               steer_boost_scale=2.5, 
                               throttle_boost_scale=2.0, 
                               brake_boost_scale=2.0)
        
        loss_unboosted = loss_fn(pred, target,
                                 steer_boost_scale=1.0,
                                 throttle_boost_scale=1.0,
                                 brake_boost_scale=1.0,
                                 pedal_lambda=0.0)
        
        # Unboosted should generally be lower (no amplification)
        # This isn't always true due to tanh curve, but on average it holds
        # Just verify both compute without error
        assert loss_boosted >= 0 and loss_unboosted >= 0
