"""
Tests for GestureCNN architecture.

Verifies output shapes, forward pass behavior,
training/eval mode differences, and edge cases.
"""

import pytest
import torch
import torch.nn as nn
from classifier.models.cnn import GestureCNN
from classifier.config import STATIC_GESTURE_CLASSES

# Fixtures

@pytest.fixture
def model():
    """Default model with 4 gesture classes."""
    return GestureCNN(num_classes=len(STATIC_GESTURE_CLASSES))

@pytest.fixture
def dummy_batch():
    """Batch of 8 random normalized images matching Module B crop format."""
    return torch.randn(8, 3, 128, 128)

@pytest.fixture
def single_image():
    """Single image as it would arrive at inference time."""
    return torch.randn(1, 3, 128, 128)

# Output shape tests

def test_output_shape_batch(model, dummy_batch):
    """Forward pass with a batch produces correct output shape."""
    out = model(dummy_batch)
    assert out.shape == (8, len(STATIC_GESTURE_CLASSES)), (
        f"Expected (8, {len(STATIC_GESTURE_CLASSES)}), got {out.shape}"
    )

def test_output_shape_single_image(model, single_image):
    """Forward pass with a single image produces correct output shape."""
    out = model(single_image)
    assert out.shape == (1, len(STATIC_GESTURE_CLASSES)), (
        f"Expected (1, {len(STATIC_GESTURE_CLASSES)}), got {out.shape}"
    )

def test_output_shape_custom_num_classes():
    """Model correctly adjusts output size for custom num_classes."""
    for num_classes in [2, 4, 6, 10]:
        m = GestureCNN(num_classes=num_classes)
        x = torch.randn(4, 3, 128, 128)
        out = m(x)
        assert out.shape == (4, num_classes), (
            f"Expected (4, {num_classes}), got {out.shape}"
        )

# Output value tests

def test_output_is_logits_not_probabilities(model, dummy_batch):
    """Output should be raw logits — values should not sum to 1 per sample."""
    out = model(dummy_batch)
    row_sums = out.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones(8), atol=1e-3), (
        "Output looks like probabilities — softmax should not be applied in forward()"
    )

def test_output_contains_no_nan(model, dummy_batch):
    """Forward pass should never produce NaN values."""
    out = model(dummy_batch)
    assert not torch.isnan(out).any(), "Output contains NaN values"

def test_output_contains_no_inf(model, dummy_batch):
    """Forward pass should never produce infinite values."""
    out = model(dummy_batch)
    assert not torch.isinf(out).any(), "Output contains infinite values"

# Parameter tests

def test_has_trainable_parameters(model):
    """Model must have at least one trainable parameter."""
    num_params = model.get_num_params()
    assert num_params > 0, "Model has no trainable parameters"

def test_get_num_params_is_consistent(model):
    """get_num_params should return the same value on repeated calls."""
    assert model.get_num_params() == model.get_num_params()

def test_output_shape_is_independent_of_parameter_count(model, dummy_batch):
    """Output shape should be (batch, num_classes) regardless of parameter count."""
    out = model(dummy_batch)
    assert out.shape[0] == dummy_batch.shape[0], "Batch dimension mismatch"
    assert out.shape[1] == len(STATIC_GESTURE_CLASSES), "Class dimension mismatch"

# Forward pass behavior tests

def test_forward_pass_does_not_modify_input(model, dummy_batch):
    """Forward pass should not modify the input tensor in place."""
    original = dummy_batch.clone()
    model(dummy_batch)
    assert torch.allclose(dummy_batch, original), "Input tensor was modified during forward pass"

def test_model_output_changes_with_different_inputs(model):
    """Different inputs should produce different outputs."""
    x1 = torch.randn(1, 3, 128, 128)
    x2 = torch.randn(1, 3, 128, 128)
    model.eval()
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
    assert not torch.allclose(out1, out2), (
        "Different inputs produced identical outputs — model may be degenerate"
    )

# Training vs eval mode tests

def test_eval_mode_is_deterministic(model, single_image):
    """Same input in eval mode must always produce the same output."""
    model.eval()
    with torch.no_grad():
        out1 = model(single_image)
        out2 = model(single_image)
    assert torch.allclose(out1, out2), (
        "Eval mode is not deterministic — model should produce identical "
        "outputs for the same input when not in training mode"
    )

def test_train_and_eval_outputs_can_differ(model, single_image):
    """Train and eval mode may produce different outputs — both must have correct shape."""
    model.train()
    out_train = model(single_image)
    model.eval()
    with torch.no_grad():
        out_eval = model(single_image)
    assert out_train.shape == (1, len(STATIC_GESTURE_CLASSES))
    assert out_eval.shape == (1, len(STATIC_GESTURE_CLASSES))

# Input validation tests

def test_wrong_input_channels_raises(model):
    """Grayscale input (1 channel) should raise an error."""
    x = torch.randn(2, 1, 128, 128)
    with pytest.raises(Exception):
        model(x)

def test_wrong_spatial_size_raises(model):
    """Input with wrong spatial dimensions should raise an error."""
    x = torch.randn(2, 3, 64, 64)
    with pytest.raises(Exception):
        model(x)