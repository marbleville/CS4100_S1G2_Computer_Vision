"""
Tests for GestureCNN architecture.

Verifies output shapes, parameter count, forward pass behavior,
training/eval mode differences, and edge cases.
"""

import pytest
import torch
import torch.nn as nn
from classifier.models.cnn import GestureCNN
from classifier.config import STATIC_GESTURE_CLASSES


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
# Output shape tests                                                   #
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
# Output value tests                                                   #
# ------------------------------------------------------------------ #

def test_output_is_logits_not_probabilities(model, dummy_batch):
    """Output should be raw logits — values should not sum to 1 per sample."""
    out = model(dummy_batch)
    row_sums = out.sum(dim=1)
    # If softmax were applied, each row would sum to ~1.0
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


# ------------------------------------------------------------------ #
# Architecture tests                                                   #
# ------------------------------------------------------------------ #

def test_parameter_count(model):
    """Model should have roughly 8-9 million trainable parameters."""
    num_params = model.get_num_params()
    assert 7_000_000 < num_params < 10_000_000, (
        f"Unexpected parameter count: {num_params}. "
        "Architecture may have changed — verify FLATTEN_SIZE is still correct."
    )


def test_flatten_size_constant():
    """FLATTEN_SIZE constant should match actual flattened output of conv blocks."""
    model = GestureCNN()
    x = torch.randn(1, 3, 128, 128)
    x = model.block1(x)
    x = model.block2(x)
    x = model.block3(x)
    actual_flatten = x.shape[1] * x.shape[2] * x.shape[3]
    assert actual_flatten == GestureCNN.FLATTEN_SIZE, (
        f"FLATTEN_SIZE constant {GestureCNN.FLATTEN_SIZE} does not match "
        f"actual flattened size {actual_flatten}"
    )


def test_block_output_shapes(model):
    """Each conv block should halve spatial dimensions."""
    x = torch.randn(2, 3, 128, 128)

    x = model.block1(x)
    assert x.shape == (2, 32, 64, 64), f"Block 1 output shape wrong: {x.shape}"

    x = model.block2(x)
    assert x.shape == (2, 64, 32, 32), f"Block 2 output shape wrong: {x.shape}"

    x = model.block3(x)
    assert x.shape == (2, 128, 16, 16), f"Block 3 output shape wrong: {x.shape}"


def test_all_parameters_are_trainable(model):
    """All parameters should require gradients by default."""
    for name, param in model.named_parameters():
        assert param.requires_grad, f"Parameter {name} does not require grad"


# ------------------------------------------------------------------ #
# Training vs eval mode tests                                          #
# ------------------------------------------------------------------ #

def test_eval_mode_is_deterministic(model, single_image):
    """In eval mode, two forward passes on the same input should be identical."""
    model.eval()
    with torch.no_grad():
        out1 = model(single_image)
        out2 = model(single_image)
    assert torch.allclose(out1, out2), (
        "Eval mode output is not deterministic — dropout may still be active"
    )


def test_train_mode_is_nondeterministic(model, single_image):
    """In train mode, dropout should make repeated forward passes differ."""
    model.train()
    out1 = model(single_image)
    out2 = model(single_image)
    # With dropout(0.5) outputs should differ — if they are identical
    # dropout is not working correctly
    assert not torch.allclose(out1, out2), (
        "Train mode outputs are identical — dropout may not be active"
    )


# ------------------------------------------------------------------ #
# Input validation tests                                               #
# ------------------------------------------------------------------ #

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