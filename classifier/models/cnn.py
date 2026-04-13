"""
GestureCNN: Convolutional neural network for static hand gesture classification.

Takes a batch of normalized 128x128 RGB hand crops and outputs raw logits
(one per gesture class). Softmax is NOT applied here — it is applied at
inference time in static_classifier.py.
"""

import torch
import torch.nn as nn


class GestureCNN(nn.Module):
    """
    CNN for static hand gesture classification.

    Architecture: 3 convolutional blocks (Conv -> BatchNorm -> ReLU -> MaxPool)
    followed by a fully connected classifier head.

    Input:  (batch_size, 3, 128, 128) — normalized RGB hand crops from Module B
    Output: (batch_size, num_classes) — raw logits, one per gesture class
    """

    # Spatial size after 3x MaxPool(2x2) on a 128x128 input: 128 -> 64 -> 32 -> 16
    FLATTEN_SIZE = 128 * 16 * 16  # 32768

    def __init__(self, num_classes: int = 4):
        """
        Args:
            num_classes: Number of gesture classes to classify.
                         Defaults to 4 (fist, open, thumbs_up, thumbs_down).
        """
        super().__init__()
        self.num_classes = num_classes

        # Block 1: (batch, 3, 128, 128) -> (batch, 32, 64, 64)
        # padding=1 keeps spatial size constant after conv so only MaxPool halves it
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),   # stabilizes training, reduces sensitivity to learning rate
            nn.ReLU(),            # introduces non-linearity so layers can learn complex patterns
            nn.MaxPool2d(kernel_size=2, stride=2),  # halves spatial dims: 128 -> 64
        )

        # Block 2: (batch, 32, 64, 64) -> (batch, 64, 32, 32)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # halves spatial dims: 64 -> 32
        )

        # Block 3: (batch, 64, 32, 32) -> (batch, 128, 16, 16)
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # halves spatial dims: 32 -> 16
        )

        # Classifier head: flattened features -> gesture logits
        # Linear(32768, 256) learns which combinations of spatial features predict each gesture
        # Dropout(0.5) zeros 50% of activations during training to prevent overfitting
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # (batch, 128, 16, 16) -> (batch, 32768)
            nn.Linear(self.FLATTEN_SIZE, 256),
            nn.ReLU(),
            nn.Dropout(0.5),                     # disabled automatically during model.eval()
            nn.Linear(256, num_classes),         # one raw score per gesture class
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 128, 128).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
            Softmax is NOT applied — use nn.CrossEntropyLoss during training
            and apply softmax manually at inference time.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

    def get_num_params(self) -> int:
        """
        Return the total number of trainable parameters.
        Useful for tracking model size and Pi optimization decisions.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)