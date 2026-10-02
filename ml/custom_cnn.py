import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    """
    GENGROGRAM Custom CNN.

    This definition matches the architecture used to train
    the frozen 20-class production checkpoint.

    Architecture:

        Conv2D(1 -> 32)
        -> BatchNorm
        -> ReLU
        -> MaxPool

        Conv2D(32 -> 64)
        -> BatchNorm
        -> ReLU
        -> MaxPool

        Conv2D(64 -> 128)
        -> BatchNorm
        -> ReLU
        -> MaxPool

        Conv2D(128 -> 256)
        -> BatchNorm
        -> ReLU

        -> Global Average Pooling
        -> Dropout
        -> Linear(256 -> num_classes)

    Input:
        [N, 1, 128, 130]

    Production model:
        20 classes
        393,940 trainable parameters
    """

    def __init__(
        self,
        num_classes: int = 20,
        in_channels: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        # --------------------------------------------------------
        # Block 1
        # --------------------------------------------------------

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # --------------------------------------------------------
        # Block 2
        # --------------------------------------------------------

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
        )

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # --------------------------------------------------------
        # Block 3
        # --------------------------------------------------------

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(128),
        )

        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # --------------------------------------------------------
        # Block 4
        # --------------------------------------------------------

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
        )

        # --------------------------------------------------------
        # Feature aggregation
        # --------------------------------------------------------

        self.gap = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        # --------------------------------------------------------
        # Classifier
        # --------------------------------------------------------

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc = nn.Linear(
            256,
            num_classes,
        )

    # ============================================================
    # FEATURE EXTRACTION
    # ============================================================

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return convolutional feature maps before
        global average pooling.

        This is also the target feature block used
        for Grad-CAM.
        """

        # Block 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Block 4
        x = F.relu(self.conv4(x))

        return x

    # ============================================================
    # FORWARD
    # ============================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass returning class logits.
        """

        features = self.forward_features(x)

        pooled = self.gap(features)

        flat = torch.flatten(
            pooled,
            start_dim=1,
        )

        dropped = self.dropout(
            flat
        )

        logits = self.fc(
            dropped
        )

        return logits

    # ============================================================
    # PROBABILITY OUTPUT
    # ============================================================

    def predict_proba(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return softmax class probabilities.
        """

        logits = self.forward(x)

        return F.softmax(
            logits,
            dim=-1,
        )


# ================================================================
# LOCAL ARCHITECTURE TEST
# ================================================================

if __name__ == "__main__":

    model = CustomCNN(
        num_classes=20
    )

    dummy_input = torch.randn(
        4,
        1,
        128,
        130,
    )

    output = model(
        dummy_input
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("=" * 60)
    print("GENGROGRAM CUSTOM CNN")
    print("=" * 60)

    print(
        "Input shape  :",
        tuple(dummy_input.shape),
    )

    print(
        "Output shape :",
        tuple(output.shape),
    )

    print(
        "Parameters   :",
        parameter_count,
    )

    print(
        "Finite output:",
        bool(torch.isfinite(output).all()),
    )