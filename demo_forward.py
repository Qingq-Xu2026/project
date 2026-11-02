from __future__ import annotations

import torch

from vit_yolo import ViTYOLO


def main() -> None:
    model = ViTYOLO(
        num_classes=10,
        neck_channels=64,
        bifpn_depth=1,
        width_mult=0.25,
        depth_mult=0.5,
        max_tokens=64,
    )
    model.eval()
    image = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        outputs = model(image)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters: {parameter_count:,}")
    for level, output in enumerate(outputs, start=3):
        print(f"P{level}: {tuple(output.shape)}")


if __name__ == "__main__":
    main()

