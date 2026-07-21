import torch.nn as nn

class PINN(nn.Module):
    def __init__(
        self,
        width,
        depth,
        activation=nn.Tanh,
        input_dimension=2,
        output_dimension=1,
    ):
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be at least 1")

        layers = [
            nn.Linear(input_dimension, width),
            activation(),
        ]

        for _ in range(depth - 1):
            layers.extend([
                nn.Linear(width, width),
                activation(),
            ])

        layers.append(
            nn.Linear(width, output_dimension)
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)