import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from numpy.ma.extras import ndenumerate


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.net(x)


# Compute derivatives using PyTorch autograd
def compute_loss(model, x):
    x.requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # ODE residual: -epsilon^2u"(x) + u(x) - 1
    residual = u_x - 4/(1+x**2)

    physics_loss = torch.mean(residual ** 2)  # Uniform weight (default)

    # Initial condition loss: u(0) = 0
    u0_pred = model(torch.tensor([[0.0]]))
    ic_loss = (u0_pred.pow(2)).pow(0.5)
    return physics_loss + ic_loss

def train_PINN(x_train):
    # Training the PINN
    model = PINN() 
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(10000):
        loss = compute_loss(model, x_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Full Training Epoch {epoch}, Loss: {loss.item():.6f}")

    return model

def main():
    x_list = np.random.uniform(0, 2, 50000)
    x_list = torch.tensor(x_list.reshape(-1, 1), dtype=torch.float32)
    model = train_PINN(x_list)
    one = [1]
    one = torch.tensor(one, dtype=torch.float32)
    print(model(one))

if __name__ == '__main__':
    main()