import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

#GLOBAL CONSTANTS
k_1 = 3
k_2 = 10

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()

        self.b1 = nn.parameter.Parameter(torch.rand(k_1))
        self.b2 = nn.parameter.Parameter(torch.rand(k_2))
        self.w1 = nn.parameter.Parameter(torch.rand((k_1, k_2)))
        self.w2 = nn.parameter.Parameter(torch.rand((k_2, 1)))

    def forward(self, x):
        x = x.reshape(-1, 1)
        l1 = torch.relu(x - self.b1.reshape(1,-1))@self.w1
        l2 = torch.relu(l1 - self.b2.reshape(1,-1))@self.w2
        return l2.squeeze(-1)

def cheating_loss(model, x):
    u = model(x)
    true_u = lambda x: torch.where(
        x < 0.8,
        torch.sin(torch.pi * x / 0.8),
        5 * x - 4
    )

    return torch.mean(torch.abs(u - true_u(x)))

def train_model(x):
    model = PINN()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(10000):
        loss = cheating_loss(model, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(epoch, loss.item())
    return model

def main():
    x = torch.linspace(0, 1, 100)
    model = train_model(x)
    x_test = torch.linspace(0, 1, 1000)
    y_test = model(x_test).detach()
    plt.plot(x_test.numpy(), y_test.numpy())
    plt.show()

main()