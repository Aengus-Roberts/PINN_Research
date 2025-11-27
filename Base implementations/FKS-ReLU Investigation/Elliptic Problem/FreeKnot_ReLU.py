import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.special import roots_legendre

EPSILON = 0.01
INNER_EPOCHS = 1000
OUTER_EPOCHS = 10

class ReLUKnot(nn.Module):
    def __init__(self, knot_points):
        super(ReLUKnot, self).__init__()
        self.coeffs = torch.randn(len(knot_points), dtype=torch.float32)
        self.interior_coeffs = torch.randn(len(knot_points)-2, dtype=torch.float32)
        self.knot_points = nn.Parameter(torch.tensor(knot_points, dtype=torch.float32))

    def forward(self, x):
        # x: shape (N, 1), knot_points: shape (K,) → reshape to (1, K) for broadcasting
        relus = torch.relu(x - self.knot_points.view(1, -1))  # (N, K)
        new_coeffs = torch.cat((torch.tensor([0]), self.interior_coeffs, torch.tensor([0])))
        coeffs = self.coeffs
        output = torch.matmul(relus, coeffs)
        return output

class ReLUWeight(nn.Module):
    def __init__(self, knot_points):
        super(ReLUWeight, self).__init__()
        self.coeffs = nn.Parameter(torch.randn(len(knot_points), dtype=torch.float32))
        self.interior_coeffs = nn.Parameter(torch.randn(len(knot_points)-2, dtype=torch.float32))
        self.knot_points = torch.tensor(knot_points, dtype=torch.float32)

    def forward(self, x):
        # x: shape (N, 1), knot_points: shape (K,) → reshape to (1, K) for broadcasting
        relus = torch.relu(x - self.knot_points.view(1, -1))  # (N, K)
        new_coeffs = torch.cat((torch.tensor([0]), self.interior_coeffs, torch.tensor([0])))
        coeffs = self.coeffs
        output = torch.matmul(relus, coeffs)
        return output

def compute_energy_loss(model, x, w, epsilon):
    x.requires_grad = True
    u = model(x).view(-1, 1)
    du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    integrand = (epsilon**2 / 2) * du**2 + (1/2) * u**2 - u

    # Boundary condition loss: u(0) = u(1) = 0
    u0_pred = model(torch.tensor([[0.0]], device=x.device))
    u1_pred = model(torch.tensor([[1.0]], device=x.device))
    bc_loss = u0_pred.pow(2) + u1_pred.pow(2)

    return torch.sum(w*integrand) + bc_loss


def get_knot_points(distribution, N=50):
    if distribution == "uniform":
        knot_points = torch.linspace(0, 1, N, dtype=torch.float32)

    elif distribution == "thirds":
        N_b = int(np.floor(N/3))
        N_i = N - 2*N_b
        start_knot_points = np.linspace(0, EPSILON, N_b + 1)[:-1]
        mid_knot_points = np.linspace(EPSILON, 1-EPSILON, N_i)
        end_knot_points = np.linspace(1-EPSILON, 1, N_b + 1)[1:]
        knot_points = np.concatenate((start_knot_points, mid_knot_points, end_knot_points))

    return knot_points #Returns np.array length K

# Quadrature points and weights (e.g., Gauss-Legendre)
def get_quad_points(N=500,type='uniform'):
        if type == 'uniform':
            x_quad = torch.linspace(0, 1, N).unsqueeze(1)
            w_quad = torch.tensor([1/N for _ in range(N)], dtype=torch.float32).unsqueeze(1)
        elif type == 'gauss':
            x_quad_np, w_quad_np = roots_legendre(N)
            x_quad = torch.tensor((x_quad_np + 1) / 2, dtype=torch.float32).unsqueeze(1)  # map from [-1,1] to [0,1]
            w_quad = torch.tensor(w_quad_np / 2, dtype=torch.float32).unsqueeze(1)        # adjust weights accordingly
        elif type == 'thirds':
            N_b = int(np.floor(N / 3))
            N_i = N - 2 * N_b
            start = torch.linspace(0, EPSILON, N_b + 2)[1:-1]
            mid = torch.linspace(EPSILON, 1 - EPSILON, N_i)
            end = torch.linspace(1 - EPSILON, 1, N_b + 2)[1:-1]
            x = torch.cat((start, mid, end))
            x_quad = x.unsqueeze(1)
            print(len(x_quad))
            #Trapezoidal Weightings
            w = torch.zeros_like(x)
            h = x[1:] - x[:-1]
            w[0] = 0.5 * h[0]
            w[1:-1] = 0.5 * (h[1:] + h[:-1])
            w[-1] = 0.5 * h[-1]
            w_quad = w.unsqueeze(1)


        else:
            raise ValueError("Unsupported quadrature method")

        return x_quad, w_quad #returns 2 tensors, length N


def train_models(x,w):
    knot_points = get_knot_points('uniform')
    knotModel = ReLUKnot(knot_points)
    coeffModel = ReLUWeight(knot_points)

    #Inner Training Loop
    def trainParam(model, parameter):
        if model == knotModel:
            model.coeffs = parameter
        elif model == coeffModel:
            model.knot_points = parameter
        optimiser = optim.LBFGS(model.parameters(), lr=0.05)

        def closure():
            optimiser.zero_grad()
            loss = compute_energy_loss(model, x, w, EPSILON)  # Correct call
            loss.backward()
            return loss

        for inner_epoch in range(INNER_EPOCHS):
            if inner_epoch % 500 == 0:
                print(str(outer_epoch) + ":" + str(inner_epoch))
            optimiser.step(closure)

        return model

    #Outer Training Loop
    for outer_epoch in range(OUTER_EPOCHS):
        coeffs = coeffModel.coeffs.detach().requires_grad_(True)
        knotModel = trainParam(knotModel, coeffs)
        knots = knotModel.knot_points.detach().requires_grad_(True)
        coeffModel = trainParam(coeffModel, knots)

    return coeffModel


# Plot the results
def create_results(x_test, x, w, color='red', label=''):
    model = train_models(x,w)
    y_pred = model(x_test).detach().numpy()
    plt.plot(x_test.numpy(), y_pred, label=label, color=color, linestyle='--')

def main():
    x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
    u2 = lambda x: 1 - np.cosh((x - 0.5) / EPSILON) / np.cosh(1 / (2 * EPSILON))
    y_true = np.array([u2(x) for x in x_test])
    plt.plot(x_test.numpy(), y_true, label='True Solution', color='green')

    x_uniform, w_uniform = get_quad_points(type='uniform')
    x_gauss, w_gauss = get_quad_points(type='gauss')
    x_thirds, w_thirds = get_quad_points(type='thirds')
    create_results(x_test, x_uniform, w_uniform, color='red',label='Uniform')
    create_results(x_test, x_gauss, w_gauss, color='blue',label='Gaussian')
    create_results(x_test, x_thirds, w_thirds, color='orange',label='Thirds')

    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    title = r"ReLU Architecture, Free Knots, DRM Energy, ε = {:.2f}".format(EPSILON)
    #title = r"Fixed Knots, Fixed Endpoints: $-ε^2 u''(x) + u(x) = 1$, ε = {:.5f}".format(EPSILON)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    main()




