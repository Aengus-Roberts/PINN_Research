import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.special import roots_legendre

EPSILON = 0.05

class FKS(nn.Module):
    def __init__(self, knot_points):
        super(FKS, self).__init__()
        self.coeffs = nn.Parameter(torch.randn(len(knot_points), dtype=torch.float32))
        self.interior_coeffs = nn.Parameter(torch.ones(len(knot_points)-2, dtype=torch.float32))
        self.knot_points = torch.tensor(knot_points, dtype=torch.float32)
        self.kminus = self.knot_points[:-2]
        self.ki = self.knot_points[1:-1]
        self.kplus = self.knot_points[2:]

    def interior_spline(self, x):
        alpha = 1/(self.ki-self.kminus)
        beta = (self.kplus - self.kminus)/((self.kplus - self.ki)*(self.ki-self.kminus))
        gamma = 1/(self.kplus - self.ki)
        xT = x.reshape(-1,1)

        output = torch.relu(xT-self.kminus)*alpha - torch.relu(xT-self.ki)*beta + torch.relu(xT-self.kplus)*gamma
        return output

    def left_spline(self,x):
        k0 = self.knot_points[0]
        k1 = self.knot_points[1]
        return torch.relu(k1-x)/(k1-k0)

    def right_spline(self,x):
        kminus = self.knot_points[-2]
        kfinal = self.knot_points[-1]
        return torch.relu(x-kminus)/(kfinal-kminus)

    def forward(self, x):
        # x: shape (N, 1), knot_points: shape (K,) → reshape to (1, K) for broadcasting
        # FKS: shape (N,K)
        FKS = torch.zeros(len(x),len(self.knot_points), dtype=torch.float32, device=x.device)
        FKS[:, 0] = self.left_spline(x).squeeze()  # first column
        FKS[:, -1] = self.right_spline(x).squeeze() # last column
        FKS[:, 1:-1] = self.interior_spline(x)
        new_coeffs = torch.cat((torch.tensor([0]),self.interior_coeffs,torch.tensor([0])))
        coeffs = new_coeffs
        output = torch.matmul(FKS, coeffs)
        return output

def compute_energy_loss(model, x, w, epsilon):
    x.requires_grad = True
    u = model(x).view(-1, 1)
    du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    integrand = torch.exp_(-x/epsilon) * ((epsilon/2)*du**2 - u)

    # Boundary condition loss: u(0) = u(1) = 0
    u0_pred = model(torch.tensor([[0.0]], device=x.device))
    u1_pred = model(torch.tensor([[1.0]], device=x.device))
    bc_loss = u0_pred.pow(2) + u1_pred.pow(2)

    return torch.sum(w*integrand) + bc_loss

def cheating_loss(model, x, w, epsilon):
    u = model(x).view(-1, 1)
    u_true = lambda x: 1 - torch.cosh((x - 0.5) / EPSILON) / np.cosh(1 / (2 * EPSILON))

    interior_loss = torch.sum(w*(u - u_true(x))**2)**0.5

    u0_pred = model(torch.tensor([[0.0]], device=x.device))
    u1_pred = model(torch.tensor([[1.0]], device=x.device))
    bc_loss = u0_pred.pow(2) + u1_pred.pow(2)

    return  interior_loss + bc_loss


def get_knot_points(distribution, N=200):
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
        elif type == 'right':
            N_L = int(np.floor(N / 2))
            N_R = N - N_L
            left = torch.linspace(0, 1 - EPSILON, N_L + 1)[:-1]
            right = torch.linspace(1 - EPSILON, 1, N_R)
            x = torch.cat((left, right))
            x_quad = x.unsqueeze(1)
            # Trapezoidal Weightings
            w = torch.zeros_like(x)
            h = x[1:] - x[:-1]
            w[0] = 0.5 * h[0]
            w[1:-1] = 0.5 * (h[1:] + h[:-1])
            w[-1] = 0.5 * h[-1]
            w_quad = w.unsqueeze(1)


        else:
            raise ValueError("Unsupported quadrature method")

        return x_quad, w_quad #returns 2 tensors, length N

def train_model(x, w):
    knot_points = get_knot_points('thirds')
    model = FKS(knot_points)
    #optimiser = optim.Adam(model.parameters(), lr = 0.05)
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    # Continue training on the full dataset
    for epoch in range(20000):
        loss = compute_energy_loss(model, x, w, EPSILON)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if epoch % 500 == 0:
            print(f"Full Training Epoch {epoch}, Loss: {loss.item():.6f}")
    """
    optimiser = optim.LBFGS(model.parameters(), lr=0.001, max_iter=10000)

    def closure():
        optimiser.zero_grad()
        loss = compute_energy_loss(model, x, w, EPSILON)  # Correct call
        loss.backward()
        return loss

    optimiser.step(closure)

        #if epoch % 500 == 0:
            #print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    """
    return model

# Plot the results
def create_results(x_test, x, w, color='red', label=''):
    model = train_model(x,w)
    y_pred = model(x_test).detach().numpy()
    plt.plot(x_test.numpy(), y_pred, label=label, color=color, linestyle='--')

def main():
    x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
    B = 1 / (1 - np.exp(1 / EPSILON))
    u1 = lambda x:x - B * (1 - np.exp(x / EPSILON))
    y_true = np.array([u1(x) for x in x_test])
    plt.plot(x_test.numpy(), y_true, label='True Solution', color='green')

    x_uniform, w_uniform = get_quad_points(type='uniform')
    #x_gauss, w_gauss = get_quad_points(type='gauss')
    #x_thirds, w_thirds = get_quad_points(type='thirds')
    #x_right, w_right = get_quad_points(type='right')
    create_results(x_test, x_uniform, w_uniform, color='red',label='Uniform')
    #create_results(x_test, x_gauss, w_gauss, color='blue',label='Gaussian')
    #create_results(x_test, x_thirds, w_thirds, color='orange',label='Thirds')
    #create_results(x_test, x_right, w_right, color='blue', label='Right')


    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    title = r"Linear Spline, Fixed Knots, DRM Energy, ε = {:.4f}".format(EPSILON)
    #title = r"Fixed Knots, Fixed Endpoints: $-ε^2 u''(x) + u(x) = 1$, ε = {:.5f}".format(EPSILON)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    main()




