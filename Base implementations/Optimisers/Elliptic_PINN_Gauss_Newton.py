import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torchmin import least_squares
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from numpy.polynomial.legendre import Legendre

EPSILON = .01


# Defined PINN via PyTorch Structure, 2 Hidden Layers
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 100),
            nn.Tanh(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        return self.net(x)


# Compute derivatives using PyTorch autograd
def compute_loss(model, x, weights=None, EPSILON=EPSILON):
    x.requires_grad_(True)
    u = model(x).view(-1,1)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # ODE residual: -epsilon^2u"(x) + u(x) - 1
    residual = -(EPSILON ** 2) * u_xx + u - 1

    # Compute weighted physics loss if weights are provided
    if weights is not None:
        physics_loss = torch.sum(weights * residual ** 2)  # Weighted sum
    else:
        physics_loss = torch.mean(residual ** 2)  # Uniform weight (default)

    # Boundary condition loss: u(0) = u(1) = 0
    u0_pred = model(torch.tensor([[0.0]], device=x.device))
    u1_pred = model(torch.tensor([[1.0]], device=x.device))
    bc_loss = u0_pred.pow(2) + u1_pred.pow(2)

    return physics_loss + bc_loss


# Gauss-Newton utilities for PINNs
def parameter_shapes(model):
    return {name: param.shape for name, param in model.named_parameters()}


def vector_to_parameter_dict(theta, shapes):
    params = {}
    pointer = 0
    for name, shape in shapes.items():
        numel = int(np.prod(shape))
        params[name] = theta[pointer:pointer + numel].view(shape)
        pointer += numel
    return params


def compute_residual_vector(model, params, x, weights=None, epsilon=EPSILON, bc_weight=100.0):
    x = x.detach().clone().requires_grad_(True)

    u = functional_call(model, params, (x,)).view(-1, 1)
    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]
    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
    )[0]

    residual = -(epsilon ** 2) * u_xx + u - 1.0

    if weights is not None:
        residual = torch.sqrt(weights) * residual

    x0 = torch.tensor([[0.0]], dtype=x.dtype, device=x.device)
    x1 = torch.tensor([[1.0]], dtype=x.dtype, device=x.device)
    u0 = functional_call(model, params, (x0,)).view(-1)
    u1 = functional_call(model, params, (x1,)).view(-1)

    bc_scale = torch.sqrt(torch.tensor(bc_weight, dtype=x.dtype, device=x.device))
    bc_residual = bc_scale * torch.cat([u0, u1])

    return torch.cat([residual.view(-1), bc_residual])


def train_PINN_gauss_newton(x_train, weights, epsilon=EPSILON):
    model = PINN().double()
    x_train = x_train.double()
    weights = weights.double()

    theta0 = parameters_to_vector(model.parameters()).detach().clone().requires_grad_(True)
    shapes = parameter_shapes(model)

    def residual_fn(theta):
        params = vector_to_parameter_dict(theta, shapes)
        return compute_residual_vector(
            model,
            params,
            x_train[1:-1],
            weights[1:-1],
            epsilon=epsilon,
            bc_weight=100.0,
        )

    result = least_squares(
        residual_fn,
        theta0,
        method='trf',
        tr_solver='exact',
        max_nfev=200,
        verbose=2,
    )

    vector_to_parameters(result.x.detach(), model.parameters())
    residual_norm = torch.linalg.norm(residual_fn(result.x.detach())).item()
    print(f"Gauss-Newton final residual norm: {residual_norm:.6e}")

    return model, [residual_norm]


def gauss_lobatto_nodes_weights(n):
    # Compute the Gauss-Lobatto nodes and weights on the interval [-1,1] for n nodes.
    if n < 2:
        raise ValueError("n must be at least 2.")

    # Endpoints are fixed
    x = np.zeros(n)
    x[0] = -1.0
    x[-1] = 1.0

    if n > 2:
        # The interior nodes are the roots of the derivative of the (n-1)th Legendre polynomial
        P = Legendre.basis(n - 1)
        dP = P.deriv()
        x[1:-1] = np.sort(dP.roots())

    # Compute the weights using the formula
    w = np.zeros(n)
    for i in range(n):
        # Evaluate the (n-1)th Legendre polynomial at x[i]
        P_val = Legendre.basis(n - 1)(x[i])
        w[i] = 2 / (n * (n - 1) * (P_val ** 2))

    return x, w


# Generate training points using different quadrature methods
def generate_training_points(method='uniform', num_points=10):
    if method == 'uniform':
        x_train = np.linspace(0, 1, num_points)
        weights = np.ones_like(x_train) / num_points  # Equal weights
    elif method == 'gauss_legendre':
        nodes, weights = roots_legendre(num_points)
        x_train = (nodes + 1)/2
        weights = weights/2
    elif method == 'gauss_lobatto':
        nodes, weights = gauss_lobatto_nodes_weights(num_points)
        x_train = (nodes + 1) * (1 / 2)  # Scale to [0,1]
        weights = weights * (1 / 2)
    elif method == 'thirds':
        third_N = int(np.ceil(num_points / 3))
        first_x_train = np.linspace(0, 2 * EPSILON, third_N)
        third_x_train = np.linspace(1 - (2 * EPSILON), 1, third_N)
        middle_x_train = np.linspace(2 * EPSILON, 1 - (2 * EPSILON), num_points - (2 * third_N))
        x_train = np.concatenate((first_x_train, middle_x_train, third_x_train))
        weights = np.ones_like(x_train) / num_points  # Equal weights
    elif method == 'outside_thirds':
        third_N = int(np.ceil(num_points / 3))
        first_x_train = np.linspace(-EPSILON, 2 * EPSILON, third_N)
        third_x_train = np.linspace(1 - (2 * EPSILON), 1 + EPSILON, third_N)
        middle_x_train = np.linspace(2 * EPSILON, 1 - (2 * EPSILON), num_points - (2 * third_N))
        x_train = np.concatenate((first_x_train, middle_x_train, third_x_train))
        weights = np.ones_like(x_train) / num_points  # Equal weights
    elif method == 'sin':
        linear = np.linspace(0, 1, num_points)
        x_train = np.sin(np.pi*linear/2)
        weights = np.ones_like(x_train) / num_points  # Equal weights
    else:
        raise ValueError("Unsupported quadrature method")
    # np.random.shuffle(x_train)
    return torch.tensor(x_train.reshape(-1, 1), dtype=torch.float32), torch.tensor(weights.reshape(-1, 1),
                                                                                   dtype=torch.float32)


def train_PINN(x_train, weights, epsilon=EPSILON):
    # Training the PINN
    model = PINN()
    optimiser = optim.Adam(model.parameters(), lr=0.01)
    loss_list = []

    # Continue training on the full dataset
    for epoch in range(20000):
        loss = compute_loss(model, x_train[1:-1], weights[1:-1], epsilon)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if epoch % 500 == 0:
            print(f"Full Training Epoch {epoch}, Loss: {loss.item():.6f}")
        loss_list.append(loss.item())

    return model, loss_list


# Plot the results
def create_results(quadrature, weights, color='red', label='', optimiser_name='adam'):
    if optimiser_name == 'adam':
        model, loss_list = train_PINN(quadrature, weights)
    elif optimiser_name == 'gauss_newton':
        model, loss_list = train_PINN_gauss_newton(quadrature, weights)
    else:
        raise ValueError("optimiser_name must be 'adam' or 'gauss_newton'")

    if optimiser_name == 'adam':
        plt.figure()
        plt.plot([i for i in range(len(loss_list))], loss_list, label=label)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

    dtype = next(model.parameters()).dtype
    x_eval = x_test.to(dtype=dtype)
    y_pred = model(x_eval).detach().cpu().numpy()
    plt.plot(x_test.numpy(), y_pred, label=label, color=color, linestyle='--')


if __name__ == "__main__":
    # Plotting True Result
    x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
    u2 = lambda x: 1 - np.cosh((x - 0.5) / EPSILON) / np.cosh(1 / (2 * EPSILON))
    y_true = np.array([u2(x) for x in x_test])
    plt.plot(x_test.numpy(), y_true, label='True Solution', color='green')

    # Getting Collocation Points and weights
    uniform, uniform_weights = generate_training_points(num_points=1000)
    gauss_10, gauss_10_weights = generate_training_points(method='gauss_legendre', num_points=1000)
    #sin, sin_weights = generate_training_points(method='sin', num_points=1000)
    # gauss_11, gauss_11_weights = generate_training_points(method='gauss_legendre', num_points=11)
    # lobatto_10, lobatto_10_weights = generate_training_points(method='gauss_lobatto')
    # lobatto_11, lobatto_11_weights = generate_training_points(method='gauss_lobatto', num_points=11)
    thirds,thirds_weights = generate_training_points(method='thirds', num_points=1000)
    # outside,outside_weights = generate_training_points(method='outside_thirds', num_points=300)

    # Plotting Quadratures
    create_results(uniform, uniform_weights, 'red', 'GN PINN: Uniform', optimiser_name='gauss_newton')
    create_results(gauss_10, gauss_10_weights, 'blue', 'GN PINN: Gauss', optimiser_name='gauss_newton')
    #create_results(sin, sin_weights, 'black', 'PINN: Sin')
    # create_results(gauss_11, gauss_11_weights, 'orange', 'PINN: Gauss_11')
    create_results(thirds, thirds_weights, 'green', 'GN PINN: Thirds', optimiser_name='gauss_newton')
    # create_results(outside, outside_weights, 'black', 'PINN: Outside')
    # create_results(lobatto_10, lobatto_10_weights, 'black', 'PINN: Lobatto_10')
    # create_results(lobatto_11, lobatto_11_weights, 'pink', 'PINN: Lobatto_11')

    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    title = r"$-ε^2 u''(x) + u(x) = 1$, ε = {:.5f}".format(EPSILON)
    plt.title(title)
    plt.show()
