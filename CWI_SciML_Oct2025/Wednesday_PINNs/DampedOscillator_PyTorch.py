import torch
import matplotlib.pyplot as plt
import numpy as np

#Model
class DampedOscillator_PyTorch(torch.nn.Module):
    def __init__(self):
        super(DampedOscillator_PyTorch, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1,10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
        )
        def forward(self, x):
            return self.net(x)


#Helper Function to implement autograd
def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                               create_graph=True)[0]

#Loss Function
def compute_loss(model,x, l1, l2):
    #model coeffs
    m = 1
    mu = 2
    k = 4

    #Residual Calculation
    x.requires_grad = True
    u = model(x)
    du = grad(u,x)
    d2u = grad(du,x)

    residual = m*d2u + mu*du + k*u
    interior_loss = torch.mean(residual**2)

    #Boundary Implementations
    x_bc = torch.tensor([0],requires_grad=True)
    u_bc = model(x_bc)
    du_bc = grad(u_bc,x_bc)

    dirichlet_loss = l1*(u_bc - 1)**2
    neumann_loss = l2*(du_bc - 0)**2

    boundary_loss = dirichlet_loss + neumann_loss

    return interior_loss + boundary_loss


def get_collocation_points(method='Uniform', num_points=1000):
    if method == 'Uniform':
        x_train = np.linspace(0,10,num_points)

    return torch.tensor(x_train,requires_grad=True)

