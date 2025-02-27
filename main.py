import numpy as np

gauss_11 = np.polynomial.legendre.leggauss(11)


def func(x):
    return (x + 1) / 2


print(gauss_11)
print(np.vectorize(func)(gauss_11))

x.requires_grad_(True)
u = model(x)
u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

# ODE residual: u"(x) + u(x) - 1
residual = -EPSILON ** 2 * u_xx + u - 1

u2 = lambda x: 1 - np.cosh((x - 1 / 2) / EPSILON) / np.cosh(1 / (2 * EPSILON))
