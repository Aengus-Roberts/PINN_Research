import torch

def PINN_Loss(model, x, x_b, beta=1.0, eps=1e-1):
    x = x.clone().detach().requires_grad_(True)

    u = model(x).view(-1,1)
    du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = du[:, 0:1]
    u_y = du[:, 1:2]
    du_x = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    du_y = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_xx = du_x[:, 0:1]
    u_yy = du_y[:, 1:2]

    laplacian_u = u_xx + u_yy

    residual = -eps**2 * laplacian_u + u - 1
    interior_loss = torch.mean(residual**2)

    boundary_residual = model(x_b).view(-1,1)
    boundary_loss = torch.mean(boundary_residual**2)

    return interior_loss + beta*boundary_loss, interior_loss, boundary_loss

def Ritz_Loss(model, x, x_b, beta=1.0, eps=1e-1):
    x = x.clone().detach().requires_grad_(True)

    u = model(x).view(-1, 1)
    du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    integrand = eps**2 * 0.5 * torch.sum(du ** 2, dim=1, keepdim=True) + 0.5 * u**2 - u
    interior_loss = torch.mean(integrand)

    boundary_loss = torch.mean(model(x_b).view(-1, 1) ** 2)

    return interior_loss + beta *boundary_loss, interior_loss, boundary_loss