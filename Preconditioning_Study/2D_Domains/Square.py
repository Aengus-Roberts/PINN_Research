import torch

def sample_interior(N, device="cpu"):
    X = 2 * torch.rand(N, 2, device=device) - 1
    return X

def sample_square_boundary(N, device="cpu"):
    n = N // 4

    s = 2 * torch.rand(n, 1, device=device) - 1

    left   = torch.cat([-torch.ones_like(s), s], dim=1)
    right  = torch.cat([ torch.ones_like(s), s], dim=1)
    bottom = torch.cat([s, -torch.ones_like(s)], dim=1)
    top    = torch.cat([s,  torch.ones_like(s)], dim=1)

    return torch.cat([left, right, bottom, top], dim=0)