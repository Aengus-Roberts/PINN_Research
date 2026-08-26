import torch

def sample_interior(N, device="cpu"):
    if N < 1:
        raise ValueError("N must be at least 1")

    X = 2 * torch.rand(N, 2, device=device) - 1
    return X

def sample_boundary(N, device="cpu"):
    if N < 1 or N % 9 != 0:
        raise ValueError("Boundary sample count must be a positive multiple of 9")

    n = N // 9

    left_s = 2 * torch.rand(2*n, 1, device=device) - 1
    right_s = 2 * torch.rand(2*n, 1, device=device) - 1
    bottom_s = 2 * torch.rand(2*n, 1, device=device) - 1
    top_s = 2 * torch.rand(2*n, 1, device=device) - 1
    slit_s = torch.rand(n,1,device=device)

    left   = torch.cat([-torch.ones_like(left_s), left_s], dim=1)
    right  = torch.cat([ torch.ones_like(right_s), right_s], dim=1)
    bottom = torch.cat([bottom_s, -torch.ones_like(bottom_s)], dim=1)
    top    = torch.cat([top_s,  torch.ones_like(top_s)], dim=1)
    slit = torch.cat([slit_s, torch.zeros_like(slit_s)], dim=1)

    return torch.cat([left, right, bottom, top, slit], dim=0)
