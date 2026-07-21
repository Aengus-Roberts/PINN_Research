import torch

def sample_interior(N, device="cpu"):
    """Sample points in [-1,1]^2 \ ((0,1] x [-1,0))."""
    points = []
    remaining = N

    while remaining > 0:
        # Oversample, then reject the removed lower-right quadrant.
        candidate = 2 * torch.rand(2 * remaining, 2, device=device) - 1
        keep = ~((candidate[:, 0] > 0.0) & (candidate[:, 1] < 0.0))
        candidate = candidate[keep]
        candidate = candidate[:remaining]
        points.append(candidate)
        remaining -= candidate.shape[0]

    return torch.cat(points, dim=0)

def sample_boundary(N, device="cpu"):
    """Sample points on the boundary of the corner domain."""
    n = N // 8

    s_left = 2 * torch.rand(2*n, 1, device=device) - 1
    s_top = 2 * torch.rand(2*n, 1, device=device) - 1
    s_right = torch.rand(n, 1, device=device)
    s_bottom = torch.rand(n, 1, device=device) - 1
    s_corner_x = torch.rand(n,1,device=device)
    s_corner_y = torch.rand(n,1,device=device) - 1

    left = torch.cat([-torch.ones_like(s_left), s_left], dim=1)       #x=-1, y in [-1,1]
    top = torch.cat([s_top, torch.ones_like(s_top)], dim=1)         #y=1, x in [-1,1]
    right = torch.cat([torch.ones_like(s_right), s_right], dim=1)    #x=1, y in [0,1]
    bottom = torch.cat([s_bottom, -torch.ones_like(s_bottom)], dim=1) #y=-1, x in [-1,0]
    corner_x = torch.cat([s_corner_x, torch.zeros_like(s_corner_x)], dim=1) #y=0, x in [0,1]
    corner_y = torch.cat([torch.zeros_like(s_corner_y), s_corner_y], dim=1) #x=0, y in [-1,0]

    return torch.cat([left, top, right, bottom, corner_x, corner_y], dim=0)