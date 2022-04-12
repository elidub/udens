import torch

def get_meshgrid(resolution, nx, ny, device=None):
    """Constructs lens plane meshgrids with padding for kernel convolution.
    Returns
    -------
    X, Y : torch.Tensor, torch.Tensor
    """
    dx = resolution
    dy = resolution

    # Coordinates at pixel centers
    x = torch.linspace(-1, 1, int(nx), device=device) * (nx - 1) * dx / 2
    y = torch.linspace(-1, 1, int(ny), device=device) * (ny - 1) * dy / 2

    # Note difference to numpy (!)
    Y, X = torch.meshgrid((y, x))

    return X, Y

def num_to_tensor(*args, device=None):
    return [torch.as_tensor(a, dtype=torch.get_default_dtype(), device=device)
            for a in args]