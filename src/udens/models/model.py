import numpy as np
import swyft.lightning as sl
import torch

from dataclasses import dataclass, field
from math import pi
from typing import Callable, Optional, Union

# Global variables controlling image size and sub-/LOS halo population
BASE_N_PIX = 400
BASE_RES = 0.0125

from udens.models.utils import get_meshgrid, num_to_tensor

Tensor = torch.Tensor
Array = np.ndarray

# class classA:
#     def __init__(self):
#         print('a')


# Argument checking
def _check_kwargs_helper(kwargs, required_keys, name):
    if kwargs.keys() != required_keys:
        raise ValueError(
            f"missing or extra {name}: got {kwargs.keys()}, "
            f"but {required_keys} is required"
        )
        
def _check_subN_kwargs(subN_kwargs):
    _check_kwargs_helper(subN_kwargs, {"log10_M", "x", "y"}, "subN_kwargs")
    
def _parse_subN_kwargs(subN_kwargs, device):
    # Reparameterize subhalo mass and batch all parameters
    log10_M: Tensor = num_to_tensor(subN_kwargs["log10_M"], device=device)[0]
    x_sub: Tensor = num_to_tensor(subN_kwargs["x"], device=device)[0]
    y_sub: Tensor = num_to_tensor(subN_kwargs["y"], device=device)[0]
    if log10_M.ndim != 1:
        raise ValueError("log10_M must be a list")
    if x_sub.ndim != 1:
        raise ValueError("x_sub must be a list")
    if y_sub.ndim != 1:
        raise ValueError("y_sub must be a list")

    # Fix c_200c and tau
    return dict(
        M=10 ** log10_M,
        x=x_sub,
        y=y_sub,
    )

class Blobs():
    """
    Model with (exponential decreasing) blobs/balls, mimicking subhalos with controllable mass (size of the balls) and position.
    """

    def __init__(
        self,
        n_pix: int = 40,
        device: str = "cuda:0",
    ):

        torch.set_default_tensor_type(torch.cuda.FloatTensor)  # type: ignore
        self.device = device
        self.nx = self.ny = n_pix
        self.res = BASE_RES * BASE_N_PIX / n_pix

        X, Y = get_meshgrid(self.res, self.nx, self.ny, self.device)
        self.coerce_XY(X, Y)
        torch.set_default_tensor_type(torch.FloatTensor)
        
    def coerce_XY(self, X: Tensor = None, Y: Tensor = None):
        self.X = X
        self.Y = Y

    def __call__(self, subN_kwargs) -> Tensor:
        _check_subN_kwargs(subN_kwargs)
        subN_kwargs = _parse_subN_kwargs(subN_kwargs, self.device)


        w = torch.tensor((1e11), device = self.device)
        blobs = torch.zeros((self.nx, self.ny), device = self.device)

        for M, x, y in torch.stack([subN_kwargs.get(k) for k in ('M', 'x', 'y')]).T:
            R= ((self.X-x)**2 + (self.Y-y)**2)**0.5
            blobs += torch.exp(-(R)**2/(M/w)**2/2)

        return blobs
    
def _make_dict_subN(z_sub):
    log10_M, x_sub, y_sub = z_sub.T
    return dict(log10_M=log10_M, x=x_sub, y=y_sub)


def _priorN_uniform(
    name: str,
    nsub_bound: Array,
    low_init: Union[list, Array],
    high_init: Union[list, Array],
    N: int,
    bounds: sl.RectangleBound,
) -> Callable[[int, dict[str, sl.RectangleBound]], sl.SampleStore]:
    """
    Returns uniform prior function for the specified parameter with the
    specified initial lower and upper bounds.
    """
    low_init = np.asarray(low_init)
    high_init = np.asarray(high_init)
    assert low_init.size == high_init.size
    assert low_init.ndim == 1 or low_init.ndim == 2
    assert high_init.ndim == 1 or high_init.ndim == 2
    
    n_coord = low_init.shape[0]
    n_sub_max = nsub_bound[1]

    # TO DO: make this compatible with bounds
    if bounds is not None:
        # For now, raise an error
        # low = bounds[name].low
        # high = bounds[name].high
        raise NotImplementedError()
    else:
#         # Make sure mass bounds are the same as for shmf
#         if not torch.allclose(
#             torch.as_tensor(low_init[:, 0]).float(), shmf.support.lower_bound.log10()
#         ) or not torch.allclose(
#             torch.as_tensor(high_init[:, 0]).float(), shmf.support.upper_bound.log10()
#         ):
#             raise ValueError("low_init/high_init do not match bounds used by shmf")
        low = low_init
        high = high_init
        
        
    nsub_draw = np.random.randint(low = nsub_bound[0], high = nsub_bound[1]+1, size=(N)) # Nsub ~ U(nsub_bound)
    nsub_draw_idxs0 = np.repeat(np.arange(len(nsub_draw)), nsub_draw) # simulation index
    nsub_draw_idxs1 = np.concatenate([np.arange(nsub) for nsub in nsub_draw]) # subhalo idx of each simumlations
    
    draw = np.random.uniform(low=low, high=high, size=(N, n_sub_max, n_coord)) 
    draw = torch.tensor(draw).float()
    
    if nsub_bound[0] != nsub_bound[1]:
        draw[nsub_draw_idxs0, nsub_draw_idxs1] = torch.zeros_like(draw[nsub_draw_idxs0, nsub_draw_idxs1]) # (M,x,y) = (0,0,0) for subhalos that are assigned not to be a subhalo.
    
    return sl.SampleStore(**{name: draw})

@dataclass  # auto-generates __init__
class ModelBlobs_uniform(sl.SwyftModel):
    """
    SwyftModel with only the subhalo's position free.
    """

    n_pix: int = 50
    sigma_n: float = 0.25
    # Initial prior range
    nsub_bound: Array = np.array([0, 5])
    grid_low: Array = np.array([10.0, -2.5, -2.5])
    grid_high: Array = np.array([11.0, 2.5,  2.5])
    # True params for mock data generation
    z_sub_true: Array = np.array(
        [[10.5, -2.1, -2.1], [10.2, 1.5, 0.5], [10.8, -0.5, -1.0]]
    )
    bounds: Optional[sl.RectangleBound] = None
    device: str = "cuda:0"
    m: None = field(init=False)

    def __post_init__(self):
        # Initialize model
        self.m = Blobs(n_pix=self.n_pix, device=self.device)
        self.sub_low_init  = self.grid_low  #np.tile(np.array(self.grid_low), (self.n_sub, 1))
        self.sub_high_init = self.grid_high #np.tile(np.array(self.grid_high), (self.n_sub, 1))

    def slow(self, pars):
        mu = self.m(_make_dict_subN(pars["z_sub"]))
        return sl.SampleStore(mu=mu)

    def fast(self, d):
        if type(d["mu"]) == Array:
            d["mu"] = torch.from_numpy(d["mu"])
        img = d["mu"] + torch.randn_like(d["mu"]) * self.sigma_n
        return sl.SampleStore(img=img)

    def gen_mock(self):
        torch.random.manual_seed(25)
        pars = {"z_sub": self.z_sub_true}
        mu = self.slow(pars)
        return dict(**self.fast(mu), **mu, **pars)

    def prior_subN_uniform(self, N, bounds: Optional[sl.RectangleBound] = None):
        return _priorN_uniform(
            "z_sub", self.nsub_bound, self.sub_low_init, self.sub_high_init, N, self.bounds
        )

    def prior(self, N, bounds: Optional[sl.RectangleBound] = None):
        return self.prior_subN_uniform(N)