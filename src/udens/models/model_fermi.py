import swyft.lightning as sl
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import powerbox as pb
import swyft
import matplotlib.pyplot as plt
import numpy.random as rd
from scipy.integrate import trapz
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.constants import c as c0
from astropy.cosmology import FlatLambdaCDM
import scipy.stats as stat
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

# Global variables controlling image size and sub-/LOS halo population
BASE_N_PIX = 400
BASE_RES = 0.1

Tensor = torch.Tensor
Array = np.ndarray

def num_to_tensor(*args, device=None):
    return [torch.as_tensor(a, dtype=torch.get_default_dtype(), device=device)
            for a in args]

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

class DM_SH_image():
    def __init__(self, N_pix = 256, DEVICE = 'cpu'):
        if DEVICE == "cuda:0":
            torch.set_default_tensor_type(torch.cuda.FloatTensor)  # type: ignore
            self.device = DEVICE
            self.nx = self.ny = N_pix
            self.res = BASE_RES * BASE_N_PIX / N_pix
        else:
            torch.set_default_tensor_type(torch.FloatTensor)  # type: ignore
            self.device = DEVICE
            self.nx = self.ny = N_pix
            self.res = BASE_RES * BASE_N_PIX / N_pix 

        X, Y = self.get_meshgrid(self.res, self.nx, self.ny, self.device)
        self.coerce_XY(X, Y)
        torch.set_default_tensor_type(torch.FloatTensor)
        
    def coerce_XY(self, X: Tensor = None, Y: Tensor = None):
        self.X = X
        self.Y = Y

    def get_meshgrid(self, resolution, nx, ny, device=None):
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
    
    def inverse_sample_function(self, dist, pnts, x_min = 1e-2, x_max = 30, n = 1e6, is_log = False, **kwargs):
        if is_log:
            x = np.logspace(np.log10(x_min), np.log10(x_max), int(n))
        else:
            x = np.linspace(x_min, x_max, int(n))
        cumulative = np.cumsum(dist(x, **kwargs))
        cumulative -= cumulative.min()
        f = interp1d(cumulative/cumulative.max(), x)
        return f(np.random.random(pnts))
    
    def get_SH_population_masses_VLII(self, alpha = 0.901, M_SH_min = 1e3, seed = None):
        if seed is None:
            rd.seed()
        else:
            rd.seed(seed)
        max_SHmass_VLII = 12134100000.0
        N_gt_Mcut = 6328
        shmf = lambda M, N0 = 1, Mcut = 5e6, alpha = alpha: N0 * (M/Mcut)**(-alpha)
        M_samples = np.logspace(np.log10(5e6), np.log10(max_SHmass_VLII), 5000)
        A = trapz(shmf(M_samples, alpha = alpha+1), M_samples)
        renorm_SHMF = N_gt_Mcut/A
        
        M_samples = np.logspace(np.log10(M_SH_min), np.log10(max_SHmass_VLII), 5000)
        N_tot = trapz(shmf(M_samples, alpha = alpha+1, N0 = renorm_SHMF), M_samples)
        
        samples_M = self.inverse_sample_function(lambda M: shmf(M, N0 = renorm_SHMF)/N_tot, int(N_tot), x_min = M_SH_min, x_max = max_SHmass_VLII, is_log = True)
        return samples_M

    def radial_subhalo_distribution(self, r, a = 0.94, b = 10.0, R0 = 785.0):
        return (r/R0)**a * np.exp(-b * (r - R0) / R0)

    def GC_spherical_to_cartesian(self, r, phi, theta):
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.cos(theta)
        z = r * np.sin(theta)
        return x, y, z

    def xyzGC_to_lpsithetaGAL_GEO(self, x,y,z):
        #//--- Returns (l,psi,theta) [kpc,rad,rad] from Gal. cart. coordinates (x,y,z) [kpc].
        #//
        #// INPUTS:
        #//  x[0]          x     [kpc]
        #//  x[1]          y     [kpc]
        #//  x[2]          z     [kpc]
        #// OUTPUTS:
        #//  x[0]          l     [kpc]
        #//  x[1]          psi   [rad]
        #//  x[2]          theta [rad]
        gMW_RSOL = 8.5
        x0 = gMW_RSOL - x
        y0 = y
        z0 = z
    
        D = np.sqrt(x0 * x0 + y0 * y0 + z0 * z0)
        l = np.arctan2(y0, x0 )
        b = np.arcsin(z0 / D)
        return D, l / np.pi * 180.0, b / np.pi * 180.0

    def get_SH_population_coordinates(self, N):
        r_samples = np.logspace(-6, 3, 50000)
        norm = trapz(self.radial_subhalo_distribution(r_samples) * r_samples**2, r_samples)
        samples_r = self.inverse_sample_function(lambda r: self.radial_subhalo_distribution(r) * r**2 / norm, int(N), x_min = 0, x_max = 1000)
        phi = rd.uniform(0., 1, int(N)) * 2.0 * np.pi
        theta = np.arcsin(2.0 * rd.uniform(0., 1., int(N)) - 1.0)
        
        xGC, yGC, zGC = self.GC_spherical_to_cartesian(samples_r, phi, theta)
        D, l, b = self.xyzGC_to_lpsithetaGAL_GEO(xGC, yGC, zGC)
        return samples_r, D, l, b

    def concentration_mass_distance_relation(self, m200, xsub):
        c0 = 19.9
        a = np.array([-0.195, 0.089, 0.089])
        b = -0.54
        return c0 * (1.0 + np.power(np.multiply(np.array([np.log10(m200/1e8)]).T ,np.repeat(np.array([a]), len(m200), axis = 0)), np.array([1, 2, 3])).sum(axis = 1)) * (1. + b * np.log10(xsub))

    def get_Jfactor_SH_population(self, M, D, rGC, cosmo = FlatLambdaCDM(Om0 = 0.3, H0=70)):
        R_200_MW = 240.0
        rho_crit = cosmo.critical_density(0)
        norm = (1.0 / (D * u.kpc)**2).to('1/cm2')
        f = lambda c: np.log10(1 + c) - c / (1. + c)
        concentrations = self.concentration_mass_distance_relation(np.array([M]), np.array([rGC/R_200_MW]))
        rs_ = self.get_scale_radius(M, concentrations)
        val = (M * u.Msun * c0**2).to('GeV') * concentrations**3 / f(concentrations)**2 * 200 / 9.0 * (rho_crit * c0**2).to('GeV / cm3')
        return (norm * val).value, np.arctan(rs_.value / D) * 180. / np.pi / 3

    def get_scale_radius(self, M, c, cosmo = FlatLambdaCDM(Om0 = 0.3, H0=70)):
        rho_crit = cosmo.critical_density(0)
        r_vir = np.power((M * u.Msun * c0**2).to('GeV') / ((rho_crit * c0**2).to('GeV / cm3') * 200 * 4/3 * np.pi), 1/3.)
        return r_vir.to('kpc') / c
    
    def __call__(self, DM_SH_data) -> Tensor:
        _check_subN_kwargs(DM_SH_data)
        subN_kwargs = _parse_subN_kwargs(DM_SH_data, self.device)
        raw_image = torch.zeros((self.nx, self.ny), device = self.device)
        
        for M, l, b in torch.stack([subN_kwargs.get(k) for k in ('M', 'x', 'y')]).T:
            if np.log10(M) == 0.0:
                continue
            rGC, D, lon, lat = self.get_SH_population_coordinates(1)
            Jfactor, angular_size = self.get_Jfactor_SH_population(M, D, rGC)
            R = ((self.X-l)**2 + (self.Y-b)**2)**0.5
            SH_radial_profile = torch.exp(-(R)**2/(angular_size)**2/2)
            raw_image += (SH_radial_profile/torch.sum(SH_radial_profile) * Jfactor)
        return raw_image

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

@dataclass
class Fermi_DM_SH_model(sl.SwyftModel):
    #### Generate a gamma-ray sky image somewhere along the Galactic plane including DM subhalos ####
    
    n_pix: int = 256
    GLON_center: float = 0.0
    GLAT_center: float = 0.0
    ROI_size: float = 40.0
    N_E: int = 10
    E_min: float = 0.5 ###GeV
    E_max: float = 500 ###GeV
    mDM: float = 100 ###GeV
    sigma_v: float = 1.0 ###cm^3/s
    seed: int = 20
    
    N_SH_bound: Array = np.array([0, 5])
    nsub_expect: float = 2.5
        
    grid_low: Array = np.array([7., -20., -20.])
    grid_high: Array = np.array([7., 20., 20.])
        
    bounds: Optional[sl.RectangleBound] = None
    device: str = "cpu"
        
    # True params for mock data generation
    z_sub_true: Array = np.array(
        [[9, -2.1, 14.0], [np.log10(5e8), -17, 1], [np.log10(3e8), 0., 0.]]
    )
        
    def __post_init__(self):
        # Initialize model
        self.DM_SH = DM_SH_image(N_pix=self.n_pix, DEVICE=self.device)
        self.grid_low = np.array([7, self.GLON_center - self.ROI_size/2.0, self.GLAT_center - self.ROI_size/2.0])
        self.grid_high = np.array([10, self.GLON_center + self.ROI_size/2.0, self.GLAT_center + self.ROI_size/2.0])
        self.sub_low_init  = self.grid_low  
        self.sub_high_init = self.grid_high 
        self.E = np.logspace(np.log10(self.E_min), np.log10(self.E_max), self.N_E)
        l = np.linspace(self.GLON_center - self.ROI_size/2.0, self.GLON_center + self.ROI_size/2.0, self.n_pix)
        b = np.linspace(self.GLAT_center - self.ROI_size/2.0, self.GLAT_center + self.ROI_size/2.0, self.n_pix)
        self.L, self.B = np.meshgrid(l, b)
        
    def dNdE_pi0(self): 
        return (self.E/0.5)**(-1.0-0.5 * np.log(self.E/0.5)) + 1e-4 * ((self.E/1e1)**(-1.4))

    def LogParabola(self, E, alpha, beta, E0 = 0.5):
        return np.power(E/E0, -alpha-beta*np.log(E/E0))

    def PLEC(self, E, gamma, Ec):
        return np.exp(-E/Ec)*(E/1.0)**(gamma) 

    def dNdE_chichi_tautau(self, mDM):
        #### from arXiv:0908.2258
        E = self.E
        avg_Ngamma = 2.0
        Emax = mDM / 2.0
        #a = -0.192
        a = -0.04
        #b = np.array([3.9, -2.26, -15.59, 32.96, -19.45])
        b = np.array([5.55, 4.29, -10.94, 0., 0.])
        c = np.array([7.36, 0.,0.])
        f = lambda x: x**a * (np.power(x[np.newaxis, :] , np.arange(len(b))[:, np.newaxis]) * b[:, np.newaxis]).sum(axis = 0)
        g = lambda x: (np.power(x[np.newaxis, :], np.arange(1, len(c)+1)[:, np.newaxis]) * c[:, np.newaxis]).sum(axis = 0)
        output = f(E/Emax) * np.exp(-g(E/Emax)) * avg_Ngamma / Emax * E
        return np.heaviside(output, 0.) * output ### multiplied with E to account for counts prediction
    
    def psc_disc(self, gamma = 0.54, Ec = 3.6, gamma_std = 0.05, N_src = 500):
        L, B = self.L, self.B
        E = self.E
        rho = np.exp(-(L/20)**2 -(B/3)**2)
        rho /= rho.sum()
        p = rho.flatten()
        idx = np.random.choice(len(p), p=p, size = N_src)
        m = np.zeros(len(p))
        Lu = np.random.lognormal(1.0, 2.0, size = len(idx))
        PLEC_indices = np.random.normal(gamma, gamma_std, size = len(idx))
        m2 = np.repeat(np.expand_dims(m, axis = 0), len(E), axis = 0)
        m2[:, idx] += (Lu[:, None] * self.PLEC(E[np.newaxis, :], PLEC_indices[:, np.newaxis], Ec)).T
        NY, NX = rho.shape
        m = m2.reshape((len(E), NY, NX))
        return m
    
    def psc_iso(self, alpha = 2.1257174, beta = 0.13385472, alpha_std = 0.39, N_src = 1000):
        L, B = self.L, self.B
        E = self.E
        rho = np.ones_like(L)
        rho /= rho.sum()
        p = rho.flatten()
        idx = np.random.choice(len(p), p=p, size = N_src)
        m = np.zeros(len(p))
        Lu = np.random.lognormal(1.0, 2.0, size = len(idx))
        LP_indices = np.random.normal(alpha, alpha_std, size = len(idx)) + 1.0  ### spectral scaling on counts rather than the spectrum itself
        m2 = np.repeat(np.expand_dims(m, axis = 0), len(E), axis = 0)
        m2[:, idx] += (Lu[:, None] * self.LogParabola(E[np.newaxis, :], LP_indices[:, np.newaxis], beta)).T
        NY, NX = rho.shape
        m = m2.reshape((len(E), NY, NX))
        return m
    
    def pi0(self, large_scale_norm):
        L, B = self.L, self.B
        p = pb.PowerBox(N = self.n_pix, dim = 2, pk = lambda k: 0.1*k**-2.3, seed = self.seed)
        x = p.delta_x()
        x = np.exp(x*2)
        x *= np.exp(-(L/40)**2)*(np.exp(-(abs(B)/2.5)**2) + large_scale_norm * np.exp(-(abs(B)/25)**2) + np.exp(-(abs(B)/1)**2))
        x = np.repeat(np.expand_dims(x, axis = 0), len(self.E), axis = 0)
        return x
    
    def ICS(self):
        L, B = self.L, self.B
        p = pb.PowerBox(N = self.n_pix, dim = 2, pk = lambda k: 0.1*k**-3, seed = self.seed)
        x = p.delta_x()
        x = np.exp(x*2)
        x *= np.exp(-(L/40)**2)*(np.exp(-(abs(B)/12)**2) + np.exp(-(abs(B)/12)**2) + np.exp(-(abs(B)/2)**2))
        x = np.repeat(np.expand_dims(x, axis = 0), len(self.E), axis = 0)
        return x
    
    def ISO(self):
        x = np.ones_like(self.L)
        x = np.repeat(np.expand_dims(x, axis = 0), len(self.E), axis = 0)
        return x

    def Fermi_PSF(self):
        E = np.log10(self.E * 1e3) ##MeV
        thresh_E = np.log10(1e4)
        m = (np.log10(6) - np.log10(0.8)) / (2 - 3)
        b = np.log10(6) - 2 * m
        return 10**np.where(E > thresh_E, thresh_E * m + b, E * m + b)

    def inverse_sample_function(self, dist, pnts, x_min = 1e-2, x_max = 30, n = 1e6, is_log = False, **kwargs):
        if is_log:
            x = np.logspace(np.log10(x_min), np.log10(x_max), int(n))
        else:
            x = np.linspace(x_min, x_max, int(n))
        cumulative = np.cumsum(dist(x, **kwargs))
        cumulative -= cumulative.min()
        f = interp1d(cumulative/cumulative.max(), x)
        return f(np.random.random(pnts))
    
    def slow(self, pars):
        #### Create full DM image with spectrum
        DM_SH_raw = self.DM_SH(_make_dict_subN(pars["z_sub"]))
        DM_SH_full = torch.tile(DM_SH_raw / 1e18, dims=(self.N_E, 1, 1)) #### apply normalization to reference J-factor 1e18 GeV^2/cm^5
        DM_SH_full *= (self.sigma_v * self.dNdE_chichi_tautau(self.mDM)[:, np.newaxis, np.newaxis])
        
        #### Create astro background
        mu = self.psc_iso()*4
        mu += self.pi0(1.0) * self.dNdE_pi0()[:, None, None]
        mu += self.ICS() * np.power(self.E / 1.0, -1.4)[:, None, None]
        mu += self.ISO() * np.power(self.E / 1.0, -1.3)[:, None, None]
        mu += self.psc_disc()
        mu = torch.from_numpy(mu).float()
        
        #### Combine signal and background
        mu += DM_SH_full * 100
        
        #### Apply PSF of Fermi-LAT
        PSF_sizes = self.Fermi_PSF()
        NY, NX = self.n_pix, self.n_pix
        l_min, l_max = self.L.min(), self.L.max()
        pix_resolution = (l_max - l_min)/NX
        filter_sizes = np.ceil(PSF_sizes / pix_resolution)
        mu2 = np.array([gaussian_filter(x[0], x[1])*10 for x in zip(mu.numpy(), filter_sizes)])
        mu3 = np.heaviside(mu2, 0.) * mu2
        mu4 = torch.from_numpy(mu3).float()

        #### Currently a simple 2D image is expected, hence this explicit selection
        #### Remove when adding flexibility to the pipeline!
        #return sl.SampleStore(mu=mu4[0])
        return sl.SampleStore(mu=mu4)
    
    def fast(self, d):
        if type(d["mu"]) == Array:
            d["mu"] = torch.from_numpy(d["mu"])
        try:
            img = torch.poisson(d["mu"])
        except:
            img = torch.ones_zeros(d["mu"])
        return sl.SampleStore(img=img)
    
    def gen_mock(self):
        torch.random.manual_seed(25)
        pars = {"z_sub": self.z_sub_true}
        mu = self.slow(pars)
        return dict(**self.fast(mu), **mu, **pars)
    
    def prior_subN_uniform(self, N, bounds: Optional[sl.RectangleBound] = None):
        return _priorN_uniform(
            "z_sub", self.N_SH_bound, self.sub_low_init, self.sub_high_init, N, self.bounds
        )

    def prior(self, N, bounds: Optional[sl.RectangleBound] = None):
        return self.prior_subN_uniform(N)
    
    def apply_afterburner(self, d):
        return d
