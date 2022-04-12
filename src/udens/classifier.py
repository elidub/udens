import torch
import swyft
import swyft.lightning as sl

import numpy as np
from matplotlib import pyplot as plt


from swyft.lightning import equalize_tensors, RatioSamples
from .unet import UNET

from .plot import plt_imshow
imkwargs = dict(extent=(-2.5, 2.5, -2.5, 2.5), origin='lower') #left, right, bottom, top


DEVICE = 'cuda'


class ParameterTransformUNet(torch.nn.Module):
    def __init__(self, grid_dim, grid_low, grid_high) -> None:
        super().__init__()
        self.grid_dim = grid_dim
        self.grid_low = grid_low
        self.grid_high = grid_high
        
    def v_to_grid(self, coord_v):
        """
        Maps real coordinates to grid on [0, 1]
        """
        
        coord_grid = (coord_v - self.grid_low) / (self.grid_high - self.grid_low)
        
        coord_grid = torch.nan_to_num(coord_grid) # set nan to zero if m_low = m_high
        
        return coord_grid

    def forward(self, coord_v):       
        
        coord_grid = self.v_to_grid(coord_v)
        n_batch, n_sub_var, _ =  coord_grid.shape # n_sub_var: the actual number of subhalos within these simulations

        z = torch.zeros((n_batch, *self.grid_dim), device = DEVICE) # intialize the 'binary target map'
  
        if not (n_batch == 0 or n_sub_var == 0):
            
            coord_grid_idxs = torch.floor(coord_grid.view(-1,3) * self.grid_dim).T # translate [0, 1] coords to pixel indices
            coord_grid_idxs[[1,2]] = coord_grid_idxs[[2,1]] # swap x, y coordinates for the indices for some strange plt.imshow reason
                              
            b_idx = torch.repeat_interleave(torch.arange(n_batch, device = DEVICE), n_sub_var) # batch index of each coordinate
            indices = torch.stack((b_idx, *coord_grid_idxs))

            # Deselect subhalos with M = 0, the 'no halo' halos
            indices_empty = indices[1] >= 0     # select non zero M indices 
            indices = indices[:,indices_empty] # only take those into account
            
            assert (torch.amax(coord_grid_idxs, dim = 1) <= self.grid_dim).all() # assert that coord indices are on gid indices
            z[tuple(indices.type(torch.long))] = 1 # all pixel containing subhalo get value one
        return z

class MarginalClassifierUNet(torch.nn.Module):
    def __init__(self, grid_dim, prior_grid):
        super().__init__()

        self.grid_dim = grid_dim
        self.prior_grid = prior_grid
        in_channels = 1 # observation has one color, so one input channel
        out_channels = self.grid_dim[0] * 2 # number of mass channels, times two because channel for pixel containing subhalo or no subhalo
        self.UNet = UNET(in_channels = in_channels, out_channels = out_channels)
                
    def forward(self, sims: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = self.UNet(sims)
        
        x = x.view(x.shape[0], 2, *self.grid_dim) # reshaping each batch so we have logratios [d_{ijk0}, d_{ijk1}], pixel containing subhalo or no subhalo
        target_map = torch.stack((1-z, z), dim = 1)

        if not x.is_cuda: x = x.to(DEVICE) # this should be fixed, why are there inputs not on cuda?
            
#         x1 = torch.sigmoid(x)
#         x2 = x1 / self.prior_grid
#         x3 = torch.log(x2)

        r = torch.exp(x)
        posts = r * self.prior_grid
        
        post_sum = torch.sum(posts, axis = 1) # Sum p(z=0|x) + p(z=1|x) for each pixel
        
        epsilon = torch.finfo(x.dtype).tiny
        post_sum[post_sum == 0.] = epsilon # because we don't want to devie by zero
        posts[posts == torch.inf] = 1.
        
        
        post_sum = torch.stack((post_sum, post_sum), dim = 1)
        post_norm = posts / post_sum # Normalize p(z=1|x)
        
        nansum = torch.isnan(post_norm).sum()
        if nansum > 0: print('post_norm', nansum)
        
        r2 = post_norm / self.prior_grid
        
        x = torch.log(r2)
        
#         x = x3

#         if np.random.randint(20) == 1:
#             x = torch.sigmoid(x)
#             x = x / self.prior_grid
#             x = torch.log(x)
        
#             print(x.max(), x.min())
            
#             plt.hist(x.flatten().cpu().detach().numpy())
#             plt.show()
#             assert 1 == 2
        
        x = x * target_map # the mapping L[C], shape: batch, 2, *grid_dim = batch, 2, n_msc, n_pix, n_pix        
        x = torch.sum(x, axis = 1) # sum so that we consider only the correct logratio
        
#         plt_imshow(x[0].cpu(), cbar = True, **imkwargs)
        
        return x

class RatioEstimatorUNet(torch.nn.Module):
    def __init__(self, grid_dim, grid_low, grid_high, prior_grid):
        super().__init__()
        self.paramtrans = ParameterTransformUNet(grid_dim, grid_low, grid_high)
        self.classifier = MarginalClassifierUNet(grid_dim, prior_grid)

        
    def forward(self, x, z):
        x, z = equalize_tensors(x, z)
        zt = self.paramtrans(z).detach()
        ratios = self.classifier(x, zt)
        w = RatioSamples(z, ratios, metadata = {"type": "ImgSegm"})
        return w
    
    
# class RatioEstimatorUNet(torch.nn.Module):
#     def __init__(self, grid_dim, grid_low, grid_high):
#         super().__init__()
#         self.paramtrans = ParameterTransformUNet(grid_dim, grid_low, grid_high)
#         self.classifier = MarginalClassifierUNet(grid_dim)

        
#     def forward(self, x, z):
#         x, z = equalize_tensors(x, z)
#         zt = self.paramtrans(z).detach()
#         ratios = self.classifier(x, zt)
#         w = RatioSamples(z, ratios, metadata = {"type": "ImgSegm"})
#         return w



    
    

