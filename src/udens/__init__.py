from torch import nn
import torch
import swyft
import swyft.lightning as sl
from swyft.networks import OnlineDictStandardizingLayer

from .classifier import RatioEstimatorUNet
from .inference import Prior

DEVICE = 'cuda'

class ImgSegmNetwork(sl.SwyftModule):
    """
    Network to infer probability that a (M,x,y) pixel contains a subhalo
        - n_pix: number of pixels of the image
        - n_msc: number of mass channels, i.e. the number of masses one has to differentiate between.
        - grid_low (grid_high): The mininimum (maximum) (M,x,y) values where one performs image segmentation
    """

    def __init__(self, cfg, n_msc):
        super().__init__(cfg)
        
        self.n_msc = n_msc
        n_pix = cfg.simulation.model.n_pix
        grid_low = torch.tensor(cfg.simulation.model.grid_low, device = DEVICE)
        grid_high = torch.tensor(cfg.simulation.model.grid_high, device = DEVICE)
        
        grid_dim = torch.tensor([n_msc, n_pix, n_pix], device = DEVICE)
        
        n_sub_expect = cfg.n_sub_expect
        prior_grid = Prior(n_sub_expect, n_pix, n_msc, grid_low, grid_high, None).prior_grid()
        
       
        self.online_z_score = swyft.networks.OnlineDictStandardizingLayer(
            dict(img = (n_pix, n_pix))
        )
        
        self.classifier = RatioEstimatorUNet(grid_dim, grid_low, grid_high, prior_grid)

        
    def forward(self, x, z):
        x = dict(img = x['img'])
        x = self.online_z_score(x)['img']
        out = self.classifier(x, z['z_sub'])
        return dict(z_pix = out)


# class ImgSegmNetwork(sl.SwyftModule):
#     """
#     Network to infer probability that a (M,x,y) pixel contains a subhalo
#         - n_pix: number of pixels of the image
#         - n_msc: number of mass channels, i.e. the number of masses one has to differentiate between.
#         - grid_low (grid_high): The mininimum (maximum) (M,x,y) values where one performs image segmentation
#     """

#     def __init__(self, cfg, n_msc):
#         super().__init__(cfg)
        
#         self.n_msc = n_msc
#         n_pix = cfg.simulation.model.n_pix
#         grid_low = torch.tensor(cfg.simulation.model.grid_low, device = DEVICE)
#         grid_high = torch.tensor(cfg.simulation.model.grid_high, device = DEVICE)
        
#         grid_dim = torch.tensor([n_msc, n_pix, n_pix], device = DEVICE)
        
       
#         self.online_z_score = swyft.networks.OnlineDictStandardizingLayer(
#             dict(img = (n_pix, n_pix))
#         )
        
#         self.classifier = RatioEstimatorUNet(grid_dim, grid_low, grid_high)

        
#     def forward(self, x, z):
#         x = dict(img = x['img'])
#         x = self.online_z_score(x)['img']
#         out = self.classifier(x, z['z_sub'])
#         return dict(z_pix = out)


