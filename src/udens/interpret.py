import torch, numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression


from tqdm import tqdm

from .plot import plt_imshow#, plt_imshow_single
import scipy.ndimage as spi

imkwargs = dict(extent=(-2.5, 2.5, -2.5, 2.5), origin='lower') #left, right, bottom, top

DEVICE = 'cuda'


class Alpha():
    def __init__(self, n_alpha = 50):
        self.n_alpha = n_alpha
        self.alpha_edges = torch.linspace(0, 1, n_alpha)#, device = DEVICE)#, dtype=torch.float64)
        self.alpha_edges = self.alpha_edges.to(DEVICE)
        self.alpha_centers = (self.alpha_edges[:-1] + self.alpha_edges[1:])/2
        
class Alpha2():
    def __init__(self, posts, prior, n_alpha_log = 50, n_alpha_normal = 50):
        self.n_alpha_log = n_alpha_log
        self.n_alpha_normal = n_alpha_normal
        self.n_alpha = n_alpha_log + n_alpha_normal
        
        self.alpha_log    = torch.logspace(torch.log10(posts.min()), torch.log10(prior.min()), n_alpha_log)
        self.alpha_normal = torch.linspace(prior.min(), posts.max(), n_alpha_normal)
        self.alpha_edges = torch.cat((self.alpha_log, self.alpha_normal))
        self.alpha_edges = self.alpha_edges.to(DEVICE)
        self.alpha_centers = (self.alpha_edges[:-1] + self.alpha_edges[1:])/2
    

class PostData(Alpha):
    def __init__(self, posts, targets, n_alpha = 50):
        super().__init__(n_alpha = n_alpha)
        self.posts = posts.to(DEVICE)
        self.targets = targets.to(DEVICE)
# class PostData(Alpha2):
#     def __init__(self, posts, targets, prior, n_alpha = 50):
#         super().__init__(posts, prior)
#         self.posts = posts.to(DEVICE)
#         self.targets = targets.to(DEVICE)
    
    def get_histogram(self):
        hist = torch.histogram(self.posts.flatten().cpu(), bins = self.alpha_edges.cpu())[0]
        hist = hist.to(DEVICE)
        return hist
    
    def get_relicurve(self, batch_size = 16):
        
        is_between_sum = torch.zeros_like(self.alpha_centers)
        
        for batch_idx in tqdm(range(int(np.ceil(len(self.posts) / batch_size))), desc='Calculating reliability curve'):
            i, j = batch_idx*batch_size, (batch_idx+1)*batch_size
            posts_alpha = torch.repeat_interleave(self.posts[i:j].unsqueeze(-1), self.n_alpha-1, dim = -1)
            targets_alpha = torch.repeat_interleave(self.targets[i:j].unsqueeze(-1), self.n_alpha-1, dim = -1)
        
            is_between = (posts_alpha > self.alpha_edges[:-1]) & (posts_alpha < self.alpha_edges[1:])
            is_between_sum += torch.sum(targets_alpha * is_between, dim = (0, 1, 2, 3))
        hist = self.get_histogram() 
        relicurve = is_between_sum/hist
        relicurve = torch.nan_to_num(relicurve)
        return relicurve

    
    def get_sum_posts(self):
        assert len(self.posts.shape) == 4
#         assert list(self.posts.shape)[1:] == [self.n_msc, self.n_pix, self.n_pix]
        return torch.sum(self.posts, axis = (1, 2, 3)).cpu()

class ObsData():
    def __init__(self, obs, grid_coords):
        m_centers, m_edges, xy_centers, xy_edges = grid_coords
        self.m_centers  = m_centers.cpu().numpy()
        self.m_edges    = m_edges.cpu().numpy()
        self.xy_centers = xy_centers.cpu().numpy()
        self.xy_edges   = xy_edges.cpu().numpy()
        
        
#         m_centers, _, xy_centers, _ = grid_coords
#         self.z_sub = self.get_z_sub(obs)
        self.z_sub = obs['z_sub'].cpu().numpy()
        self.z_sub_idx = self.get_z_sub_idx(self.z_sub, self.m_centers, self.xy_centers)
        
#         condition_x = {'img' : mock['img'].unsqueeze(0).cpu()} 
#         self.post   = infer.predict(net, condition_x ).squeeze(0).cpu().numpy()
#         self.priors = infer.calc_prior().cpu().numpy()
        
#     def get_z_sub(self, obs):
#         z_sub = np.array([v.numpy() for k, v in obs['z_sub'].items()]).T
#         return z_sub
        
    def get_z_sub_idx(self, z_sub, m_centers, xy_centers):
#         m_centers = infer.m_centers.numpy()
#         xy_centers = infer.xy_centers.numpy()
        z_sub_idx = np.array([np.abs( (z_sub_coord.reshape(1,-1) - centers.reshape(-1,1)) ).argmin(axis=0) 
                              for centers, z_sub_coord in zip([m_centers, xy_centers, xy_centers], z_sub.T)]).T
        return z_sub_idx
    
class IsotonicRegressionCalibration():
    def __init__(self, posts_uncalib, targets):
#         super().__init__()
        self.posts_uncalib = posts_uncalib
        self.targets = targets
        
        
#         assert torch.sum(torch.isnan(posts_uncalib.view(-1))).item() == 0
        
        
        self.post_data_uncalib = PostData(posts_uncalib, targets)
    
        self.relicurve_uncalib = self.post_data_uncalib.get_relicurve().cpu()
        
        self.ir = self.get_ir(self.relicurve_uncalib, self.post_data_uncalib.alpha_centers.cpu())
    
    
    def get_ir(self, relicurve, alpha_centers):
        alpha_centers_zero = torch.cat((torch.tensor([0]), alpha_centers))
        relicurve_zero     = torch.cat((torch.tensor([0]), relicurve))

        ir = IsotonicRegression(out_of_bounds = 'clip')
        
        ir.fit(alpha_centers_zero, relicurve_zero);
        return ir
    
    def calibrate(self, posts):
        posts_calib = self.ir.predict(posts.cpu().flatten()).reshape(posts.shape)
        posts_calib = torch.tensor(posts_calib, device = posts.device, dtype = posts.dtype)
        return posts_calib