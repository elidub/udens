import torch, numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from .plot import plt_imshow#, plt_imshow_single
from .interpret import PostData, ObsData
from .bounds import GetBounds
import scipy.ndimage as spi


imkwargs = dict(extent=(-2.5, 2.5, -2.5, 2.5), origin='lower') #left, right, bottom, top

DEVICE = 'cuda'


# figsize = (4,3)
    
class LogPost(PostData):
    def __init__(self, tbl, posts, targets, n_alpha = 50, fig_kwargs = dict(dpi = 250, figsize = (4,3), tight_layout = True), title = 'uncalibrated'):
        super().__init__(posts, targets, n_alpha)
        self.tbl = tbl
        self.fig_kwargs = fig_kwargs
        self.title = title

    
    def plot_relicurve(self, title = "relicurve"):
        relicurve = self.get_relicurve(self.n_alpha).cpu()
        title = title + ' ' + self.title
        
        fig = plt.figure(**self.fig_kwargs)
        plt.stairs(relicurve, self.alpha_edges.cpu())
        plt.plot((0, 1), (0, 1), 'k:')
        plt.ylim(-0.02, 1.02)
        plt.xlim(-0.02, 1.02)
        plt.xlabel('Predicted pixel posterior')
        plt.ylabel('Fraction containing subhalo')
        self.tbl.experiment.add_figure(title, fig) if self.tbl is not None else plt.show()
        
    def plot_hist_posts(self, title = "hist_posts"):
        hist = self.get_histogram().cpu()
        title = title + ' ' + self.title
        
        fig = plt.figure(**self.fig_kwargs)
        plt.stairs(hist, self.alpha_edges.cpu(), fill=True)
        plt.yscale('log')
        plt.xlabel('Predicted pixel posteriors.')
        plt.ylabel('Counts')
        self.tbl.experiment.add_figure(title, fig) if self.tbl is not None else plt.show()
        
    def plot_hist_posts_freebinning(self, title = "hist_posts_freebinning"):
        title = title + ' ' + self.title
        
        bins = torch.logspace(torch.log10(self.posts.min()), torch.log10(self.posts.max()), 100)
        values, edges = torch.histogram(
            self.posts.cpu(), 
            bins = torch.logspace(torch.log10(self.posts.min()), torch.log10(self.posts.max()), 100)
        )
        
        fig = plt.figure(**self.fig_kwargs)
        plt.stairs(values, edges)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Predicted pixel posteriors.')
        plt.ylabel('Counts')
        self.tbl.experiment.add_figure(title, fig) if self.tbl is not None else plt.show()
            
    def plot_hist_sum_posts(self, title = "hist_sum_posts"):
        sum_posts = self.get_sum_posts().cpu().numpy()
        bins = 50 # np.arange(int(sum_posts.max()) + 1)
        title = title + ' ' + self.title
        
        if (sum_posts == 1.).all():
            print('everything 1!')

        fig = plt.figure(**self.fig_kwargs)
        hist, _, _ = plt.hist(sum_posts, bins = bins)
#         hist, _, _ = plt.hist(sum_posts, bins = [0, 1, 2, 1e8])
#         plt.plot((n_msc, n_msc), (0, hist.max()), 'k:')
        plt.xlabel('Sum of predicted pixel posteriors')
        plt.ylabel('Counts')
        self.tbl.experiment.add_figure(title, fig) if self.tbl is not None else plt.show()
    
    def plot_all(self):
        self.plot_relicurve()
        self.plot_hist_posts()
        self.plot_hist_sum_posts()
        self.plot_hist_posts_freebinning()

        
class LogSingleSub():
    def __init__(self, tbl, obs, obs_post, prior_grid, grid_coords, fig_kwargs = dict(dpi = 250)):
        
        self.tbl = tbl
        self.fig_kwargs = fig_kwargs
        
        
        assert obs_post.shape[:2] == (1, 1)
        self.obs_post  = obs_post.squeeze().cpu()
        self.obs_prior = prior_grid[1].squeeze(0).cpu()
        self.obs = obs
        self.xy_centers = grid_coords[2]


        
    def plot_x(self, title = 'x_sub', xlims = None):
        obs_post_x = torch.sum(self.obs_post, dim = 0)
        obs_prior_x = torch.sum(self.obs_prior, dim = 0)
        x_sub = self.obs['z_sub'][:,1]
        
        fig = plt.figure(**self.fig_kwargs)
        plt.step(self.xy_centers, obs_prior_x, where = 'mid', color = 'tab:blue')
        plt.step(self.xy_centers, obs_post_x, where = 'mid', color = 'tab:orange')
        plt.vlines(x_sub, 0, 1, color = 'tab:red')
        if xlims is not None: plt.xlim(xlims[0], xlims[1])
        self.tbl.experiment.add_figure(title, fig) if self.tbl is not None else plt.show()
        
    def plot_y(self, title = 'y_sub', ylims = None):
        obs_post_y = torch.sum(self.obs_post, dim = 1)
        obs_prior_y = torch.sum(self.obs_prior, dim = 1)
        y_sub = self.obs['z_sub'][:,2]
        
        fig = plt.figure(**self.fig_kwargs)
        plt.step(self.xy_centers, obs_prior_y, where = 'mid', color = 'tab:blue')
        plt.step(self.xy_centers, obs_post_y, where = 'mid', color = 'tab:orange')
        plt.vlines(y_sub, 0, 1, color = 'tab:red')
        if ylims is not None: plt.xlim(ylims[0], ylims[1])
        self.tbl.experiment.add_figure(title, fig) if self.tbl is not None else plt.show()
        
    def plot_all(self, xlims = None, ylims = None):
        self.plot_x(xlims = xlims)
        self.plot_y(ylims = ylims)
        
class LogObs(ObsData):
    def __init__(self, tbl, obs, obs_post, prior, grid_coords, fig_kwargs = dict(dpi=250)):
        super().__init__(obs, grid_coords)
        self.tbl = tbl
        self.obs_img = obs['img'].unsqueeze(0).cpu().numpy() 
        self.obs_post  = obs_post.cpu().numpy() #.squeeze(0)
        self.prior = prior.cpu().numpy()
        self.fig_kwargs = fig_kwargs
    
    def plot_msc(self, plot_true = True, tbl_title = 'msc with true', zlog = False, vminmax = False, title = None, **kwargs):
        x_sub, y_sub = self.z_sub.T[[1,2]]
        m_idx = self.z_sub_idx.T[0]
        msc_scatter = np.stack((m_idx, x_sub, y_sub)).T 
        msc_scatter = msc_scatter[np.sum(np.abs(msc_scatter), axis = 1) != 0] # Drop the values that have [0,0,0] coordinates
        msc_scatter = msc_scatter if plot_true is True else None
        
#         titles = [fr'${m_center:.2f} \, M_{{\odot}}$' for m_center in self.m_centers]
        titles = [fr'${m_left:.2f}-{m_right:.2f} \, M_{{\odot}}$' for m_left, m_right in zip(self.m_edges[:-1], self.m_edges[1:])]
        width = len(titles)*3

        kwargs = dict(**kwargs, **imkwargs)
        
        
        fig = plt_imshow(self.obs_post, 
                   vminmax = vminmax,
                   tbl = self.tbl, tbl_title = tbl_title,
                   zlog = zlog,
                   msc_scatter = msc_scatter, 
                   title = title,
                   titles = titles,
                   title_size = 12,
                   tl = True, priors = self.prior, cbar = True, 
                   fig_kwargs = dict(dpi = 250, figsize = (width, 3)), 
                   **kwargs)
        return fig
        
    def plot_obs(self):
        plt_imshow(self.obs_img, 
                   tbl = self.tbl, tbl_title = 'obs',
                   circle_scatter = self.z_sub, 
                   cbar = True, fig_kwargs = dict(dpi = 250, figsize = (3, 3)), **imkwargs)
    
    def plot_all(self):
        self.plot_msc()
        self.plot_msc(plot_true = False, tbl_title = 'msc')
        self.plot_obs()

        
class LogIRC:
    def __init__(self, tbl, irc, fig_kwargs = dict(dpi = 250), title = 'calibration'):
        
        self.tbl = tbl
        self.fig_kwargs = fig_kwargs
        self.title = title
                
        self.posts_uncalib = irc.posts_uncalib
        self.targets = irc.targets
        self.relicurve_uncalib = irc.relicurve_uncalib
        self.alpha_edges = irc.post_data_uncalib.alpha_edges.cpu()
        self.alpha_centers = irc.post_data_uncalib.alpha_centers.cpu()
        self.ir = irc.ir

        
        self.posts_calib = irc.calibrate(self.posts_uncalib)
        self.relicurve_calib = PostData(self.posts_calib, self.targets).get_relicurve().cpu()
        
    def plot(self):
        fig = plt.figure(**self.fig_kwargs)
        plt.stairs(self.relicurve_uncalib, self.alpha_edges, label = 'Uncalibrated')
        plt.stairs(self.relicurve_calib, self.alpha_edges, label = 'Calibrated')
        plt.plot(self.alpha_centers, self.ir.predict(self.alpha_centers), label = 'Fit')
        plt.legend(loc = 2)
        plt.plot((0, 1), (0, 1), 'k:')
        plt.xlabel('Predicted pixel posterior')
        plt.ylabel('Fraction containing subhalo')
        self.tbl.experiment.add_figure(self.title, fig) if self.tbl is not None else plt.show()

class LogBounds(GetBounds):
    def __init__(self, tbl, obs, obs_post, grid_coords, grid_low, grid_high, fig_kwargs = dict(dpi=250)):
        super().__init__(obs, obs_post, grid_coords)
        self.tbl = tbl
        self.fig_kwargs = fig_kwargs
        

    def plot_features_vs_threshold(self):
        fig = plt.figure(**self.fig_kwargs)
        plt.plot(self.thresholds, self.n_features_th, label = r'$N_{features}$')
        plt.plot((0., 1.), (self.obs_n_sub, self.obs_n_sub), 'k:', label = r'$N_{sub,obs}$')
        plt.legend()
        plt.xlabel('Threshold')
        plt.ylabel('N')
        plt.title(f'lowest threshold = {self.lowest_threshold:.3f}')
        self.tbl.experiment.add_figure("features_vs_threshold", fig) if self.tbl is not None else plt.show()

        
    def plot_img_labeled(self):
        fig = plt_imshow(self.obs_post_labeled, 
                         tbl = self.tbl, tbl_title = 'labels',
                         cbar = True, vmin = 0, vmax = self.obs_n_sub, 
                         fig_kwargs = self.fig_kwargs, **imkwargs)
    
    def plot_bounds(self):
        pixel_bounds = [ np.mgrid[l[0]:h[0]+1, l[1]:h[1]+1, l[2]:h[2]+1].squeeze().T.reshape(-1, 3) for l, h in zip(self.bounds_low_idxs, self.bounds_high_idxs)]
        
        post_pixel_bounds = np.zeros(self.obs_post.shape)

        for i in range(self.obs_n_sub):
            post_pixel_bounds[tuple(pixel_bounds[i].T)] = i+1
            
        fig = plt_imshow(post_pixel_bounds, 
                         tbl = self.tbl, tbl_title = "pixel_bounds",
                         cbar = True,
                         fig_kwargs = self.fig_kwargs, **imkwargs)
            

    def plot_all(self):
        self.plot_features_vs_threshold()
        self.plot_img_labeled()
        self.plot_bounds()
        
        