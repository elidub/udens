import torch
import numpy as np
from tqdm import tqdm
DEVICE = 'cuda'


class Prior():
    def __init__(self, n_sub, n_pix, n_msc, grid_low, grid_high, shmf):
        self.n_sub = n_sub
        self.n_pix = n_pix
        self.n_msc = n_msc
        self.grid_low = grid_low 
        self.grid_high = grid_high
        self.shmf = shmf
        
        
        self.m_centers, self.m_edges, self.xy_centers, self.xy_edges = self.grid_coords = self.get_grid_coords()
        self.prior, M_frac = self.calc_prior()
        
        print('Prior,    M_frac    in subhalo log10 mass range')
        for m_left, m_right, p, m  in zip(self.m_edges[:-1], self.m_edges[1:], self.prior, M_frac):
            print(f"{p:.2e}, {m:.2e}:    [{m_left:.3f} - {m_right:.3f}]")
    
    def get_grid_coords(self):
        """
        Returns pixel center and edge locations in mass and position (xy) direction.      
        """
        m_all = torch.linspace(self.grid_low[0], self.grid_high[0], 2*self.n_msc+1)
        m_edges, m_centers = m_all[0::2], m_all[1::2]
        
        assert (self.grid_low[1], self.grid_high[1]) ==  (self.grid_low[2], self.grid_high[2])
        xy_all = torch.linspace(self.grid_low[1], self.grid_high[1], 2*self.n_pix+1)
        xy_edges, xy_centers = xy_all[0::2], xy_all[1::2] 
        return m_centers, m_edges, xy_centers, xy_edges
    
    def calc_prior(self):
        """
        Prior for each mass channel.
        """
        if torch.all(self.m_edges == self.m_edges[0]) or self.shmf is None:
            M_frac = torch.full((self.n_msc,), 1/self.n_msc).to(DEVICE) #torch.tensor([1])
        else:
            M_frac = torch.diff(self.shmf.cdf(torch.pow(10, self.m_edges))).to(DEVICE) # calculating fraction of subhalo in each mass channel with SHMF
        prior = self.n_sub / (self.n_pix*self.n_pix) * M_frac
        
        return prior, M_frac
        
    
    def prior_grid(self, n_batch=1):
        """
        Prior map for each (M,x,y) pixel.
        """
        prior = self.prior.unsqueeze(1).unsqueeze(1).repeat(1, self.n_pix, self.n_pix)
#         return torch.cat((1-prior, prior), dim = 0)
        return torch.stack((1-prior, prior), dim = 0)
#         return prior

    

    
class Infer(Prior):
    
    def __init__(self, simulator, network, datamodule, n_sub_expect):
        super().__init__(
            n_sub = n_sub_expect, #simulator.n_sub,
            n_pix = simulator.n_pix,
            n_msc = network.n_msc,
            grid_low = simulator.grid_low,
            grid_high = simulator.grid_high,
            shmf = simulator.shmf if hasattr(simulator, 'shmf') else None
        )
        network.eval()
        self.network = network
        self.datamodule = datamodule

        self.z_sub_empty, self.z_sub_all = self.get_1_z_sub(self.grid_low, self.grid_high, self.m_edges, self.n_pix)
                
        
    def get_1_z_sub(self, grid_low, grid_high, m_edges, n_pix):
        """
        Puts a subhalo in each (no) pixel and creates the target binary map for existing of only ones (zeros) in each pixel.
        """
        grid = torch.linspace(grid_low[1], grid_high[1], n_pix+1)[:-1]
        x, y = torch.meshgrid(grid, grid, indexing = 'xy')
        ms = [torch.full((x.shape), m_i) for m_i in m_edges[:-1]]

        z_sub_empty = torch.tensor((), device = DEVICE, dtype = torch.float).reshape(1, 0, 3).to(DEVICE)
        z_sub_all   = torch.cat([ torch.stack((m, x, y), dim = 2).flatten(end_dim = 1) for m in ms]).unsqueeze(0).to(DEVICE)

        return dict(z_sub = z_sub_empty), dict(z_sub = z_sub_all)
    

    
    def unsqueeze_obs(self, s0):
        """
        Unsqueezes a single observation so it can pass trough the network which only accepts batches
        """
        
        s0_copy = s0.copy()
        for k, v in s0_copy.items():
            if type(v) == torch.Tensor:
                s0_copy[k] = v.unsqueeze(0).cpu()
            if type(v) == np.ndarray:
                 s0_copy[k] = np.expand_dims(v, 0)
        return s0_copy
    
    def squeeze_obs(self, s0):
        """
        Squeezes a single observation so it can pass trough the network which only accepts batches
        """
        
        s0_copy = s0.copy()
        for k, v in s0_copy.items():
            if type(v) == torch.Tensor:
                s0_copy[k] = v.squeeze(0).cpu()
            if type(v) == np.ndarray:
                 s0_copy[k] = np.squeeze(v, 0)
        return s0_copy
    
    def drop_nan_posts(self, posts_norm, posts_unnorm, targets, batch_size = 16):
        nan_idxs = np.unique(np.argwhere(np.isnan(posts_norm.cpu()))[0])

        if len(nan_idxs) > 0:
            print(f'Watch out! There are {len(nan_idxs)} predictions with NaN values!')

            bool_idxs = torch.zeros_like(posts_norm, dtype = bool)
            bool_idxs[nan_idxs] = True
            
#             posts_norm_without_nan = [] #torch.zeros_like(posts_norm, device = posts_norm.device)
#             posts_unnorm_without_nan = [] #torch.zeros_like(posts_unnorm, device = posts_unnorm.device)
#             targets_without_nan = [] #torch.zeros_like(targets, device = targets.device)
            
#             for batch_idx in tqdm(range(int(np.ceil(len(posts_norm) / batch_size))), desc='Removing NaN values'):
#                 i, j = batch_idx*batch_size, (batch_idx+1)*batch_size
                
# #                 print(i, j, posts_norm_without_nan[i:j].shape, posts_norm[i:j].shape, bool_idxs[i:j].shape, posts_norm[i:j][~bool_idxs[i:j]].view(-1, self.n_msc, self.n_pix, self.n_pix).shape)
                
#                 posts_norm_without_nan.append( posts_norm[i:j][~bool_idxs[i:j]].view(-1, self.n_msc, self.n_pix, self.n_pix) )
#                 posts_unnorm_without_nan.append( posts_unnorm[i:j][~bool_idxs[i:j]].view(-1, self.n_msc, self.n_pix, self.n_pix) )
#                 targets_without_nan.append( targets[i:j][~bool_idxs[i:j]].view(-1, self.n_msc, self.n_pix, self.n_pix))
            
#             posts = posts[~bool_idxs].view(-1, self.n_msc, self.n_pix, self.n_pix)
            posts_norm = posts_norm[~bool_idxs].view(-1, self.n_msc, self.n_pix, self.n_pix)
            posts_unnorm = posts_unnorm[~bool_idxs].view(-1, self.n_msc, self.n_pix, self.n_pix)
            targets = targets[~bool_idxs].view(-1, self.n_msc, self.n_pix, self.n_pix)

#             return torch.cat(posts_norm_without_nan), torch.cat(posts_unnorm_without_nan), torch.cat(targets_without_nan)
            return posts_norm, posts_unnorm, targets

        else:
            return posts_norm, posts_unnorm, targets
    
    def get_post(self, s0):
        """
        Puts a target where each (no) pixel contains a subhalo trough the trained network together with observation(s) s0, returning the logratio
        Mulitplies the logratio with the corresponding prior returing unnormalized posterior.
        Normalizes posterior so that posterior is a actual probability.
        """

        n_batch = len(s0['img'])

        # Get logratios
        z_sub_empty = self.z_sub_empty.copy()
        z_sub_empty['z_sub'] = z_sub_empty['z_sub'].repeat(n_batch, 1, 1) # Repeat so it works for batches
        
        z_sub_all = self.z_sub_all.copy()
        z_sub_all['z_sub'] = z_sub_all['z_sub'].repeat(n_batch, 1, 1) # Repeat so it works for batches
        
        logratios = torch.zeros((n_batch, 2, self.n_msc, self.n_pix, self.n_pix), device = DEVICE) 
        logratios[:,0] = self.network(s0, z_sub_empty)['z_pix'].ratios.detach() # log r(x,z=0)
        logratios[:,1] = self.network(s0, z_sub_all)['z_pix'].ratios.detach()   # log r(x,z=1)
        
        # Posterior   
        posts = torch.exp(logratios) * self.prior_grid(n_batch) # Calculate posterior = ratio * prior
        posts = posts.cpu()
        post_sum = torch.sum(posts, axis = 1) # Sum p(z=0|x) + p(z=1|x) for each pixel
        post_norm = posts[:,1] / post_sum # Normalize p(z=1|x)
        posts_unnorm = posts[:,1]
        return post_norm, posts_unnorm
    
    def get_posts(self, dataloader, max_n_test):
        """
        Calculates the N = max_n_test posteriors from the dataloader observations.      
        """
        n_dataset = len(dataloader.dataset)
        max_n_test = max_n_test if max_n_test <= n_dataset else n_dataset # Maximum number of prediction we want to
        max_n_test_batch = max_n_test // self.datamodule.batch_size + 1   # Corresponding number of batches we want to do

        posts_norm, posts_unnorm, targets = [], [], []
        for _, s_batch in tqdm(zip(range(max_n_test_batch), dataloader), total = max_n_test_batch, desc='Calculating posteriors'):
            post_norm, post_unnorm = self.get_post(s_batch)
            posts_norm.append( post_norm )
            posts_unnorm.append( post_unnorm )
            targets.append( self.network.classifier.paramtrans(s_batch['z_sub'].to(DEVICE)) )
        posts_norm = torch.cat(posts_norm)
        posts_unnorm = torch.cat(posts_unnorm)
        targets = torch.cat(targets).cpu()
#         posts_norm, posts_unnorm, targets = self.drop_nan_posts(posts_norm, posts_unnorm, targets)
        return posts_norm, posts_unnorm, targets
    
    def get_posts_from_sims(self, simulations, max_n_test):
        """
        Calculates the N = max_n_test posteriors from the dataloader observations.      
        """
        n_dataset = len(simulations)
        max_n_test = max_n_test if max_n_test <= n_dataset else n_dataset # Maximum number of prediction we want to
        batch_size = self.datamodule.batch_size
#         max_n_test_batch = max_n_test // self.datamodule.batch_size + 1   # Corresponding number of batches we want to do

        posts, targets = [], []
        
        for batch_idx in tqdm(range(int(np.ceil(max_n_test / batch_size))), desc='Calculating posteriors'):
            i, j = batch_idx*batch_size, (batch_idx+1)*batch_size
            s_batch = simulations[i:j]
        
            posts.append(  self.get_post(s_batch) )
            targets.append( self.network.classifier.paramtrans(s_batch['z_sub'].to(DEVICE)) )
        posts = torch.cat(posts)
        targets = torch.cat(targets)
#         posts, targets = self.drop_nan_posts(posts, targets)
        return posts, targets
