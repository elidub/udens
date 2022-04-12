from .interpret import PostData, ObsData


class GetBounds(ObsData):
    def __init__(self, obs, obs_post, grid_coords, connectivity = 3):
        super().__init__(obs, grid_coords)
        
        
        self.obs_post = obs_post.squeeze(0).cpu().numpy()
        self.s = spi.generate_binary_structure(3, connectivity)
        
        self.obs_n_sub = len(self.z_sub)
        
        self.n_features_th, self.thresholds, self.lowest_threshold = self.determine_threshold()
        self.obs_post_labeled, self.n_features = self.label(self.lowest_threshold)
        self.bounds_low, self.bounds_high, self.bounds_low_idxs, self.bounds_high_idxs = self.determine_bounds(self.obs_post_labeled, self.n_features)
        
        
    def determine_threshold(self, n_thresholds = 50):
        thresholds = np.linspace(0., 1., n_thresholds)
        
        n_features_th = np.array([spi.label(self.obs_post > threshold, structure = self.s)[1] 
                                 for threshold in thresholds])

#         for threshold in thresholds:
#             plt_imshow( spi.label(self.obs_post > threshold, structure = self.s)[0], title = f'threshold = {threshold:.2f}', **imkwargs)

        correct_thresholds_idxs = np.where(n_features_th == self.obs_n_sub)[0]
        lowest_threshold = thresholds[correct_thresholds_idxs[0]]
        
        assert (np.diff(correct_thresholds_idxs) == 1).all(), "n_sub = num_features not stable!"
        return n_features_th, thresholds, lowest_threshold
    
    
    def label(self, threshold):
        obs_post_labeled, n_features = spi.label(self.obs_post > threshold, structure = self.s)
        
        if self.obs_n_sub != n_features:
            raise ValueError(f'number of subhalos found {n_features} is not the same as initiated {self.obs_n_sub}!')
        return obs_post_labeled, n_features
    
    
        
    def determine_bounds(self, obs_post_labeled, n_features):
        bounds_low_idxs  = np.zeros((self.obs_n_sub, 3), dtype = int)
        bounds_high_idxs = np.zeros((self.obs_n_sub, 3), dtype = int)

        for i, feature in enumerate(np.arange(1, n_features+1)):
            where_above_threshold = np.array(np.where(obs_post_labeled == feature))

            bounds_low_idxs[i]  = np.amin(where_above_threshold, axis = 1)
            bounds_high_idxs[i] = np.amax(where_above_threshold, axis = 1)

        bounds_low, bounds_high = np.array([
                        [[edges[:-1][i], edges[1:][j]] for i, j in zip(bounds_low_idx, bounds_high_idx)]
                        for edges, bounds_low_idx, bounds_high_idx in zip([self.m_edges, self.xy_edges, self.xy_edges], bounds_low_idxs.T, bounds_high_idxs.T)
                    ]).T
        return bounds_low, bounds_high, bounds_low_idxs, bounds_high_idxs

    
