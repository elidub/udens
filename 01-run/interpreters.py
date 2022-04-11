import os
import numpy as np
import pylab as plt
import torch
import swyft.lightning as sl

from lensx.nn.subN.logging_utils_subN import LogPost, LogObs, LogBounds, LogSingleSub
from lensx.nn.subN.inference import Infer

def interpret(cfg, simulator, network, trainer, datamodule, tbl):
    # Loading the inference class and 
    infer = Infer(simulator, network, datamodule)
    
    # Prior information necessary for loggers
    prior, prior_grid = infer.calc_prior(), infer.prior_grid()
    grid_coords = infer.get_grid_coords()
    grid_low, grid_high = infer.grid_low, infer.grid_high
    
    # Simulations inference
    posts, targets = infer.get_posts(datamodule.predict_dataloader(), cfg.inference.n_infer)
    torch.save(posts, os.path.join(tbl.experiment.get_logdir(), 'posts.pt'))
    torch.save(targets, os.path.join(tbl.experiment.get_logdir(), 'targets.pt'))
    
    # Observation inference
    obs = torch.load(cfg.inference.obs_path)
    obs_post = infer.get_post( dict(img=obs['img'].unsqueeze(0).cpu()))
    
    # Log observation inference
    log_obs = LogObs(tbl, obs, obs_post, prior, grid_coords, fig_kwargs = dict(dpi = 250, figsize = (8, 5)))
    log_obs.plot_all()   
    
    # Log simulation inference
    log_post = LogPost(tbl, posts, targets, fig_kwargs = dict(dpi = 250))
    log_post.plot_all()
    
    if (cfg.simulation.model.n_sub, cfg.estimation.network.n_msc) == (1, 1):
        log_single_sub = LogSingleSub(tbl, obs, obs_post, prior_grid, grid_coords)
        log_single_sub.plot_all()

    # Log bounds
#     log_bounds = LogBounds(tbl, obs, obs_post, grid_coords, grid_low, grid_high)
#     log_bounds.plot_all()
                              
    tbl.experiment.flush()
    print("logdir:", tbl.experiment.get_logdir())