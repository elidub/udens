import os
import numpy as np
import pylab as plt
import torch
import swyft.lightning as sl

from lensx.nn.subN.interpret import IsotonicRegressionCalibration
from lensx.nn.subN.logging_utils_subN import LogIRC, LogPost, LogObs, LogBounds, LogSingleSub
from lensx.nn.subN.inference import Infer, Prior

def interpret(cfg, simulator, network, trainer, datamodule, tbl):
    logdir = tbl.experiment.get_logdir()
    
    # Calculate expected n_sub
    Ms = datamodule.predict_dataloader().dataset[:]['z_sub'][:,:,0]
    n_sub_expect = torch.mean( torch.sum(Ms == 0, dim = 1).type(torch.float) )
    
    # Loading the inference class and 
    infer = Infer(simulator, network, datamodule, n_sub_expect)
    
    # Prior information necessary for loggers
    prior, prior_grid = infer.calc_prior()[0], infer.prior_grid()
    grid_coords = infer.get_grid_coords()
    grid_low, grid_high = infer.grid_low, infer.grid_high
    
    # Simulations inference
    posts_uncalib, targets = infer.get_posts(datamodule.predict_dataloader(), cfg.inference.n_infer)
    torch.save(posts_uncalib, os.path.join(logdir, 'posts_uncalib.pt'))
    torch.save(targets, os.path.join(logdir, 'targets.pt'))
#     posts_uncalib = torch.load(os.path.join(tbl.experiment.get_logdir(), 'posts_uncalib.pt'))
#     targets       = torch.load(os.path.join(tbl.experiment.get_logdir(), 'targets.pt'))
    
    # Calibration
    irc = IsotonicRegressionCalibration(posts_uncalib, targets)    
    posts_calib = irc.calibrate(posts_uncalib)
    torch.save(posts_calib, os.path.join(logdir, 'posts_calib.pt'))
    
    
    

    
#     # Observation inference
#     obs = torch.load(cfg.inference.obs_path)
#     obs_post = infer.get_post( dict(img=obs['img'].unsqueeze(0).cpu()))
    
#     # Log observation inference
#     log_obs = LogObs(tbl, obs, obs_post, prior, grid_coords, fig_kwargs = dict(dpi = 250, figsize = (8, 5)))
#     log_obs.plot_all()   
    
    # Log simulation inference
    LogPost(tbl, posts_uncalib, targets, fig_kwargs = dict(dpi = 100, figsize = (4,3))).plot_all()
    LogPost(tbl, posts_calib,   targets, fig_kwargs = dict(dpi = 100, figsize = (4,3)), calib = 'calibrated').plot_all()
    LogIRC(tbl, irc).plot()

#     if (cfg.simulation.model.n_sub, cfg.estimation.network.n_msc) == (1, 1):
#         log_single_sub = LogSingleSub(tbl, obs, obs_post, prior_grid, grid_coords)
#         log_single_sub.plot_all()

    # Log bounds
#     log_bounds = LogBounds(tbl, obs, obs_post, grid_coords, grid_low, grid_high)
#     log_bounds.plot_all()
                              
    tbl.experiment.flush()
    print("logdir:", tbl.experiment.get_logdir())