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
    n_sub_expect = torch.mean(torch.tensor(np.count_nonzero(Ms.numpy(), axis  = 1), dtype = torch.float32))
    
    # Loading the inference class and 
    infer = Infer(simulator, network, datamodule, n_sub_expect)
    
    # Prior information necessary for loggers
    prior, prior_grid = infer.calc_prior()[0], infer.prior_grid()
    grid_coords = infer.get_grid_coords()
    grid_low, grid_high = infer.grid_low, infer.grid_high
    
    # Simulations inference
#     posts_norm, posts_unnorm, targets = infer.get_posts(datamodule.predict_dataloader(), cfg.inference.n_infer)
#     torch.save(posts_norm, os.path.join(logdir, 'posts_norm.pt'))
#     torch.save(posts_unnorm, os.path.join(logdir, 'posts_unnorm.pt'))
#     torch.save(targets, os.path.join(logdir, 'targets.pt'))
    posts_norm = torch.load(os.path.join(logdir, 'posts_norm.pt'))
    posts_unnorm = torch.load(os.path.join(logdir, 'posts_unnorm.pt'))
    targets       = torch.load(os.path.join(logdir, 'targets.pt'))
    
    # Calibration
    irc_norm = IsotonicRegressionCalibration(posts_norm, targets)    
    posts_norm_calib = irc_norm.calibrate(posts_norm)
    torch.save(posts_norm_calib, os.path.join(logdir, 'posts_norm_calib.pt'))
    
    irc_unnorm = IsotonicRegressionCalibration(posts_unnorm, targets)    
    posts_unnorm_calib = irc_unnorm.calibrate(posts_unnorm)
    torch.save(posts_unnorm_calib, os.path.join(logdir, 'posts_unnorm_calib.pt'))
    
    
    # Log simulation inference
    LogPost(tbl, posts_norm, targets, title = 'norm_uncalib').plot_all()
    LogPost(tbl, posts_norm_calib, targets, title = 'norm_calib').plot_all()
    LogIRC(tbl, irc_norm, title = 'norm_calibration').plot()
    
    LogPost(tbl, posts_unnorm, targets, title = 'unnorm_uncalib').plot_all()
    LogPost(tbl, posts_unnorm_calib, targets, title = 'unnorm_calib').plot_all()
    LogIRC(tbl, irc_unnorm, title = 'unnorm_calibration').plot()

#     # Example observation
#     for _ in range(10000):
#         test_sim = simulator.sample(1)
#         if (test_sim['z_sub'][0,:,0] > 9.).sum() > 2:
#             break
    test_sim = simulator.sample(1)
    
    test_post_uncalib = infer.get_post(test_sim)[0].squeeze(0)
    test_sim = infer.squeeze_obs(test_sim)
    test_post = irc_norm.calibrate(test_post_uncalib)

    log_obs = LogObs(tbl, test_sim, test_post, prior, grid_coords)
    log_obs.plot_obs()

                              
    tbl.experiment.flush()
    print("logdir:", tbl.experiment.get_logdir())