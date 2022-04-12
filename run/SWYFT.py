import os
import shutil

import hydra
import numpy as np
import pylab as plt
import swyft.lightning as sl
import torch
from lensx.logging_utils import log_post_plots, log_target_plots, log_train_plots
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import sys

plt.switch_backend("agg")

def print_dict(d, indent=0):
    """
    Pretty print the config.yaml file
    """
    instance = type(d)
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, instance):
            print_dict(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))


def simulate(cfg):
    # Loading simulator (potentially bounded)
    simulator = hydra.utils.instantiate(cfg.simulation.model)

    # Generate or load training data & generate datamodule
    train_samples = sl.file_cache(
        lambda: simulator.sample(cfg.simulation.store.store_size),
        cfg.simulation.store.path,
    )[: cfg.simulation.store.train_size]
    datamodule = sl.SwyftDataModule(
        store=train_samples,
        model=simulator,  # Adds noise on the fly. `None` uses noise in store.
        batch_size=cfg.estimation.batch_size,
        num_workers=cfg.estimation.num_workers,
    )

    return datamodule, simulator

def load(cfg, simulator):
    print('Loading trained network')
    tbl = pl_loggers.TensorBoardLogger(
        save_dir=cfg.tensorboard.save_dir,
        name=cfg.tensorboard.name,
        version=cfg.tensorboard.version,
        default_hp_metric=False,
    )
    logdir = (
        tbl.experiment.get_logdir()
    )  # Directory where all logging information and checkpoints etc are stored
    
    checkpoints = os.listdir( os.path.join(logdir, 'checkpoint') )
    if 'best.ckpt' in checkpoints:
        best_ckpt = 'best.ckpt'
    else:
        best_idx = np.argmax(list(map(int, [checkpoint[6:8] for checkpoint in checkpoints])))
        best_ckpt = checkpoints[best_idx]
    print(f'best checkpoint is {best_ckpt}')
    
    checkpoint = torch.load(
        os.path.join(logdir, f'checkpoint/{best_ckpt}'), map_location='cpu'
    )

    network = hydra.utils.instantiate(cfg.estimation.network, cfg)
    network.load_state_dict(checkpoint["state_dict"])

    train_samples = torch.load(cfg.simulation.store.path)
    
    trainer = sl.SwyftTrainer(accelerator=cfg.estimation.accelerator, gpus=1)
    trainer.setup(None)
    
    datamodule = sl.SwyftDataModule(store=train_samples, model=simulator)
    datamodule.setup()
    
    trainer.model = network
    
    return network, trainer, tbl, datamodule

def analyse(cfg, datamodule):
    # Setting up tensorboard logger, which defines also logdir (contains trained network)
    tbl = pl_loggers.TensorBoardLogger(
        save_dir=cfg.tensorboard.save_dir,
        name=cfg.tensorboard.name,
        version=cfg.tensorboard.version,
        default_hp_metric=False,
    )
    logdir = (
        tbl.experiment.get_logdir()
    )  # Directory where all logging information and checkpoints etc are stored

    # Load network and train (or re-load trained network)
    network = hydra.utils.instantiate(cfg.estimation.network, cfg)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=cfg.estimation.early_stopping.min_delta,
        patience=cfg.estimation.early_stopping.patience,
        verbose=False,
        mode="min",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=logdir + "/checkpoint/",
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    trainer = sl.SwyftTrainer(
        accelerator=cfg.estimation.accelerator,
        gpus=1,
        max_epochs=cfg.estimation.max_epochs,
        logger=tbl,
        callbacks=[lr_monitor, early_stop_callback, checkpoint_callback],
    )
    best_checkpoint = logdir + "/checkpoint/best.ckpt"
    if not os.path.isfile(best_checkpoint):
        trainer.fit(network, datamodule)
        shutil.copy(checkpoint_callback.best_model_path, best_checkpoint)
        trainer.test(network, datamodule)
    else:
        trainer.fit(network, datamodule, ckpt_path=best_checkpoint)

    return network, trainer, tbl


def interpret(cfg, simulator, network, trainer, datamodule, tbl):
    hydra.utils.call(
        cfg.inference.interpreter, cfg, simulator, network, trainer, datamodule, tbl
    )

@hydra.main(config_path=".", config_name="config")
def main(cfg):
    print_dict(cfg)
    datamodule, simulator = simulate(cfg)
    
    if cfg.load:
        network, trainer, tbl, datamodule = load(cfg, simulator)
    else:
        network, trainer, tbl = analyse(cfg, datamodule)
    interpret(cfg, simulator, network, trainer, datamodule, tbl)


if __name__ == "__main__":
    main()
