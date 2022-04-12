# `udens`

To infer the posteriors of (M,x,y) pixels with image segmentation (ImgSeg). Please note that this is written for (M,x,y) analysis in the `pyrofit-lensing` pipeline! It has not been rewritten for generalizations. ** So please make heavy changes for generalizations!**

## Structure
- `models` containing the simulator to model.
- `__init__.py` contains `ImgSegmNetwork`, the setup of the ImgSegm network. It does not use the standard swyft classifiers. 
- `bounds.py` some start for bounding, but it is not up to date currenntly
- `classifier.py`: the custom classifiers designed for UNet.
- `inference.py`: the custom inference for the UNet predictions
- `interpret.py`: necessities for calibration and observations 
- `lavalamp.py`: plot function to plot the nice lavalamp
- `log.py`: custom plotting of the inference in tensorboard
- `plot.py` function that plots multiple `plt.imshow` easy
- `unet.py`: only the actual neural network

## To do:
- `DEVICE = 'cuda'` is all over the place, should be initiated in model classes.
