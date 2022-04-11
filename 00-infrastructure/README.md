# `subN`

To infer the posteriors of (x,y,M) pixels with image segmentation (ImgSeg).

- `__init__.py` contains `ImgSegmNetwork`, the setup of the ImgSegm network. It does not use the standard swyft classifiers. 
- `classifier.py`: the custom classifiers designed for UNet.
- `unet.py`: only the actual neural network

Files below should be placed somewhere else, but I didn't what a good location would be
- `inference.py`: the custom inference for the UNet predictions
- `logging_utils_subN.py`: custom plotting of the inference
- `plot.py` function that plots multiple `plt.imshow` easy

To do:
- `DEVICE = 'cuda'` is all over the place, should be initiated in model classes.
