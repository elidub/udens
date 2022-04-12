# udens

## Structure
- `src/udens` contains all the code that do the classification/inference
  - `src/udens/models` contains the simulators
- `run_blobs` does the simulating, analysing and interpreting with swyft-lightning
- TO DO: `run_pointsource` a folder very similar to `run_blobs` but for pointsources.

## Installation
```
git clone https://github.com/elidub/udens.git
pip install -e udens
```
It is my first time making something `pip`-installable, so some things might not work directly.

It is using the [38f15af](https://github.com/undark-lab/swyft/commit/38f15aff59d4ad8378226e1acf9561f21773d453) commit on the `lightning` branch:
```
cd ~/swyft
git checkout -b lightning_with_hydra 38f15af
```

One should also install some extra things, these should be transfered to the `setup.cfg`
```
pip install jupyter
pip install hydra-core --upgrade
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

## Background
Ongoing notes of the math are on Overleaf (see Slack for link)


