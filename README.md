# udens

## Structure
- `src/udens` contains all the code that do the classification/inference
  - `src/udens/models` contains the simulators
- `run_blobs` does the simulating, analysing and interpreting with swyft-lightning

@Christopher: I propose that you make a new folder very similar to `run_blobs` where you analyze your model which is inside `src/udens/models`

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

## Background
Ongoing notes of the math are on [Overleaf](https://www.overleaf.com/project/624bff9f26e3a2a309468557)


