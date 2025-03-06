# Diffusion model for climate simulation uncertainty

Based on the paper [Latent Diffusion Model for Generating Ensembles of Climate Simulations](https://arxiv.org/abs/2407.02070)

This repository gives a possible implementation for the paper, using ERA5 dataset.

## Quickstart

Install the necessary python dependencies (soon):

```bash
pip install -r requirements.txt
```

Download the dataset in grib format (u10, v10, 2m temperature):


https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download


## CLI Commands

The following cli commands are available. Run `python main.py --help` for more information.

| Command              | Description                                 |
| -------------------- | ------------------------------------------- |
| `test`               | Test the diffusion                          |
| `train`              | Train the diffusion model                   |

### Testing the model

Make sure that you have the relevant model checkpoints. \

```bash
python main.py test --batch_size 16 --device cuda --noise_scale 0.43390241265296936  --window_size 5 --overlap 1 --n_simulations 2
```

### Training the model

```bash
python main.py train --batch_size 32 --device cuda --epochs 100 --learning_rate 1e-3 --noise_scale 0.4351429343223572 --window_size 5 --overlap 1
```
