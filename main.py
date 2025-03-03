"""
CLI entrypoint for climate uncertainty diffusion.
"""

import os
import click
import torch
import wandb
from torch.utils.data import DataLoader

import src.utils.data_utils as du
import src.models.variational_encoder as ve

from src.models.diffusion import DiffusionModel, DiffusionUNet
from src.train.train_diffusion import train_diffusion
from src.test.diffusion import test_diffusion

from src.utils.cli import (
    batch_size,
    checkpoints,
    device,
    save_results,
    epochs,
    learning_rate,
    use_wandb,
)

from src.utils.files import (
    get_filename,
    recursive_list_files,
)


@click.group()
def main():
    pass


@main.command()
@batch_size
@device
@epochs
@learning_rate
@use_wandb
@click.option(
    "--noise_scale",
    help="The scale of the noise to add to the input, if None, std will be recomputed.",
    type=float,
    default=None,
)
@click.option(
    "--window_size",
    help="The size of the time window to use for the diffusion model",
    type=int,
    default=20,
)
@click.option(
    "--overlap",
    help="The overlap between windows",
    type=int,
    default=4,
)
def train(
    batch_size: int,
    device: str,
    epochs: int,
    learning_rate: float,
    noise_scale: float | None,
    window_size: int,
    overlap: int,
    use_wandb: bool,
    run_name: str | None,
    run_id: str | None,
):
    """Train Diffusion."""

    if use_wandb:
        run = wandb.init(
            entity="corentin-lachevre-ensta-paris",
            project="climate-uncertainty-diffusion",
            name=run_name,
            id=run_id,
            resume="must",
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "beta_start": 0.0001,
                "beta_end": 0.008,
                "epochs": epochs,
                "window_size": window_size,
                "overlap": overlap,
            },
        )

    # Load the dataset
    dataset = du.open_grib_file(
        "/home/ensta/ensta-lachevre/climate-uncertainty-diffusion/dataset.grib"
    )

    image_sequence_dataset = du.ImageSequenceDataset(
        dataset=dataset,
        window_size=window_size,
        overlap=overlap,
        variables=["u10", "v10", "t2m"],
        skip_last=True,
    )

    # Initialiser le DataLoader
    data_loader = DataLoader(
        image_sequence_dataset,
        batch_size=batch_size,  # Nombre de fenêtres par batch
        shuffle=True,  # Mélanger les données
        num_workers=2,  # Nombre de workers pour le chargement des données
        drop_last=True,  # Ignorer les batches incomplets
    )

    # Initialiser le VAE
    vae = ve.VAE()
    LATENT_CHANNELS = 4  # Nombre de canaux de la représentation latente

    # Variance du bruit résiduel des données
    noise_scale = (
        noise_scale
        if noise_scale is not None
        else du.compute_residual_variance(data_loader, vae)
    )

    # Initialiser le modèle de diffusion
    transformer_time_steps = (
        2  # For example, 4 channels split into 2 tokens (each of dimension 2).
    )

    assert (
        LATENT_CHANNELS % transformer_time_steps == 0
    ), "LATENT_CHANNELS must be divisible by transformer_time_steps"

    model = DiffusionUNet(
        in_channels=LATENT_CHANNELS * window_size,
        base_channels=192,
        time_emb_dim=128,
        transformer_time_steps=transformer_time_steps,
    )
    # Initialiser l'optimiseur
    diffusion = DiffusionModel(
        model, timesteps=1000, noise_scale=noise_scale, device=device
    )
    optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=learning_rate)

    if os.path.exists(
        "/home/ensta/ensta-lachevre/climate-uncertainty-diffusion/checkpoints/diffusion_model.pt"
    ):
        model.load_state_dict(
            torch.load(
                "/home/ensta/ensta-lachevre/climate-uncertainty-diffusion/checkpoints/diffusion_model.pt"
            )
        )
        optimizer.load_state_dict(
            torch.load(
                "/home/ensta/ensta-lachevre/climate-uncertainty-diffusion/checkpoints/diffusion_optimizer.pt"
            )
        )
        print("Model loaded.")

    

    # Entraîner le modèle
    train_diffusion(
        diffusion,
        vae,
        optimizer,
        data_loader,
        device,
        run,
        epochs,
        window_size,
        LATENT_CHANNELS,
    )


@main.command()
@batch_size
@device
@click.option(
    "--noise_scale",
    help="The scale of the noise to add to the input, if None, std will be recomputed.",
    type=float,
    default=None,
)
@click.option(
    "--window_size",
    help="The size of the time window to use for the diffusion model",
    type=int,
    default=20,
)
@click.option(
    "--overlap",
    help="The overlap between windows",
    type=int,
    default=4,
)
@click.option(
    "--n_simulations",
    help="The number of simulations to run",
    type=int,
    default=1,
)
@click.option(
    "--compute_dataset_stats",
    help="Whether to compute statistics of the dataset",
    type=bool,
    default=False,
    is_flag=True,
)
def test(
    batch_size: int,
    device: str,
    noise_scale: float | None,
    window_size: int,
    overlap: int,
    n_simulations: int,
    compute_dataset_stats: bool,
):
    """Train Diffusion."""
    # Load the dataset
    dataset = du.open_grib_file(
        "/home/ensta/ensta-lachevre/climate-uncertainty-diffusion/dataset.grib"
    )

    image_sequence_dataset = du.TestImageSequenceDataset(
        dataset=dataset,
        window_size=window_size,
        overlap=overlap,
        variables=["u10", "v10", "t2m"],
    )

    # Initialiser le DataLoader
    data_loader = DataLoader(
        image_sequence_dataset,
        batch_size=batch_size,  # Nombre de fenêtres par batch
        shuffle=False,  # Mélanger les données
        num_workers=2,  # Nombre de workers pour le chargement des données
        drop_last=True,  # Ignorer les batches incomplets
    )

    # Initialiser le VAE
    vae = ve.VAE()
    LATENT_CHANNELS = 4  # Nombre de canaux de la représentation latente

    # Variance du bruit résiduel des données
    noise_scale = (
        noise_scale
        if noise_scale is not None
        else du.compute_residual_variance(data_loader, vae)
    )

    # Initialiser le modèle de diffusion
    transformer_time_steps = (
        2  # For example, 4 channels split into 2 tokens (each of dimension 2).
    )

    assert (
        LATENT_CHANNELS % transformer_time_steps == 0
    ), "LATENT_CHANNELS must be divisible by transformer_time_steps"

    model = DiffusionUNet(
        in_channels=LATENT_CHANNELS * window_size,
        base_channels=192,
        time_emb_dim=128,
        transformer_time_steps=transformer_time_steps,
    )
    if os.path.exists(
        "/home/ensta/ensta-lachevre/climate-uncertainty-diffusion/checkpoints/diffusion_model.pt"
    ):
        model.load_state_dict(
            torch.load(
                "/home/ensta/ensta-lachevre/climate-uncertainty-diffusion/checkpoints/diffusion_model.pt"
            )
        )
        print("Model loaded.")
    diffusion = DiffusionModel(
        model, timesteps=1000, noise_scale=noise_scale, device=device
    )

    # Tester le modèle
    dataset_stats = du.compute_statistics(dataset, compute_dataset_stats)
    time_values = [str(date) for date in dataset["time"].values]
    test_diffusion(
        diffusion,
        vae,
        data_loader,
        device,
        window_size,
        LATENT_CHANNELS,
        ["u10", "v10", "t2m"],
        image_sequence_dataset.max_values,
        n_simulations=n_simulations,
        time_values=time_values,
    )


if __name__ == "__main__":
    main()
