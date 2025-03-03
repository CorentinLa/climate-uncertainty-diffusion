"""Reusable CLI flags and options"""

import click
import torch


def batch_size(func):
    return click.option(
        "--batch_size",
        default=32,
        help="The batch size to use for training/testing",
        type=int,
    )(func)

def epochs(func):
    return click.option(
        "--epochs",
        default=10,
        help="The number of epochs to train for",
        type=int,
    )(func)

def learning_rate(func):
    return click.option(
        "--learning_rate",
        default=1e-3,
        help="The learning rate to use for training",
        type=float,
    )(func)


def device(func):
    return click.option(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=click.Choice(["cpu", "cuda"]),
        help="The device to use. Defaults to 'cuda' if available, otherwise 'cpu'",
    )(func)


def checkpoints(func, required=True):
    return click.option(
        "--checkpoints",
        help="The path to the model checkpoints",
        required=required,
        type=str,
    )(func)


def save_results(func):
    return click.option(
        "--save_results",
        default=None,
        help="The path to save the results. If not provided, results will not be saved.",
        type=str,
    )(func)

def use_wandb(func):
    func = click.option(
        "--use_wandb",
        is_flag=True,
        help="Whether to use Weights & Biases for logging",
    )(func)
    func = click.option(
        "--run_name",
        default=None,
        help="The name of the run in Weights & Biases",
        type=str,
    )(func)
    func = click.option(
        "--run_id",
        default=None,
        help="The ID of the run in Weights & Biases",
        type=str,
    )(func)
    return func