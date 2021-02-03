import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

import torch
from torch import tensor
import pyro
import pyro.distributions as dist
import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.utils.data.dataloader as dataloader

import pyro
from pyro import distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.autograd import Variable
import argparse

import os, sys
import tqdm
from stanpyro import dppl

from torch.nn import functional as F

sns.set_style("white")
sns.set_context("paper", font_scale=1.8)

StanName = "Stan"
DeepStanName = "DeepStan"
DeepStanSVIName = "DeepStanSVI"
PyroName = "Pyro"

colors = {
    DeepStanName: "#7fc97f",
    StanName: "#beaed4",
    DeepStanSVIName: "#fdc086",
    PyroName: "#beaed4",
}


def make_plots(left, right, path, ticks=None):
    plt.hist(
        left,
        bins=10,
        histtype="stepfilled",
        alpha=0.75,
        label=left.name,
        color=colors[left.name],
    )
    plt.hist(
        right,
        bins=10,
        histtype="stepfilled",
        alpha=0.5,
        label=right.name,
        color=colors[right.name],
    )
    plt.xlabel(r"Accuracy")
    if ticks:
        plt.xticks(*ticks)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()


def loadData(batch_size):
    train = MNIST(
        os.environ.get("DATA_DIR", ".") + "/data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),  # ToTensor does min-max normalization.
            ]
        ),
    )
    test = MNIST(
        os.environ.get("DATA_DIR", ".") + "/data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),  # ToTensor does min-max normalization.
            ]
        ),
    )
    dataloader_args = dict(
        shuffle=True, batch_size=batch_size, num_workers=3, pin_memory=False
    )
    train_loader = dataloader.DataLoader(train, **dataloader_args)
    test_loader = dataloader.DataLoader(test, **dataloader_args)
    return train_loader, test_loader


model_code = "mlp_model.stan"

batch_size, nx, nh, ny = 128, 28 * 28, 1024, 10


def build_mlp():
    # Model

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.l1 = torch.nn.Linear(nx, nh)
            self.l2 = torch.nn.Linear(nh, ny)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            h = self.relu(self.l1(x.view((-1, nx))))
            yhat = self.l2(h)
            return F.log_softmax(yhat, dim=-1)

    return MLP()


def predict(data, posterior):
    predictions = [model(data) for model in posterior]
    prediction = torch.stack(predictions).mean(dim=0)
    return prediction.argmax(dim=-1)


def guide_hand_coded(batch_size, imgs, labels, mlp):
    l1wloc = pyro.param("l1wloc", torch.randn(mlp.l1.weight.shape))
    l1wscale = pyro.param("l1wscale", torch.randn(mlp.l1.weight.shape))
    l1bloc = pyro.param("l1bloc", torch.randn(mlp.l1.bias.shape))
    l1bscale = pyro.param("l1bscale", torch.randn(mlp.l1.bias.shape))
    l2wloc = pyro.param("l2wloc", torch.randn(mlp.l2.weight.shape))
    l2wscale = pyro.param("l2wscale", torch.randn(mlp.l2.weight.shape))
    l2bloc = pyro.param("l2bloc", torch.randn(mlp.l2.bias.shape))
    l2bscale = pyro.param("l2bscale", torch.randn(mlp.l2.bias.shape))
    guide_dict = {
        "l1.weight": dist.Normal(l1wloc, F.softplus(l1wscale)),
        "l1.bias": dist.Normal(l1bloc, F.softplus(l1bscale)),
        "l2.weight": dist.Normal(l2wloc, F.softplus(l2wscale)),
        "l2.bias": dist.Normal(l2bloc, F.softplus(l2bscale)),
    }
    lifted_mlp = pyro.random_module("mlp", mlp, guide_dict)
    return lifted_mlp()


def model_hand_coded(batch_size, imgs, labels, mlp):
    prior = {
        "l1.weight": dist.Normal(
            torch.zeros(mlp.l1.weight.shape), torch.ones(mlp.l1.weight.shape)
        ),
        "l1.bias": dist.Normal(
            torch.zeros(mlp.l1.bias.shape), torch.ones(mlp.l1.bias.shape)
        ),
        "l2.weight": dist.Normal(
            torch.zeros(mlp.l2.weight.shape), torch.ones(mlp.l2.weight.shape)
        ),
        "l2.bias": dist.Normal(
            torch.zeros(mlp.l2.bias.shape), torch.ones(mlp.l2.bias.shape)
        ),
    }
    mlp = pyro.random_module("mlp", mlp, prior)()
    logits = mlp(imgs)
    pyro.sample("labels", dist.Categorical(logits=logits), obs=labels)


def generate_posterior(svi, n, args):
    return [svi.guide(*args) for _ in range(n)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", type=int, default=20, help="number of epochs to run")

    args = parser.parse_args()

    train_loader, test_loader = loadData(batch_size)
    pyro.clear_param_store()
    mlp_hand_coded = build_mlp()
    epochs = args.epochs
    lr = 0.01
    posterior_size = 100
    svi = pyro.infer.SVI(
        model=model_hand_coded,
        guide=guide_hand_coded,
        optim=pyro.optim.Adam({"lr": lr}),
        loss=pyro.infer.Trace_ELBO(),
    )
    for epoch in tqdm.tqdm(range(epochs)):  # loop over the dataset multiple times
        for j, (imgs, lbls) in enumerate(train_loader, 0):
            # calculate the loss and take a gradient step
            loss = svi.step(batch_size, imgs, lbls, mlp_hand_coded)

    posterior_hand_made = generate_posterior(
        svi, posterior_size, (None, None, None, mlp_hand_coded)
    )
    accuracies_hand_made = []
    for j, data in enumerate(test_loader, 0):
        images, labels = data
        accuracy = (
            (predict(images, posterior_hand_made) == labels).type(torch.float).mean()
        )
        accuracies_hand_made.append(accuracy)
    accuracies_hand_made = pd.Series([x.item() for x in accuracies_hand_made])
    print("Pyro")
    print(accuracies_hand_made.describe())

    pyro.clear_param_store()
    mlp = build_mlp()
    model = dppl.PyroModel(model_code)
    svi = model.svi(
        optim=pyro.optim.Adam({"lr": lr}),
        loss=pyro.infer.Trace_ELBO(),)
    losses = []
    for epoch in tqdm.tqdm(range(epochs)):  # loop over the dataset multiple times
        for j, (imgs, lbls) in enumerate(train_loader, 0):
            # calculate the loss and take a gradient step
            loss = svi.step(
                batch_size=batch_size,
                nx=nx,
                nh=nh,
                ny=ny,
                imgs=imgs,
                labels=lbls + 1,
                mlp=mlp,
            )
            losses.append(loss)
    posterior = svi.posterior(
        posterior_size,
        batch_size=batch_size,
        nx=nx,
        nh=nh,
        ny=ny,
        imgs=imgs,
        labels=lbls + 1,
        mlp=mlp,
    )
    posterior = [p["mlp"] for p in posterior]

    accuracies = []
    for j, data in enumerate(test_loader, 0):
        images, labels = data
        accuracy = (predict(images, posterior) == labels).type(torch.float).mean()
        accuracies.append(accuracy)
        assert accuracy > 0.6
    accuracies = pd.Series([x.item() for x in accuracies])
    print("DeepStan")
    print(accuracies.describe())

    agreements = []
    for j, data in enumerate(test_loader, 0):
        images, labels = data
        agreement = (
            (predict(images, posterior_hand_made) == predict(images, posterior))
            .type(torch.float)
            .mean()
        )
        agreements.append(agreement)

    agreements = pd.Series([x.item() for x in agreements])
    print("Agreement")
    print(agreements.describe())
    df = pd.DataFrame(
        {
            DeepStanSVIName: accuracies,
            PyroName: accuracies_hand_made,
            "agreement": agreements,
        }
    )

    print(f"KS_test (N={len(df)}):", stats.ks_2samp(df[DeepStanSVIName], df[PyroName]))

    make_plots(df[PyroName], df[DeepStanSVIName], f"pyro-vs-deepstan.pdf")
