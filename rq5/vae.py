import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style("white")

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


import os, sys
import tqdm
import argparse
from stanpyro import dppl

from sklearn.cluster import KMeans


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


side = 28
batch_size, nx, nh, nz = 256, side * side, 1024, 5


def build_vae():
    # Model

    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.lh = nn.Linear(nz, nh)
            self.lx = nn.Linear(nh, nx)

        def forward(self, z):
            hidden = F.relu(self.lh(z))
            mu = self.lx(hidden)
            return torch.sigmoid(mu.view(-1, 1, side, side))

    # define the PyTorch module that parameterizes the
    # diagonal gaussian distribution q(z|x)
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.lh = torch.nn.Linear(nx, nh)
            self.lz_mu = torch.nn.Linear(nh, nz)
            self.lz_sigma = torch.nn.Linear(nh, nz)
            self.softplus = nn.Softplus()

        def forward(self, x):
            x = x.view((-1, nx))
            hidden = F.relu(self.lh(x))
            z_mu = self.lz_mu(hidden)
            z_sigma = self.softplus(self.lz_sigma(hidden))
            return z_mu, z_sigma

    return Encoder(), Decoder()


tp = lambda star, hat: np.multiply(star, hat).sum().sum().astype(float)
fp = lambda star, hat: (star < hat).sum().sum().astype(float)
fn = lambda star, hat: (star > hat).sum().sum().astype(float)


def metrics(star, hat):
    _tp, _fp, _fn = [f(star, hat) for f in (tp, fp, fn)]
    p = _tp / (_tp + _fp)
    r = _tp / (_tp + _fn)
    return p, r


def f1(p, r):
    return 2.0 * (p * r) / (p + r)


def evaluate(encoder, name, pair_star, test_loader):
    encoded = encoder(test_loader.dataset.data.float())[0].data.numpy()
    encoded_df = pd.DataFrame(encoded)
    encoded_df.to_csv(f"{name}-vae.csv")
    kmeans = KMeans(n_clusters=10, random_state=0)
    fitted = kmeans.fit(encoded)
    pair_hat = make_pair_table(fitted.labels_)
    precision, recall = metrics(pair_star, pair_hat)
    print(
        f"{name}  precision: {precision}, recall: {recall}, f1: {f1(precision, recall)}"
    )


def train_and_evaluate(
    epochs, name, pair_star, svi, encoder, decoder, train_loader, test_loader
):
    for epoch in tqdm.tqdm(range(epochs)):  # loop over the dataset multiple times
        for j, (imgs, _) in enumerate(train_loader, 0):
            # calculate the loss and take a gradient step
            k = len(imgs)
            loss = svi.step(
                batch_size=k, nz=nz, x=imgs, encoder=encoder, decoder=decoder
            )
    evaluate(encoder, name, pair_star, test_loader)


def build_and_evaluate(
    epochs, name, builder, params, pair_star, train_loader, test_loader
):
    encoder, decoder = build_vae()
    svi = builder(params)
    train_and_evaluate(
        epochs, name, pair_star, svi, encoder, decoder, train_loader, test_loader
    )


def make_pair_table(labels):
    y = pd.get_dummies(labels)
    return y.dot(y.T)


model_code = "vae_model.stan"


def guide_pyro(*, batch_size, nz, x, encoder, decoder):
    pyro.module("encoder_pyro", encoder)
    mu, sigma = encoder(x)
    latent = pyro.sample("latent", dist.Normal(mu, sigma, batch_size))


def model_pyro(*, batch_size, nz, x, encoder, decoder):
    pyro.module("decoder_pyro", decoder)
    latent = pyro.sample(
        "latent", dist.Normal(torch.zeros(nz), torch.ones(nz), batch_size)
    )
    loc_img = decoder(latent)
    pyro.sample("x", dist.Bernoulli(loc_img), obs=x)


def build_deep_stan_svi(params):
    model = dppl.PyroModel(model_code)
    svi = model.svi(params=params)
    return svi


def build_pyro_svi(params):
    svi = pyro.infer.SVI(
        model=model_pyro,
        guide=guide_pyro,
        optim=pyro.optim.Adam(params),
        loss=pyro.infer.Trace_ELBO(),
    )
    return svi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", type=int, default=20, help="number of epochs to run")

    args = parser.parse_args()

    train_loader, test_loader = loadData(batch_size)
    pair_star = make_pair_table(test_loader.dataset.targets.data.numpy())
    params = {"lr": 0.01}
    build_and_evaluate(
        args.epochs,
        "DeepStanSVI",
        build_deep_stan_svi,
        params,
        pair_star,
        train_loader,
        test_loader,
    )
    pyro.clear_param_store()
    build_and_evaluate(
        args.epochs,
        "Pyro",
        build_pyro_svi,
        params,
        pair_star,
        train_loader,
        test_loader,
    )
