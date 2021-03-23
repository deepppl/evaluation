import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import pyro
from pyro import distributions as dist
from pyro import infer
from pyro.infer import mcmc, Trace_ELBO
from pyro.optim import Adam
import tqdm

import torch
from torch.distributions import constraints

from cmdstanpy import CmdStanModel

from stanpyro.dppl import PyroModel
from stannumpyro.dppl import NumPyroModel
import jax.random
import time

num_chains = 1
iterations = 10000
warmup = 500
svi_steps = 70000

StanName = "Stan(NUTS)"
StanADVIName = "Stan(ADVI)"
DeepStanName = "DeepStan(NUTS)"
DeepStanSVIName = "DeepStan(VI)"

sns.set_style("white")
sns.set_context("paper", font_scale=2.5)
colors = {
    DeepStanName: "#7fc97f",
    StanName: "#beaed4",
    StanADVIName: "#ffff99",
    DeepStanSVIName: "#fdc086",
}

model_code = "multimodal_model.stan"
model_code_guided = "multimodal_guide_model.stan"


def deepstan_sampler(model_code):
    model = NumPyroModel(model_code)
    mcmc = model.mcmc(iterations // num_chains, warmup, num_chains)
    mcmc.run(jax.random.PRNGKey(11), {})
    samples = pd.Series(mcmc.get_samples()["theta"], name=r"$\theta$")
    return samples


def stan_sampler(model_code):
    sm = CmdStanModel(stan_file=model_code)
    fit = sm.sample(
        iter_sampling=iterations // num_chains,
        iter_warmup=warmup,
        chains=num_chains,
    )
    samples = fit.stan_variable("theta").theta
    return samples.iloc[np.random.permutation(len(samples))].reset_index(drop=True)


def stan_sampler_advi(model_code):
    sm = CmdStanModel(stan_file=model_code)
    fit = sm.variational(
        iter=svi_steps, algorithm="fullrank", output_samples=iterations, tol_rel_obj=10
    )
    samples = pd.Series(fit.variational_sample[fit.column_names.index("theta")])
    return samples.iloc[np.random.permutation(len(samples))].reset_index(drop=True)


def deepstan_svi_sampler(model_code):
    model_guided = PyroModel(model_code)
    optimizer = Adam({"lr": 0.005, "betas": (0.95, 0.999)})
    svi = model_guided.svi(optimizer, Trace_ELBO())
    for step in tqdm.tqdm(range(svi_steps)):
        svi.step({})
    samples = pd.Series(
        [float(model_guided.module.guide()["theta"]) for _ in range(iterations)]
    )
    return samples


def make_plots(left, right, path, ticks=None, xlim=None, ylim=None):
    plt.hist(
        left,
        range=xlim,
        bins=100,
        histtype="stepfilled",
        alpha=0.8,
        label=left.name,
        color=colors[left.name],
    )
    plt.hist(
        right,
        range=xlim,
        bins=100,
        histtype="stepfilled",
        alpha=0.75,
        label=right.name,
        color=colors[right.name],
    )
    plt.xlabel(r"$\theta$")
    if ticks is not None:
        plt.xticks(*ticks)
        if len(ticks) and not ticks[0]:
            plt.xlabel("")
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.legend(fontsize="x-small", loc="upper center")
    ax = plt.gca()
    ax.set_aspect(10 / 1000)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()


if __name__ == "__main__":
    deepstan_guided = deepstan_svi_sampler(model_code_guided)
    samples_deepstan = deepstan_sampler(model_code)
    samples_stan = stan_sampler(model_code)
    samples_stan_advi = stan_sampler_advi(model_code)
    df = pd.DataFrame(
        {
            DeepStanName: samples_deepstan,
            StanName: samples_stan,
            DeepStanSVIName: deepstan_guided,
            StanADVIName: samples_stan_advi,
        }
    )
    print(df[[StanName, DeepStanName, DeepStanSVIName, StanADVIName]].describe())
    make_plots(
        df[StanName],
        df[DeepStanSVIName],
        f"stan-vs-deepstansvi.pdf",
        xlim=[-5, 24], ylim=[0, 1200],
    )
    make_plots(
        df[DeepStanName],
        df[DeepStanSVIName],
        f"deepstan-vs-deepstansvi.pdf",
        xlim=[-5, 24], ylim=[0, 1200],
    )
    make_plots(
        df[DeepStanSVIName],
        df[StanADVIName],
        f"deepstansvi-vs-stanadvi.pdf",
        xlim=[-5, 24], ylim=[0, 1200],
    )
