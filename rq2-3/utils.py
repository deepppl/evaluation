import os
import numpy, numpyro, pyro
import pathlib

from typing import Any, Dict, IO
from dataclasses import dataclass, field
from pandas import DataFrame, Series
from posteriordb import PosteriorDatabase
from os.path import splitext, basename
from itertools import product
from cmdstanpy import CmdStanModel

from stannumpyro.dppl import NumPyroModel, compile as compile_numpyro
from stanpyro.dppl import PyroModel, compile as compile_pyro
import jax.random


@dataclass
class Config:
    iterations: int
    warmups: int
    chains: int
    thin: int
    seed: int


def parse_config(posterior):
    """
    Parse configuration from PosteriorDB
    """
    args = posterior.reference_draws_info()["inference"]["method_arguments"]
    return Config(
        iterations=args["iter"],
        warmups=args["warmup"],
        chains=args["chains"],
        thin=args["thin"],
        seed=args["seed"],
    )


def _valid_ref(pdb, name):
    """
    Test if reference exists in PosteriorDB
    """
    try:
        posterior = pdb.posterior(name)
        posterior.reference_draws_info()
        return True
    except Exception:
        return False


pdb_root = "../posteriordb"
pdb_path = os.path.join(pdb_root, "posterior_database")
my_pdb = PosteriorDatabase(pdb_path)
golds = [x for x in my_pdb.posterior_names() if _valid_ref(my_pdb, x)]


def get_posterior(name):
    return my_pdb.posterior(name)


def gold_summary(posterior):
    samples = posterior.reference_draws()
    if isinstance(samples, list):
        # Multiple chains
        assert len(samples) > 0
        res = samples[0]
        for c in samples[1:]:
            res = {k: v + c[k] for k, v in res.items()}
    else:
        # Only one chain
        assert isinstance(samples, dict)
        res = samples
    res = {k: numpy.array(v) for k, v in res.items()}
    summary_dict = numpyro.diagnostics.summary(res, group_by_chain=False)
    columns = list(summary_dict.values())[0].keys()
    index = []
    rows = []
    for name, stats_dict in summary_dict.items():
        shape = stats_dict["mean"].shape
        if len(shape) == 0:
            index.append(name)
            rows.append(stats_dict.values())
        else:
            for idx in product(*map(range, shape)):
                idx_str = "[{}]".format(",".join(map(str, idx)))
                index.append(name + idx_str)
                rows.append([v[idx] for v in stats_dict.values()])
    return DataFrame(rows, columns=columns, index=index)


def compile_pyro_model(posterior, backend, mode):
    model = posterior.model
    stanfile = model.code_file_path("stan")
    build_dir = f"_build_{backend}_{mode}"

    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
        pathlib.Path(f"{build_dir}/__init__.py").touch()

    if backend == "numpyro":
        compile_numpyro(mode, stanfile, build_dir=build_dir)
    else:
        compile_pyro(mode, stanfile, build_dir=build_dir)


def compile_stan_model(posterior):
    stanfile = posterior.model.code_file_path(framework="stan")
    _ = CmdStanModel(stan_file=stanfile)


def compile_model(*, posterior, backend, mode):
    if backend == "stan":
        compile_stan_model(posterior)
    else:
        compile_pyro_model(posterior, backend, mode)


def run_pyro_model(*, posterior, backend, mode, config):
    """
    Compile and run the model.
    Returns the summary Dataframe
    """
    model = posterior.model
    data = posterior.data.values()
    stanfile = model.code_file_path("stan")
    build_dir = f"_build_{backend}_{mode}"
    if backend == "numpyro":
        numpyro_model = NumPyroModel(stanfile, recompile=False, build_dir=build_dir)
        mcmc = numpyro_model.mcmc(
            samples=config.iterations,
            warmups=config.warmups,
            chains=config.chains,
            thin=config.thin,
        )
        mcmc.run(jax.random.PRNGKey(config.seed), data)
        return mcmc
    elif backend == "pyro":
        pyro.set_rng_seed(config.seed)
        pyro_model = PyroModel(stanfile, recompile=False, build_dir=build_dir)
        mcmc = pyro_model.mcmc(
            samples=config.iterations,
            warmups=config.warmups,
            chains=config.chains,
            thin=config.thin,
        )
        mcmc.run(data)
        return mcmc
    else:
        assert False, "Invalid backend (should be one of pyro, numpyro, or stan)"


def run_stan_model(*, posterior, config):
    """
    Compile and run the stan model
    Return the summary Dataframe
    """
    stanfile = posterior.model.code_file_path(framework="stan")
    model = CmdStanModel(stan_file=stanfile)
    data = posterior.data.values()
    fit = model.sample(
        data=data,
        iter_warmup=config.warmups,
        iter_sampling=config.iterations,
        thin=config.thin,
        chains=config.chains,
        seed=config.seed,
    )
    return fit
