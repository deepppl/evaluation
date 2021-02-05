import time, datetime
import os, sys, traceback, logging, argparse
import numpy, numpyro
import pathlib
import re

from typing import Any, Dict, IO
from dataclasses import dataclass, field
from pandas import DataFrame, Series
from posteriordb import PosteriorDatabase
from os.path import splitext, basename
from itertools import product
from cmdstanpy import CmdStanModel

from stannumpyro.dppl import NumPyroModel, compile
from stanpyro.dppl import PyroModel
import jax.random


logger = logging.getLogger(__name__)


@dataclass
class Config:
    iterations: int
    warmups: int
    chains: int
    thin: int


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
    )


def valid_ref(pdb, name):
    """
    Test if reference exists in PosteriorDB
    """
    try:
        posterior = pdb.posterior(name)
        posterior.reference_draws_info()
        return True
    except Exception:
        return False


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
    if not os.path.exists("_tmp"):
        os.makedirs("_tmp")
        pathlib.Path("_tmp/__init__.py").touch()

    model = posterior.model
    stanfile = model.code_file_path("stan")
    compile(args.mode, stanfile)


def compile_stan_model(posterior):
    stanfile = posterior.model.code_file_path(framework="stan")
    _ = CmdStanModel(stan_file=stanfile)


def compile_model(*, posterior, backend, mode):
    if backend == "stan":
        compile_stan_model(posterior)
    else:
        compile_pyro_model(posterior, backend, mode)


def run_pyro_model(*, posterior, backend, config):
    """
    Compile and run the model.
    Returns the summary Dataframe
    """
    model = posterior.model
    data = posterior.data.values()
    stanfile = model.code_file_path("stan")
    if backend == "numpyro":
        numpyro_model = NumPyroModel(stanfile, recompile=False)
        mcmc = numpyro_model.mcmc(
            samples=config.iterations,
            warmups=config.warmups,
            chains=config.chains,
            thin=config.thin,
        )
        mcmc.run(jax.random.PRNGKey(0), data)
        return mcmc.summary()
    elif backend == "pyro":
        pyro_model = PyroModel(stanfile, recompile=False)
        mcmc = pyro_model.mcmc(
            samples=config.iterations,
            warmups=config.warmups,
            chains=config.chains,
            thin=config.thin,
        )
        mcmc.run(data)
        return mcmc.summary()
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
    )
    summary = fit.summary()
    summary = summary[~summary.index.str.endswith("__")]
    summary = summary.rename(columns={"Mean": "mean", "StdDev": "std"})
    return summary[["mean", "std"]]


class ComparisonError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def compare(*, posterior, backend, mode, config):
    """
    Compare gold standard with model.
    """
    logger.info(f"Processing {posterior.name}")
    sg = gold_summary(posterior)
    if backend == "stan":
        sm = run_stan_model(posterior=posterior, config=config)
    else:
        sm = run_pyro_model(posterior=posterior, backend=backend, config=config)
    sm["err"] = abs(sm["mean"] - sg["mean"])
    sm["rel_err"] = sm["err"] / sg["std"]
    assert not sm.dropna().empty
    # perf_cmdstan condition: err > 0.0001 and (err / stdev) > 0.3
    comp = sm[(sm["err"] > 0.0001) & (sm["rel_err"] > 0.3)].dropna()
    if not comp.empty:
        logger.error(f"Failed {posterior.name}")
        raise ComparisonError(str(comp))
    else:
        logger.info(f"Success {posterior.name}")


@dataclass
class Monitor:
    """
    Monitor execution and log results in a csv file.
    - Successes log `success` and duration
    - Comparison errors log `mismatch` and duration
    - Failures log `error` and the exception string (no duration)
    - !! All exception are catched (including keyboard interuptions)
    """

    name: str
    file: IO

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        duration = time.perf_counter() - self.start
        if exc_type == ComparisonError:
            print(f"{name},{duration},mismatch", file=self.file, flush=True)
            return True
        elif not exc_type:
            print(f"{name},{duration},success", file=self.file, flush=True)
            return True
        else:
            return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run experiments on PosteriorDB models."
    )
    parser.add_argument(
        "--backend",
        help="inference backend (pyro, numpyro, or stan)",
        required=True,
    )
    parser.add_argument(
        "--mode",
        help="compilation mode (generative, comprehensive, mixed)",
        default="comprehensive",
    )
    parser.add_argument("--runs", type=int, default=1, help="number of runs")

    parser.add_argument(
        "--scaled",
        help="Run scaled down experiment (iterations = 10, warmups = 10, chains = 1, thin = 1)",
        action="store_true",
    )

    # Override posteriorDB configs
    parser.add_argument("--iterations", type=int, help="number of iterations")
    parser.add_argument("--warmups", type=int, help="warmups steps")
    parser.add_argument("--chains", type=int, help="number of chains")
    parser.add_argument("--thin", type=int, help="thinning factor")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    numpyro.set_host_device_count(20)
    # numpyro.set_platform('gpu')

    pdb_root = "../posteriordb"
    pdb_path = os.path.join(pdb_root, "posterior_database")
    my_pdb = PosteriorDatabase(pdb_path)

    if not os.path.exists("logs"):
        os.makedirs("logs")

    for i in range(args.runs):

        today = datetime.datetime.now()
        logpath = f"logs/{today.strftime('%y%m%d_%H%M')}_{args.backend}"
        if args.backend != "stan":
            logpath += f"_{args.mode}"
        logpath += f"_{i}.csv"

        golds = [x for x in my_pdb.posterior_names() if valid_ref(my_pdb, x)]

        with open(logpath, "a") as logfile:
            print(",time,status,exception", file=logfile, flush=True)
            for name in (n for n in golds):
                # Configurations
                posterior = my_pdb.posterior(name)

                if args.scaled:
                    config = Config(iterations=10, warmups=10, chains=1, thin=1)
                else:
                    config = parse_config(posterior)

                if args.iterations is not None:
                    config.iterations = args.iterations
                if args.warmups is not None:
                    config.warmups = args.warmups
                if args.chains is not None:
                    config.chains = args.chains
                if args.thin is not None:
                    config.thin = args.thin

                try:
                    # Compile
                    compile_model(
                        posterior=posterior, backend=args.backend, mode=args.mode
                    )

                    # Run
                    with Monitor(name, logfile):
                        compare(
                            posterior=posterior,
                            backend=args.backend,
                            mode=args.mode,
                            config=config,
                        )
                except:
                    exc_type, exc_value, _ = sys.exc_info()
                    err = " ".join(traceback.format_exception_only(exc_type, exc_value))
                    err = re.sub(r"[\n\r\",]", " ", err)[:150] + "..."
                    logger.error(f"Failed {name} with {err}")
                    print(f'{name},,error,"{err}"', file=logfile, flush=True)
