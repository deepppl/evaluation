import datetime
from dataclasses import dataclass, field
from posteriordb import PosteriorDatabase
import os, sys, argparse

from stanpyro.dppl import PyroModel
from stannumpyro.dppl import NumPyroModel
import jax.random

stanc = "stanc"
pdb_root = "../posteriordb"
pdb_path = os.path.join(pdb_root, "posterior_database")
my_pdb = PosteriorDatabase(pdb_path)


@dataclass
class Config:
    iterations: int = 1
    warmups: int = 0
    chains: int = 1
    thin: int = 1


def test(posterior, config, backend="pyro", mode="mixed"):
    model = posterior.model
    data = posterior.data
    stanfile = model.code_file_path("stan")
    try:
        if backend == "pyro":
            pyro_model = PyroModel(
                stanfile, mode=mode, compiler=[stanc], recompile=True
            )
        else:
            pyro_model = NumPyroModel(
                stanfile, mode=mode, compiler=[stanc], recompile=True
            )
    except Exception as e:
        return {
            "code": 1,
            "msg": f"compilation error ({posterior.name}): {model.name}",
            "exn": e,
        }
    try:
        mcmc = pyro_model.mcmc(
            config.iterations,
            warmups=config.warmups,
            chains=config.chains,
            thin=config.thin,
        )
        if backend == "pyro":
            mcmc.run(data.values())
        else:
            mcmc.run(jax.random.PRNGKey(0), data.values())
    except Exception as e:
        return {
            "code": 2,
            "msg": f"Inference error ({posterior.name}): {model.name}({data.name})",
            "exn": e,
        }
    return {"code": 0, "samples": mcmc.get_samples()}


def log(logfile, res):
    test = res["test"]
    code = res["code"]
    exn = res["exn"] if res["code"] != 0 else ""
    print(f"{test}, {code}, {exn}", file=logfile, flush=True)


def test_all(posteriors, backend, mode, config):
    if not os.path.exists("logs"):
        os.makedirs("logs")
    today = datetime.datetime.now()
    logpath = f"logs/{today.strftime('%y%m%d_%H%M')}_{args.backend}_{args.mode}.csv"
    success = 0
    compile_error = 0
    inference_error = 0
    with open(logpath, "a") as logfile:
        print("test,exit code,exn", file=logfile, flush=True)
        for name in posteriors:
            print(f"- Test {backend} {mode}: {name}")
            posterior = my_pdb.posterior(name)
            res = test(posterior, config, backend, mode)
            res["test"] = name
            if res["code"] == 0:
                success = success + 1
            elif res["code"] == 1:
                compile_error = compile_error + 1
            elif res["code"] == 2:
                inference_error = inference_error + 1
            print(
                f"success: {success}, compile errors: {compile_error}, inference errors: {inference_error}, total: {success + compile_error + inference_error}"
            )
            log(logfile, res)
    return {
        "success": success,
        "compile_error": compile_error,
        "inference_error": inference_error,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run experiments on PosteriorDB models."
    )
    parser.add_argument(
        "--backend",
        help="inference backend (pyro, numpyro)",
        required=True,
    )
    parser.add_argument(
        "--mode",
        help="compilation mode (generative, comprehensive, mixed)",
        default="mixed",
    )
    parser.add_argument("--posteriors", nargs="+", help="select the examples to execute")
    parser.add_argument("--iterations", type=int, help="number of iterations")
    parser.add_argument("--warmups", type=int, help="warmups steps")
    parser.add_argument("--chains", type=int, help="number of chains")
    parser.add_argument("--thin", type=int, help="thinning factor")

    args = parser.parse_args()

    config = Config()
    if args.iterations is not None:
        config.iterations = args.iterations
    if args.warmups is not None:
        config.warmups = args.warmups
    if args.chains is not None:
        config.chains = args.chains
    if args.thin is not None:
        config.thin = args.thin

    posteriors = my_pdb.posterior_names()
    if args.posteriors:
        assert all(p in posteriors for p in args.posteriors), "Bad posterior name"
        posteriors = args.posteriors

    res = test_all(posteriors, args.backend, args.mode, config)

    print("Summary")
    print("-------")
    print(f"{args.backend} {args.mode}: {res}")
    with open("logs/summary.log", "a") as file:
        print(f"posteriordb {args.backend}-{args.mode}: {res}", file=file, flush=True)
