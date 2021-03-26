import logging, datetime, os, sys, traceback, re, argparse
import numpyro
from utils import (
    gold_summary,
    run_stan_model,
    run_pyro_model,
    Config,
    parse_config,
    compile_model,
    golds,
    get_posterior,
)

logger = logging.getLogger(__name__)


class ComparisonError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def compare(*, posterior, backend, mode, config, logfile):
    """
    Compare gold standard with model.
    """
    logger.info(f"Processing {posterior.name}")
    sg = gold_summary(posterior)
    if backend == "stan":
        fit = run_stan_model(posterior=posterior, config=config)
        summary = fit.summary()
        summary = summary[~summary.index.str.endswith("__")]
        summary = summary.rename(
            columns={"Mean": "mean", "StdDev": "std", "N_Eff": "n_eff"}
        )
        sm = summary[["mean", "std", "n_eff"]]
    else:
        mcmc = run_pyro_model(
            posterior=posterior, backend=backend, mode=mode, config=config
        )
        sm = mcmc.summary()
    sm = sm[["mean", "std", "n_eff"]]
    sm["err"] = abs(sm["mean"] - sg["mean"])
    sm["rel_err"] = sm["err"] / sg["std"]
    if len(sm.dropna()) != len(sg):
        raise RuntimeError("Missing parameter")
    # perf_cmdstan condition: err > 0.0001 and (err / stdev) > 0.3
    comp = sm[(sm["err"] > 0.0001) & (sm["rel_err"] > 0.3)].dropna()
    if not comp.empty:
        logger.error(f"Failed {posterior.name}")
        print(f"{name},mismatch,{sm['n_eff'].mean()}", file=logfile, flush=True)
    else:
        logger.info(f"Success {posterior.name}")
        print(f"{name},success,{sm['n_eff'].mean()}", file=logfile, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run accuracy experiment on PosteriorDB models."
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

    parser.add_argument(
        "--scaled",
        help="Run scaled down experiment (iterations = 100, warmups = 100, chains = 1, thin = 1)",
        action="store_true",
    )

    parser.add_argument("--posteriors", nargs="+", help="select the examples to execute")

    # Override posteriorDB configs
    parser.add_argument("--iterations", type=int, help="number of iterations")
    parser.add_argument("--warmups", type=int, help="warmups steps")
    parser.add_argument("--chains", type=int, help="number of chains")
    parser.add_argument("--thin", type=int, help="thinning factor")

    args = parser.parse_args()

    if args.posteriors:
        golds = args.posteriors

    logging.basicConfig(level=logging.INFO)

    numpyro.set_host_device_count(20)

    if not os.path.exists("logs"):
        os.makedirs("logs")

    today = datetime.datetime.now()
    logpath = f"logs/status_{args.backend}"
    if args.backend != "stan":
        logpath += f"_{args.mode}"
    logpath += f"_{today.strftime('%y%m%d_%H%M%S')}.csv"
    with open(logpath, "a") as logfile:
        print(",status,n_eff,exception", file=logfile, flush=True)
        for name in (n for n in golds):
            # Configurations
            posterior = get_posterior(name)
            config = parse_config(posterior)
            if args.scaled:
                config.iterations = 100
                config.warmups = 100
                config.chains = 1
                config.thin = 1
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
                compile_model(posterior=posterior, backend=args.backend, mode=args.mode)
                # Run and Compare
                compare(
                    posterior=posterior,
                    backend=args.backend,
                    mode=args.mode,
                    config=config,
                    logfile=logfile,
                )
            except:
                exc_type, exc_value, _ = sys.exc_info()
                err = " ".join(traceback.format_exception_only(exc_type, exc_value))
                err = re.sub(r"[\n\r\",]", " ", err)[:150] + "..."
                logger.error(f"Failed {name} with {err}")
                print(f'{name},error,,"{err}"', file=logfile, flush=True)
