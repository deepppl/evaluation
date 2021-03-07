import logging, datetime, os, sys, traceback, re, time, argparse, random
import numpyro
from utils import (
    run_stan_model,
    run_pyro_model,
    Config,
    parse_config,
    compile_model,
    golds,
    get_posterior,
)

logger = logging.getLogger(__name__)


def run(*, posterior, backend, mode, config, logfile):
    """
    Compare gold standard with model.
    """
    logger.info(f"Processing {posterior.name}")
    start = time.perf_counter()
    if backend == "stan":
        _ = run_stan_model(posterior=posterior, config=config)
        duration = time.perf_counter() - start
    else:
        _ = run_pyro_model(
            posterior=posterior, backend=backend, mode=mode, config=config
        )
        duration = time.perf_counter() - start
    print(f"{name},{duration}", file=logfile, flush=True)


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
    parser.add_argument("--runs", type=int, default=5, help="number of runs")

    parser.add_argument(
        "--scaled",
        help="Run scaled down experiment (iterations = 10, warmups = 10, chains = 1, thin = 1, runs = 1)",
        action="store_true",
    )

    # Override posteriorDB configs
    parser.add_argument("--iterations", type=int, help="number of iterations")
    parser.add_argument("--warmups", type=int, help="warmups steps")
    parser.add_argument("--chains", type=int, help="number of chains")
    parser.add_argument("--thin", type=int, help="thinning factor")

    args = parser.parse_args()

    if args.scaled:
        args.runs = 1

    logging.basicConfig(level=logging.INFO)

    numpyro.set_host_device_count(20)

    if not os.path.exists("logs"):
        os.makedirs("logs")

    for i in range(args.runs):
        today = datetime.datetime.now()
        logpath = f"logs/duration_{args.backend}"
        if args.backend != "stan":
            logpath += f"_{args.mode}"
        logpath += f"_{i}_{today.strftime('%y%m%d_%H%M%S')}.csv"

        with open(logpath, "a") as logfile:
            print(",time,exception", file=logfile, flush=True)
            for name in (n for n in golds):
                # Configurations
                posterior = get_posterior(name)
                config = parse_config(posterior)
                if args.scaled:
                    config.iterations = 10
                    config.warmups = 10
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

                config.seed = random.randint(0, 10000)
                print(f"Seed: {config.seed}")

                try:
                    # Compile
                    compile_model(
                        posterior=posterior, backend=args.backend, mode=args.mode
                    )

                    # Run
                    run(
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
                    print(f'{name},,"{err}"', file=logfile, flush=True)
