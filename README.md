# DeepStan Evaluation: the Stan++ to (Num)Pyro compiler

The fork of Stanc3 that compiles programs to (Num)Pyro is available at https://github.com/deepppl/stanc3.

## Getting Started

You need to install the following dependencies:
- [opam](https://opam.ocaml.org/): the OCaml package manager
- [bazel](https://bazel.build/): required by tensorflow-probability

Stanc requires version 4.07.0 of OCaml which can be installed with:
```
opam switch create 4.07.0
opam switch 4.07.0
```

Then simply run the following command to install all the dependencies, including the compiler.
```
make init
```

### Dockerfile

We also provide a dockerfile to setup an environment with all the dependencies.
Build the image with (you might need to increase available memory in docker preferences):
```
make docker-build
```

Run with:
```
make docker-run
```

You can also follow the instruction of the dockerfile to install everything locally.

------------------------------------------------------------------

## Experiments

To evaluate our compilation scheme from Stan to Pyro and Numpyro we consider the following questions:
- RQ1: Can we compile and run all Stan models?
- RQ2: What is the impact of the compilation on accuracy?
- RQ3: What is the impact of the compilation on speed?

Then to evaluate DeepStan extensions, we consider two additional questions
- RQ4: Are explicit variational guides useful?
- RQ5: For deep probabilistic models, how does our extended Stan compare to hand-written Pyro code?

We assume that the current working directory is `evaluation`.

### RQ1

To compile all the examples of `example-models` from https://github.com/stan-dev/example-models, you can use the bash script `test_example-models.sh`.
The script expects two arguments: the backend (`pyro` or `numpyro`), and the compilation mode (`generative`, `comprehensive`, or `mixed`).

```
cd rq1
./test_example-models.sh pyro comprehensive
```

This command generates a file named `logs/pyro-comprehensive.csv` containing the name of the compiled examples and the exit code:
`0` meaning success,
`1` meaning semantics error raised from stanc3,
and `2` meaning compilation error due to the new backend.
The summary of the results is printed on the standard output and add it to the file `logs/summay.log`.

To test the compilation and inference, we use the models and data of [PosteriorDB](https://github.com/stan-dev/posteriordb) that are available in the directory `posteriordb`.
The Python the script `test_posteriordb.py` compiles and executes one iteration of the inference on all the examples of `posteriordb`.
The script is parameterized by the backend and the compilation scheme.
For example it can run with the numpyro backend and the comprehensive compilation scheme as follows:

```
python test_posteriordb.py  --backend numpyro --mode comprehensive
```

This will generate a csv file `logs/YYMMDD_HHMM_numpyro_comprehensive.csv` containing the exit code of each experiments and add a summary in `logs/summay.log`.

The summary of all the experiments can be display with:

```
cat logs/summay.log
```

### RQ2-RQ3

To compare accuracy of our backends with Stan, you can use the `test_accuracy.py` script.

```
cd rq2-3
python test_accuracy.py --help

usage: test_accuracy.py [-h] --backend BACKEND [--mode MODE] [--scaled]
                        [--iterations ITERATIONS] [--warmups WARMUPS]
                        [--chains CHAINS] [--thin THIN]

Run accuracy experiment on PosteriorDB models.

optional arguments:
  -h, --help            show this help message and exit
  --backend BACKEND     inference backend (pyro, numpyro, or stan)
  --mode MODE           compilation mode (generative, comprehensive, mixed)
  --scaled              Run scaled down experiment (iterations = 100, warmups
                        = 100, chains = 1, thin = 1)
  --posteriors POSTERIORS [POSTERIORS ...]
                        select the examples to execute
  --iterations ITERATIONS
                        number of iterations
  --warmups WARMUPS     warmups steps
  --chains CHAINS       number of chains
  --thin THIN           thinning factor
```

For instance, to test the NumPyro backend with the comprehensive translation using PosteriorDB configurations on all examples that have a reference, the command is:

```
python test_accuracy.py --backend numpyro --mode comprehensive
```

This will generate a csv file `status_numpyro_comprehensive_YYMMDD_HHMMSS.csv` containing a summary of the experiments.

To run the reference Stan implementation:

```
python test_accuracy.py --backend stan
```

To compare the speed of our backends with Stan, you can use the `test_speed.py` script.

```
python test_speed.py --help

usage: test_speed.py [-h] --backend BACKEND [--mode MODE] [--runs RUNS]
                     [--scaled] [--iterations ITERATIONS] [--warmups WARMUPS]
                     [--chains CHAINS] [--thin THIN]

Run experiments on PosteriorDB models.

optional arguments:
  -h, --help            show this help message and exit
  --backend BACKEND     inference backend (pyro, numpyro, or stan)
  --mode MODE           compilation mode (generative, comprehensive, mixed)
  --runs RUNS           number of runs
  --scaled              Run scaled down experiment (iterations = 10, warmups =
                        10, chains = 1, thin = 1)
  --posteriors POSTERIORS [POSTERIORS ...]
                        select the examples to execute
  --iterations ITERATIONS
                        number of iterations
  --warmups WARMUPS     warmups steps
  --chains CHAINS       number of chains
  --thin THIN           thinning factor
```

For instance, to launch 5 runs with the NumPyro backend and the comprehensive translation using PosteriorDB configurations except for the seed which is picked randomly at each run, the command is:

```
python test_speed.py --backend numpyro --mode comprehensive --runs 5
```

This will generate 5 csv files (one per run) `duration_numpyro_comprehensive_i_YYMMDD_HHMMSS.csv` containing a summary of the experiments.


:warning: Experiments with the pyro backend take a very long time (e.g., >60h for one example).

:warning: A keyboard interrupt only stops one example.

A scaled down version of the experiments can be run for both `test_accuracy.py` and `test_speed.py` with the `--scaled` option.

```
python test_accuracy.py --backend numpyro --mode comprehensive --scaled
python test_speed.py --backend numpyro --mode comprehensive --scaled
```

The option `--posterior` a select the examples to execute for both `test_accuracy.py` and `test_speed.py`.
The example must be one of the [posterior with reference draws](https://github.com/stan-dev/posteriordb/tree/master/posterior_database/reference_posteriors/summary_statistics/mean/info).
E.g.,

```
python test_accuracy.py --backend numpyro --posterior nes1976-nes earnings-earn_height
python test_speed.py --backend numpyro --posterior nes1976-nes earnings-earn_height
```


The script `results_analysis.py` can be used to analyze the results.
Option `--logdir` specifies the repository containing the log files (default `./logs`).

```
python results_analysis.py
```

### RQ4

The script `multimodal.py` regenerates the plots of Figure 10 in pdf format to compare Stan NUTS, Stan ADVI, DeepStan NUTS, and DeepStan VI with explicit guides.

```
cd rq4
python multimodal.py
```

This will generate the files  `deepstan-vs-deepstansvi.pdf`, `deepstansvi-vs-stanadvi.pdf`, and `stan-vs-deepstansvi.pdf`.
The stan code is in the two files `multimodal_model.stan` and `multimodal_guide_model.stan`.

### RQ5

The last experiments are on deep probabilistic models and compare our Stan extension with hand-written Pyro code.
The variational autoencoder (VAE) example is in `vae_model.stan` and the hand-written Pyro version with the comparison code is in `vae.py`.

```
cd rq5
python vae.py
```

The script executes both versions and print the precision, recall, and f1 score on each on the standard output.

The MLP in DeepStan is in `mlp_model.py` and the hand-written Pyro version with the comparison code is in `mlp.py`.

```
python mlp.py
```

This script prints the result of the comparison on the standard output and produce a file `pyro-vs-deepstan.pdf` showing the distributions of accuracy of the sampled MLPs.
