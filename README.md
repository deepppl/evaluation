# DeepStan Evaluation: the Stan++ to (Num)Pyro compiler

## Getting Started

The evaluation is based on posteriordb which listed as a submodule.
First clone the depo with its submodule and install [posteriordb](https://github.com/stan-dev/posteriordb)
```
git submodule init && git submodule update
pip install ./posteriordb/python
```

You then need to install the fork of the stanc3 compiler and the runtime libraries.
Be careful that Stanc requires version 4.07.0 of OCaml.
Using [Opam](https://opam.ocaml.org/) (the OCaml package manager) and Pip:

```
opam switch 4.07.0
opam pin -k git git+https://github.com/deepppl/stanc3.git
pip install -r requirements.txt
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


### Compilation
The file `coin.stan` contains the Stan code for a simple biased coin model.
To compile this example with both backend:
```
stanc --pyro --o coin_pyro.py coin.stan
stanc --numpyro --o coin_numpyro.py coin.stan
```
The compiled code is in the files `coin_pyro.py` and `coin_numpyro.py`


### Inference
To run the inference on a posteriordb model (for example eight_schools stored in PosteriorDB):

```python
import os
from posteriordb import PosteriorDatabase
from stannumpyro.dppl import NumPyroModel
import jax.random

pdb_path = os.path.abspath("posteriordb/posterior_database")
my_pdb = PosteriorDatabase(pdb_path)
posterior = my_pdb.posterior("eight_schools-eight_schools_centered")
stanfile = posterior.model.code_file_path("stan")
data = posterior.data

model = NumPyroModel(stanfile)
mcmc = model.mcmc(
    samples=100,
    warmups=10,
    chains=1,
    thin=2,
)

inputs = model.module.convert_inputs(data.values())
mcmc.run(jax.random.PRNGKey(0), inputs)

print(mcmc.summary())
```

Run this example with:
```
python eight_schools.py
```

------------------------------------------------------------------

## Experiments

To evaluate our compilation scheme from Stan to Pyro and Numpyro we consider the following questions:
- RQ1: Can we compile and run all Stan models?
- RQ2: What is the impact of the compilation on accuracy?
- RQ3: What is the impact of the compilation on speed?

Then to evaluate DeepStan extensions, we consider two additional questions
- RQ4: Are explicit variational guides useful?
- RQ5: For deep probabilistic models, how does our extended Stan compare to hand-written Pyro code?

### RQ1

To compile all the examples of `example-models` from https://github.com/stan-dev/example-models, you can use the bash scrip `test_example-models.sh`:
```
cd rq1
./test_example-models.sh
```

This will generates files named `logs-$backend-$mode` where `$backend` is `pyro` or `numpyro`, and `$mode` is `generative`, `comprehensive`, or `mixed` containing the name of the compiled examples and the exit code:
`0` meaning success,
`1` meaning semantics error raised from stanc3,
and `2` meaning compilation error due to the new backend.
The summary of the results is printed on the standard output.

This directory also contains the Python the script `test_posteriordb.py` to compile and execute one iteration of the inference on all the examples of `posteriordb`.
The script is parameterized by the backend and the compilation scheme.
For example it can run with the numpyro backend and the comprehensive compilation scheme as follows:
```
python test_posteriordb.py  --backend numpyro --mode comprehensive
```

This will generate a csv file `YYMMDD_HHMM_numpyro_comprehensive.csv` containing the exit code of each experiments.

### RQ2-RQ3

To compare accuracy and speed of our backends compared to Stan you can use the `test_posteriordb.py` script.
For example to test the numpyro backend with the comprehensive translation on all `posteriordb` examples that have a reference:
```
cd rq2-3
python test_posteriordb.py --backend numpyro --mode comprehensive
```

This will generate a csv file `YYMMDD_HHMM_numpyro_comprehensive.csv` containing a summary of the experiments.
:warning: A keyboard interrupt only stops one example.

To run the reference Stan implementation:
```
python test_posteriordb.py --backend stan
```

The notebook `analysis.ipynb` can be used to analyse the results.

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
