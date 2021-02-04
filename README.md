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
Let start with the simple eight schools example from Gelman et al (Bayesian Data Analysis: Sec. 5.5, 2003).
The file `schools.stan` contains the Stan code of the model.
To compile this example with both backend:
```
stanc --numpyro --o schools_numpyro.py schools.stan
stanc --pyro --o schools_pyro.py schools.stan
```
The compiled code is in the files `schools_numpyro.py` and `schools_pyro.py`. 


### Inference

The compiler generates up to 6 functions:
- `convert_inputs`: convert a dictionary of inputs to the correct names and type
- `transformed_data` (optional): proprocess the data
- `model`: the probabilistic model
- `guide` (optional): the guide for variational inference
- `generated_quantities` (optional): generate one sample of the generated quantities
- `map_generated_quantities` (optional): generated multiple samples of the generated quantities

You can then use these functions to run (Num)Pyro inference algorithms.
On the simple coin example:

```python
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import print_summary
import schools_numpyro as schools
import jax.random

data = {
    "J": 8,
    "y": [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
    "sigma": [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
}

mcmc = MCMC(NUTS(schools.model), 100, 100)
data = schools.convert_inputs(data)
# inputs = schools.transformed_data(data)  # Not needed for this example
mcmc.run(jax.random.PRNGKey(0), **data)
samples = mcmc.get_samples()
gen = schools.map_generated_quantities(mcmc.get_samples(), **data)
samples.update(gen)
print_summary(samples, group_by_chain=False)
```

Alternatively, we provide a simplified python interface which compiles the stan files and run the inference.

```python
from stannumpyro.dppl import NumPyroModel
import jax.random

data = {
    "J": 8,
    "y": [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
    "sigma": [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
}

numpyro_model = NumPyroModel("schools.stan")
mcmc = numpyro_model.mcmc(samples=100, warmups=100)
mcmc.run(jax.random.PRNGKey(0), data)
print(mcmc.summary())
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
