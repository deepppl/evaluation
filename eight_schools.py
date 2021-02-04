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
