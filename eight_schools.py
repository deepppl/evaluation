import os
import sys
from posteriordb import PosteriorDatabase

sys.path.append("stanc3")
from runtimes.dppl import PyroModel, NumpyroModel

pdb_path = os.path.abspath("posteriordb/posterior_database")
my_pdb = PosteriorDatabase(pdb_path)
posterior = my_pdb.posterior("eight_schools-eight_schools_centered")
stanfile = posterior.model.code_file_path("stan")
data = posterior.data

# model = PyroModel(stanfile, recompile=True, mode="mixed", compiler=["stanc.exe"])
model = NumpyroModel(stanfile, recompile=True, mode="mixed", compiler=["stanc.exe"])
mcmc = model.mcmc(
    samples = 100,
    warmups = 10, 
    chains=1,
    thin=2,
)

inputs = model.module.convert_inputs(data.values())
mcmc.run(**inputs)
print(mcmc.summary())