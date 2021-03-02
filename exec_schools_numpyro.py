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
