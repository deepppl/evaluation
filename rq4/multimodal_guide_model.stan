parameters {
  real cluster;
  real theta;
}
model {
  real mu;
  cluster ~ normal(0.0, 1.0);
  if (cluster > 0.0) {
    mu = 20.0;
  } else {
    mu = 0.0;
  }
  theta ~ normal(mu, 1.0);
}
guide parameters {
  real mu0;
  real mu1;
  real<lower=0> sigma0;
  real<lower=0> sigma1;
}
guide {
  cluster ~ normal(0, 1);
  if (cluster > 0) {
    theta ~ normal(mu0, sigma0);
  } else {
    theta ~ normal(mu1, sigma1);
  }
}
