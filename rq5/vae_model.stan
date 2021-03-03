networks {
  real[,] decoder(real[] x);
  real[,] encoder(int[,] x);
}
data {
  int nz;
  int<lower=0, upper=1> x[28, 28];
}
parameters {
  real z[nz];
}
model {
  real mu[28, 28];
  z ~ normal(0, 1);
  mu = decoder(z);
  x ~ bernoulli(mu);
}
guide {
  real encoded[2, nz] = encoder(x);
  real mu[nz] = encoded[1];
  real sigma[nz] = encoded[2];
  z ~ normal(mu, sigma);
}