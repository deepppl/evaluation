// model {
//   latent ~ Normal(zeros(nz), ones(nz), batch_size);
// }

// guide {
//   latent ~ Normal(mu, sigma, batch_size);
// }

networks {
  real[,] decoder(real[] x);
  real[,] encoder(int[,] x);
}
data {
  int<lower=0, upper=1> x[28, 28];
  int nz;
  int batch_size;
}
parameters {
  real latent[nz];
}
model {
  real loc_img[28, 28];
  latent ~ normal(0, 1);
  loc_img = decoder(latent);
  x ~ bernoulli(loc_img);
}
guide {
  real encoded[2, nz] = encoder(x);
  real mu[nz] = encoded[1];
  real sigma[nz] = encoded[2];
  latent ~ normal(mu, sigma);
}