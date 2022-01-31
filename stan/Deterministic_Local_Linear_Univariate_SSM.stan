// Deterministic Local Level Linear State-Space Model in Stan
data {
  int<lower = 1> N; // Number of Observations in the Time Series
  vector[N] y; // The Response Vector
  real sigma_y; // Standard Deviation of the Response Vector
}

parameters {
  real mu; // Level Component
  real<lower = 0> sigma; // Disturbance Component
}

model {
  // Priors
  target += normal_lpdf(mu | 0, 3);
  target += normal_lpdf(sigma | 0, sigma_y);
  
  // Gaussian Likelihood
  target += normal_lpdf(y | mu, sigma);
}

generated quantities {
  vector[N] log_lik; // Pointwise Log Likelihood
  vector[N] predictions; // Model Predictions

  // Calculate the Pointwise Log Likelihood
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | mu, sigma);
  }
  
  for (n in 1:N) {
    predictions[n] = normal_rng(mu, sigma);
  }
}
