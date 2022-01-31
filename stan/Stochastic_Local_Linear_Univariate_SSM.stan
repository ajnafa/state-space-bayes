// Stochastic Local Level Linear State-Space Model in Stan
data {
  int<lower = 1> N; // Number of Observations in the Time Series
  vector[N] y; // The Response Vector
  real sigma_y; // Standard Deviation of the Response Vector
}

parameters {
  vector[N] mu; // Stochastic Level Component
  real<lower = 0> sigma_level; // Level Error Term
  real<lower = 0> sigma_irreg; // Observation Error Term
}

transformed parameters {
  vector[N] y_hat;
  for(n in 1:N) {
    y_hat[n] = mu[n];
  }
}

model {
  // Priors
  target += normal_lpdf(mu | 0, 10);
  target += normal_lpdf(sigma_level | 0, sigma_y);
  target += normal_lpdf(sigma_irreg | 0, sigma_y);
  
  // Stochastic Level
  for(n in 2:N) {
    mu[n] ~ normal(mu[n - 1], sigma_level);
  }
  // Gaussian Likelihood
  for(n in 1:N) {
    y[n] ~ normal(y_hat[n], sigma_irreg);
  }
}

generated quantities {
  vector[N] log_lik; // Pointwise Log Likelihood
  vector[N] predictions; // Model Predictions
  
  // Generate Model Predictions
  for (n in 1:N) {
    predictions[n] = normal_rng(y_hat[n], sigma_irreg);
  }

  // Calculate the Pointwise Log Likelihood for the Observation-Level Model
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | y_hat[n], sigma_irreg);
  }
}
