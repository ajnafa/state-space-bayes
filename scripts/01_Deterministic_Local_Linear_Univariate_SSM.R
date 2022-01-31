#------------State Space Models in R and Stan: Local Level Models---------------
#-Author: A. Jordan Nafa-----------------------------Created: January 29, 2022-#
#-R Version: 4.1.0-----------------------------------Revised: January 30, 2022-#

# Set Project Options----
options(
  digits = 4, # Significant figures output
  scipen = 999, # Disable scientific notation
  repos = getOption("repos")["CRAN"],
  mc.cores = 12L
)

# Load the necessary libraries----
pacman::p_load(
  "tidyverse",
  "data.table",
  "dtplyr",
  "cmdstanr",
  "tidybayes",
  "latex2exp"
)

# Directory for the model files
models_dir <- "models/Deterministic Local Linear SSM/"

# Load the helper functions
source("scripts/00_Helper_Functions.R")

#------------------------------------------------------------------------------#
#------------------------------Data Pre-Processing------------------------------
#------------------------------------------------------------------------------#

# Load the V-Dem v. 11.1 Data
data(vdem, package = "vdemdata")

# Data Preparation
vdem_df <- vdem %>% 
  # Keep only observations for the United States
  filter(country_text_id == "USA") %>% 
  # Extract the liberal democracy index
  transmute(
    # Country Abbreviations
    country = country_text_id,
    # Observation Year
    time = year,
    # Liberal Democracy Index
    libdem = v2x_libdem * 100,
    # Liberal Democracy Index Measurement Noise
    libdem_sd = v2x_libdem_sd * 100,
    # Calculate log and delta
    across(
      libdem,
      list(
        delta = ~ .x - lag(.x, n = 1L),
        lagt1 = ~ lag(.x, n = 1L),
        log = ~ log(.x)
      ),
      .names = "{.col}_{.fn}"
    )
  ) %>% 
  # Drop NA values
  drop_na()

#------------------------------------------------------------------------------#
#--------------Deterministic Local Level Linear State-Space Model---------------
#------------------------------------------------------------------------------#

# Load the Stan Model File
DLL_SSM <- "stan/Deterministic_Local_Linear_Univariate_SSM.stan"

# Compile the Stan model
dll_ssm_mod <- cmdstan_model(
  DLL_SSM, 
  dir = models_dir,
  force_recompile = TRUE
  )

# Print the model code
str_split(dll_ssm_mod$code(), pattern = ";", simplify = T)

# Prepare the Data to Pass to Stan
stan_data <- list(
  N = nrow(vdem_df),
  y = vdem_df$libdem_delta,
  sigma_y = sd(vdem_df$libdem_delta)*2.5
)

# Fit the Stan Model
dll_ssm_fit <- dll_ssm_mod$sample(
  data = stan_data,
  seed = 123456,
  refresh = 50,
  output_dir = models_dir,
  sig_figs = 5,
  parallel_chains = 6,
  chains = 6,
  iter_warmup = 2000,
  iter_sampling = 2000,
  max_treedepth = 11
)

# Write the model object to an RDS file
dll_ssm_fit$save_object(file = str_c(models_dir, "dll_ssm_fit.rds"))

# Extract the Posterior Draws
draws <- dll_ssm_fit$draws(variables = c("mu", "sigma", "predictions"), format = "df")

# Print a Summary of the Posterior Draws
(summ_draws <- summarise_draws(draws))

#------------------------------------------------------------------------------#
#--------------------------Posterior Predictions--------------------------------
#------------------------------------------------------------------------------#

# Create a data frame of the generated posterior predictions
post_preds <- draws %>% 
  spread_draws(predictions[i]) %>% 
  mutate(time = i + 1789) %>%  # Convert index to year
  left_join(vdem_df, by = c("time")) %>% 
  mutate(y_pred = libdem_delta - predictions)

# Plot the predicted change in democracy over the previous year
struct_breaks <- ggplot(post_preds, aes(x = time, y = y_pred)) +
  # Difference between observed and predicted
  stat_gradientinterval(
    aes(slab_alpha = stat(pdf), fill = stat(y > 0)),
    fill_type = "gradient",
    point_interval = mean_qi,
    .width = c(0.68, 0.80)
  ) +
  # Set the fill parameter for each group
  scale_fill_manual(
    values = c("firebrick", "royalblue"),
    labels = c("Negative", "Positive")
    ) +
  # Add custom theme settings
  plot_theme(plot.margin = margin(5, 1, 3, 5, "mm")) +
  # Add labels to the plot
  labs(
    x = "Time",
    y = TeX(r'(Liberal Democracy $\, \Delta_{t}$)'),
    subtitle = "Figure 1.1 Democracy in America (1789 - 2020)"
  ) +
  # Adjust the breaks on the x axis
  scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) +
  # Adjust the breaks on the y axis
  scale_y_continuous(breaks = scales::pretty_breaks(n = 10)) +
  # Setting the parameters for the plot legend
  guides(fill = guide_legend(
    title = "Direction",
    override.aes = list(
      fill = c("firebrick", "royalblue"),
      size = 6
    )
  ),
  shape = "none"
  )

# Save the generated plot object as a .jpeg file
ggsave(
  filename = "Figure_1.1_Structural_Breaks.jpeg",
  plot = struct_breaks,
  device = "jpeg",
  path = models_dir,
  width = 20,
  height = 12,
  units = "in",
  dpi = "retina",
  type = "cairo"
)
