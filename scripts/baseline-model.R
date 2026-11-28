# PSTAT 197A: Gradient Boosting Vignette
# Baseline Linear Regression Model

library(tidyverse)

# Load prepared data and evaluation function
source("scripts/model-preparation.R")
source("scripts/evaluation-metrics.R")

# Add intercept column
X_train_lm <- cbind(Intercept = 1, X_train)
X_validation_lm <- cbind(Intercept = 1, X_validation)

X_test_lm <- cbind(Intercept = 1, X_test)

# Fit multiple linear regression
baseline_model <- lm.fit(
  x = X_train_lm,
  y = y_train
)

# Extract estimated coefficients
baseline_coefs <- baseline_model$coefficients

# Handle redundant predictors if present
cat(
  "Redundant coefficients:",
  sum(is.na(baseline_coefs)),
  "\n"
)

baseline_coefs[is.na(baseline_coefs)] <- 0

# Generate training predictions
train_pred_lm <- as.vector(
  X_train_lm %*% baseline_coefs
)

# Generate validation predictions
validation_pred_lm <- as.vector(
  X_validation_lm %*% baseline_coefs
)

# Generate test predictions
test_pred_lm <- as.vector(
  X_test_lm %*% baseline_coefs
)

# Evaluate performance
baseline_results <- bind_rows(
  
  evaluate_model(y_train, train_pred_lm) %>%
    mutate(Dataset = "Training"),
  
  evaluate_model(y_validation, validation_pred_lm) %>%
    mutate(Dataset = "Validation"),
  
  evaluate_model(y_test, test_pred_lm) %>%
    mutate(Dataset = "Test")
  
) %>%
  mutate(Model = "Linear Regression") %>%
  select(Model, Dataset, everything())

print(baseline_results)

# Create data frame of validation predictions
baseline_predictions <- tibble(
  actual = y_validation,
  predicted = validation_pred_lm,
  residual = y_validation - validation_pred_lm
)

# Set reproducible visualization sample
set.seed(199)

baseline_plot_data <- baseline_predictions %>%
  slice_sample(n = 5000)

# Actual versus predicted prices
p_baseline <- ggplot(
  baseline_plot_data,
  aes(x = actual, y = predicted)
) +
  geom_point(
    alpha = 0.2,
    color = "cyan3"
  ) +
  geom_abline(
    intercept = 0,
    slope = 1,
    color = "green3",
    linetype = "dashed",
    linewidth = 1
  ) +
  scale_x_continuous(
    labels = scales::label_dollar(prefix = "£")
  ) +
  scale_y_continuous(
    labels = scales::label_dollar(prefix = "£")
  ) +
  coord_equal() +
  theme_minimal() +
  labs(
    title = "Linear Regression: Actual vs. Predicted Prices",
    subtitle = "Validation dataset",
    x = "Actual price",
    y = "Predicted price"
  )

print(p_baseline)

ggsave(
  "img/fig-baseline-predictions.png",
  plot = p_baseline,
  width = 8,
  height = 6,
  dpi = 300
)

print(baseline_results)


write_csv(
  baseline_results,
  "data/processed/baseline-results.csv"
)