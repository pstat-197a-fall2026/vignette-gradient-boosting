# PSTAT 197A: Gradient Boosting Vignette
# XGBoost Regression Model

library(tidyverse)
library(xgboost)

# Load prepared data and evaluation metrics
source("scripts/model-preparation.R")
source("scripts/evaluation-metrics.R")

set.seed(197)

# Convert training data into DMatrix format
dtrain <- xgb.DMatrix(
  data = X_train,
  label = y_train,
  missing = NA
)

# Convert validation data into DMatrix format
dvalidation <- xgb.DMatrix(
  data = X_validation,
  label = y_validation,
  missing = NA
)

# Specify initial XGBoost parameters
xgb_parameters <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  tree_method = "hist",
  nthread = 4
)

# Specify datasets to monitor during training
evaluation_sets <- list(
  train = dtrain,
  validation = dvalidation
)

# Train XGBoost regression model
xgb_model <- xgb.train(
  params = xgb_parameters,
  data = dtrain,
  nrounds = 300,
  evals = evaluation_sets,
  verbose = 1,
  print_every_n = 25
)

# Generate predictions
train_pred_xgb <- predict(xgb_model, dtrain)

validation_pred_xgb <- predict(
  xgb_model,
  dvalidation
)

# Calculate evaluation metrics
xgb_results <- bind_rows(
  
  evaluate_model(y_train, train_pred_xgb) %>%
    mutate(Dataset = "Training"),
  
  evaluate_model(y_validation, validation_pred_xgb) %>%
    mutate(Dataset = "Validation")
  
) %>%
  mutate(Model = "XGBoost") %>%
  select(Model, Dataset, everything())

print(xgb_results)

# Extract performance at each boosting round
learning_curve <- as_tibble(
  attr(xgb_model, "evaluation_log")
)

print(head(learning_curve))
print(dim(learning_curve))
print(names(learning_curve))
print(head(learning_curve))

# Convert to long format for plotting
learning_curve_long <- learning_curve %>%
  pivot_longer(
    cols = c(train_rmse, validation_rmse),
    names_to = "Dataset",
    values_to = "RMSE"
  ) %>%
  mutate(
    Dataset = recode(
      Dataset,
      train_rmse = "Training",
      validation_rmse = "Validation"
    )
  )

# Plot training and validation performance
p_learning <- ggplot(
  learning_curve_long,
  aes(x = iter, y = RMSE, color = Dataset)
) +
  geom_line(linewidth = 1) +
  scale_color_manual(
    values = c(
      "Training" = "cyan3",
      "Validation" = "green3"
    )
  ) +
  scale_y_continuous(
    labels = scales::label_dollar(prefix = "£")
  ) +
  theme_minimal() +
  labs(
    title = "XGBoost Learning Curves",
    subtitle = "Training and validation RMSE across boosting rounds",
    x = "Boosting round",
    y = "RMSE"
  )

print(p_learning)

ggsave(
  "img/fig-xgboost-learning-curve.png",
  plot = p_learning,
  width = 8,
  height = 5,
  dpi = 300
)
# Create validation predictions data frame
xgb_predictions <- tibble(
  actual = y_validation,
  predicted = validation_pred_xgb,
  residual = y_validation - validation_pred_xgb
)

# Use reproducible sample for visualization
set.seed(199)

xgb_plot_data <- xgb_predictions %>%
  slice_sample(n = 5000)

# Plot actual versus predicted prices
p_xgb <- ggplot(
  xgb_plot_data,
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
    title = "XGBoost: Actual vs. Predicted Prices",
    subtitle = "Validation dataset",
    x = "Actual price",
    y = "Predicted price"
  )

print(p_xgb)

ggsave(
  "img/fig-xgboost-predictions.png",
  plot = p_xgb,
  width = 8,
  height = 6,
  dpi = 300
)

# Extract feature importance
importance <- xgb.importance(
  model = xgb_model
)

print(head(importance, 15))

# Plot the 15 most important encoded features
p_importance <- importance %>%
  slice_max(
    order_by = Gain,
    n = 15,
    with_ties = FALSE
  ) %>%
  ggplot(
    aes(
      x = reorder(Feature, Gain),
      y = Gain
    )
  ) +
  geom_col(fill = "cyan3") +
  coord_flip() +
  scale_y_continuous(
    labels = scales::label_percent()
  ) +
  theme_minimal() +
  labs(
    title = "XGBoost Feature Importance",
    subtitle = "Top 15 features ranked by gain",
    x = "Feature",
    y = "Relative gain"
  )

print(p_importance)

ggsave(
  "img/fig-xgboost-importance.png",
  plot = p_importance,
  width = 8,
  height = 6,
  dpi = 300
)