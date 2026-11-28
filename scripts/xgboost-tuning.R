

library(tidyverse)
library(xgboost)

source("scripts/model-preparation.R")
source("scripts/evaluation-metrics.R")

set.seed(197)

dtrain <- xgb.DMatrix(
  data = X_train,
  label = y_train,
  missing = NA
)

dvalidation <- xgb.DMatrix(
  data = X_validation,
  label = y_validation,
  missing = NA
)

dtest <- xgb.DMatrix(
  data = X_test,
  label = y_test,
  missing = NA
)

# Specify datasets to monitor during training
evaluation_sets <- list(
  train = dtrain,
  validation = dvalidation
)

# Define hyperparameter tuning grid
# Focus on key gradient boosting controls:
#   eta              - learning rate (shrinkage per boosting round)
#   max_depth        - max depth of each tree (interaction complexity)
#   min_child_weight - min sum of instance weight in a leaf (regularization)
#   subsample        - row subsampling ratio per tree
#   colsample_bytree - feature subsampling ratio per tree
tuning_grid <- expand_grid(
  eta = c(0.01, 0.05, 0.1, 0.3),
  max_depth = c(4, 6, 8),
  min_child_weight = c(1, 5),
  subsample = 0.8,
  colsample_bytree = 0.8
)


# Run grid search with early stopping on validation RMSE
max_rounds <- 2500
early_stop <- 30

tuning_results <- vector("list", nrow(tuning_grid))

for (i in seq_len(nrow(tuning_grid))) {
  cfg <- tuning_grid[i, ]

  params <- list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    eta = cfg$eta,
    max_depth = cfg$max_depth,
    min_child_weight = cfg$min_child_weight,
    subsample = cfg$subsample,
    colsample_bytree = cfg$colsample_bytree,
    tree_method = "hist",
    nthread = 4
  )

  set.seed(197)
  fit <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = max_rounds,
    evals = evaluation_sets,
    early_stopping_rounds = early_stop,
    verbose = 0
  )

  eval_log <- as_tibble(attr(fit, "evaluation_log"))
  best_iter <- which.min(eval_log$validation_rmse)
  best_train_rmse <- eval_log$train_rmse[best_iter]
  best_val_rmse <- eval_log$validation_rmse[best_iter]

  val_pred <- predict(
    fit,
    dvalidation,
    iterationrange = c(1, best_iter)
  )
  val_metrics <- evaluate_model(y_validation, val_pred)

  tuning_results[[i]] <- cfg %>%
    mutate(
      best_iteration = best_iter,
      train_rmse = best_train_rmse,
      val_rmse = val_metrics$RMSE,
      val_mae = val_metrics$MAE,
      val_r2 = val_metrics$R2
    )

  cat(
    sprintf(
      "[%2d/%2d] eta=%.2f | depth=%d | mcw=%d | best_iter=%4d | val_RMSE=£%.2f | val_R2=%.4f\n",
      i, nrow(tuning_grid),
      cfg$eta, cfg$max_depth, cfg$min_child_weight,
      best_iter, val_metrics$RMSE, val_metrics$R2
    )
  )
}

# Combine and rank all tuning configurations
tuning_summary <- bind_rows(tuning_results) %>%
  arrange(val_rmse)

print(tuning_summary, n = Inf, width = Inf)

# Save tuning summary for vignette and slides
write_csv(
  tuning_summary,
  "data/processed/xgboost-tuning-results.csv"
)

# Extract best hyperparameter configuration
best_config <- tuning_summary %>%
  slice(1)

cat("\nBest hyperparameter configuration:\n")
print(best_config, width = Inf)

# Refit best tuned model to capture full learning curve and feature importance
best_params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = best_config$eta,
  max_depth = best_config$max_depth,
  min_child_weight = best_config$min_child_weight,
  subsample = best_config$subsample,
  colsample_bytree = best_config$colsample_bytree,
  tree_method = "hist",
  nthread = 4
)

set.seed(197)
xgb_tuned <- xgb.train(
  params = best_params,
  data = dtrain,
  nrounds = max_rounds,
  evals = evaluation_sets,
  early_stopping_rounds = early_stop,
  verbose = 1,
  print_every_n = 100
)

tuned_log <- as_tibble(attr(xgb_tuned, "evaluation_log"))
best_iter <- which.min(tuned_log$validation_rmse)

# Generate predictions at the optimal early-stopping iteration
train_pred_tuned <- predict(
  xgb_tuned,
  dtrain,
  iterationrange = c(1, best_iter)
)

validation_pred_tuned <- predict(
  xgb_tuned,
  dvalidation,
  iterationrange = c(1, best_iter)
)

test_pred_tuned <- predict(
  xgb_tuned,
  dtest,
  iterationrange = c(1, best_iter)
)

# Evaluate tuned model across the different sets
tuned_results <- bind_rows(
  evaluate_model(y_train, train_pred_tuned) %>%
    mutate(Dataset = "Training"),
  evaluate_model(y_validation, validation_pred_tuned) %>%
    mutate(Dataset = "Validation"),
  evaluate_model(y_test, test_pred_tuned) %>%
    mutate(Dataset = "Test")
) %>%
  mutate(Model = "XGBoost (Tuned)") %>%
  select(Model, Dataset, everything())

print(tuned_results)

# Hyperparameter tuning comparison
p_tuning <- tuning_summary %>%
  mutate(
    eta_label = paste0("eta = ", eta),
    mcw_label = factor(min_child_weight)
  ) %>%
  ggplot(
    aes(
      x = factor(max_depth),
      y = val_rmse,
      fill = mcw_label
    )
  ) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  facet_wrap(~ eta_label) +
  scale_fill_manual(
    values = c("1" = "cyan3", "5" = "green3"),
    name = "min_child_weight"
  ) +
  scale_y_continuous(
    labels = scales::label_dollar(prefix = "£")
  ) +
  theme_minimal(base_size = 12) +
  labs(
    title = "XGBoost Hyperparameter Tuning Results",
    subtitle = "Validation RMSE across learning rate, tree depth, and min_child_weight with early stopping",
    x = "Maximum tree depth",
    y = "Validation RMSE"
  )

ggsave(
  "img/fig-xgboost-tuning.png",
  plot = p_tuning,
  width = 9,
  height = 5,
  dpi = 300
)

# Plot 2: Tuned XGBoost learning curve with early stopping marker
tuned_log_long <- tuned_log %>%
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

p_tuned_learning <- ggplot(
  tuned_log_long,
  aes(x = iter, y = RMSE, color = Dataset)
) +
  geom_line(linewidth = 1) +
  geom_vline(
    xintercept = best_iter,
    linetype = "dashed",
    color = "red3",
    linewidth = 0.8
  ) +
  scale_color_manual(
    values = c(
      "Training" = "cyan3",
      "Validation" = "green3"
    )
  ) +
  scale_y_continuous(
    labels = scales::label_dollar(prefix = "£")
  ) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Tuned XGBoost Learning Curves with Early Stopping",
    subtitle = sprintf(
      "eta = %.2f, max_depth = %d, min_child_weight = %d | Best round = %d",
      best_config$eta, best_config$max_depth, best_config$min_child_weight, best_iter
    ),
    x = "Boosting round",
    y = "RMSE"
  )

ggsave(
  "img/fig-xgboost-tuned-learning-curve.png",
  plot = p_tuned_learning,
  width = 8,
  height = 5,
  dpi = 300
)

# Plot 3: Tuned XGBoost actual vs. predicted prices (validation set)
set.seed(199)
tuned_plot_data <- tibble(
  actual = y_validation,
  predicted = validation_pred_tuned
) %>%
  slice_sample(n = 5000)

p_tuned_pred <- ggplot(
  tuned_plot_data,
  aes(x = actual, y = predicted)
) +
  geom_point(alpha = 0.2, color = "cyan3") +
  geom_abline(
    intercept = 0,
    slope = 1,
    color = "green3",
    linetype = "dashed",
    linewidth = 1
  ) +
  scale_x_continuous(labels = scales::label_dollar(prefix = "£")) +
  scale_y_continuous(labels = scales::label_dollar(prefix = "£")) +
  coord_equal() +
  theme_minimal(base_size = 12) +
  labs(
    title = "Tuned XGBoost: Actual vs. Predicted Prices",
    subtitle = "Validation dataset (5,000 sampled vehicles)",
    x = "Actual price",
    y = "Predicted price"
  )

ggsave(
  "img/fig-xgboost-tuned-predictions.png",
  plot = p_tuned_pred,
  width = 8,
  height = 6,
  dpi = 300
)

cat("Hyperparameter tuning complete. Figures and summary CSV saved.\n")


print(tuned_results)

write_csv(
  tuned_results,
  "data/processed/tuned-xgboost-results.csv"
)