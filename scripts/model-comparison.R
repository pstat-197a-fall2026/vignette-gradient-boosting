
# PSTAT 197A: Gradient Boosting Vignette
# Model Comparison

library(tidyverse)

# Load model results

baseline_results <- read_csv(
  "data/processed/baseline-results.csv",
  show_col_types = FALSE
)

xgb_results <- read_csv(
  "data/processed/xgboost-results.csv",
  show_col_types = FALSE
)

tuned_results <- read_csv(
  "data/processed/tuned-xgboost-results.csv",
  show_col_types = FALSE
)

lgb_results <- read_csv(
  "data/processed/lightgbm-results.csv",
  show_col_types = FALSE
)


# Combine results

model_comparison <- bind_rows(
  baseline_results,
  xgb_results,
  tuned_results,
  lgb_results
) %>%
  mutate(
    Dataset = str_to_title(Dataset),
    Model = recode(
      Model,
      "XGBoost (Tuned)" = "Tuned XGBoost"
    ),
    Model = factor(
      Model,
      levels = c(
        "Linear Regression",
        "XGBoost",
        "Tuned XGBoost",
        "LightGBM"
      )
    ),
    Dataset = factor(
      Dataset,
      levels = c("Training", "Validation", "Test")
    )
  ) %>%
  arrange(Model, Dataset)

print(model_comparison)

write_csv(
  model_comparison,
  "data/processed/model-comparison-results.csv"
)


# Test set performance

test_comparison <- model_comparison %>%
  filter(Dataset == "Test") %>%
  arrange(RMSE) %>%
  mutate(
    RMSE_Rank = row_number(),
    MAE_Rank = rank(MAE),
    R2_Rank = rank(-R2)
  )

print(test_comparison)

write_csv(
  test_comparison,
  "data/processed/test-performance-comparison.csv"
)


# Test RMSE

p_test_rmse <- ggplot(
  test_comparison,
  aes(x = reorder(Model, RMSE), y = RMSE)
) +
  geom_col() +
  geom_text(
    aes(label = round(RMSE, 0)),
    hjust = -0.15
  ) +
  coord_flip() +
  expand_limits(y = max(test_comparison$RMSE) * 1.15) +
  labs(
    title = "Test RMSE Across Models",
    subtitle = "Lower values indicate more accurate predictions",
    x = NULL,
    y = "RMSE (£)"
  ) +
  theme_minimal()

print(p_test_rmse)

ggsave(
  "img/fig-test-rmse-comparison.png",
  p_test_rmse,
  width = 8,
  height = 5,
  dpi = 300
)


# Test MAE

p_test_mae <- ggplot(
  test_comparison,
  aes(x = reorder(Model, MAE), y = MAE)
) +
  geom_col() +
  geom_text(
    aes(label = round(MAE, 0)),
    hjust = -0.15
  ) +
  coord_flip() +
  expand_limits(y = max(test_comparison$MAE) * 1.15) +
  labs(
    title = "Test MAE Across Models",
    subtitle = "Lower values indicate smaller average absolute prediction errors",
    x = NULL,
    y = "MAE (£)"
  ) +
  theme_minimal()

print(p_test_mae)

ggsave(
  "img/fig-test-mae-comparison.png",
  p_test_mae,
  width = 8,
  height = 5,
  dpi = 300
)


# Test R-squared

p_test_r2 <- ggplot(
  test_comparison,
  aes(x = reorder(Model, R2), y = R2)
) +
  geom_col() +
  geom_text(
    aes(label = round(R2, 3)),
    hjust = -0.15
  ) +
  coord_flip() +
  expand_limits(y = max(test_comparison$R2) * 1.08) +
  labs(
    title = "Test R² Across Models",
    subtitle = "Higher values indicate more variation explained",
    x = NULL,
    y = "R²"
  ) +
  theme_minimal()

print(p_test_r2)

ggsave(
  "img/fig-test-r2-comparison.png",
  p_test_r2,
  width = 8,
  height = 5,
  dpi = 300
)


# Combined test metric comparison

test_metrics_long <- test_comparison %>%
  select(Model, RMSE, MAE, R2) %>%
  pivot_longer(
    cols = c(RMSE, MAE, R2),
    names_to = "Metric",
    values_to = "Value"
  )

p_test_metrics <- ggplot(
  test_metrics_long,
  aes(x = reorder(Model, Value), y = Value)
) +
  geom_col() +
  coord_flip() +
  facet_wrap(
    ~ Metric,
    scales = "free_x"
  ) +
  labs(
    title = "Test Performance Across Models",
    x = NULL,
    y = "Metric Value"
  ) +
  theme_minimal()

print(p_test_metrics)

ggsave(
  "img/fig-test-metrics-comparison.png",
  p_test_metrics,
  width = 10,
  height = 6,
  dpi = 300
)


# Performance across datasets: RMSE

p_rmse_all <- ggplot(
  model_comparison,
  aes(x = Dataset, y = RMSE, group = Model, color = Model)
) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  labs(
    title = "RMSE Across Training, Validation, and Test Sets",
    x = NULL,
    y = "RMSE (£)",
    color = "Model"
  ) +
  theme_minimal()

print(p_rmse_all)

ggsave(
  "img/fig-rmse-all-datasets.png",
  p_rmse_all,
  width = 9,
  height = 6,
  dpi = 300
)


# Performance across datasets: R-squared

p_r2_all <- ggplot(
  model_comparison,
  aes(x = Dataset, y = R2, group = Model, color = Model)
) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  labs(
    title = "R² Across Training, Validation, and Test Sets",
    x = NULL,
    y = "R²",
    color = "Model"
  ) +
  theme_minimal()

print(p_r2_all)

ggsave(
  "img/fig-r2-all-datasets.png",
  p_r2_all,
  width = 9,
  height = 6,
  dpi = 300
)


# Generalization comparison

generalization_comparison <- model_comparison %>%
  select(Model, Dataset, RMSE) %>%
  pivot_wider(
    names_from = Dataset,
    values_from = RMSE
  ) %>%
  mutate(
    Validation_Gap = Validation - Training,
    Test_Gap = Test - Training,
    Test_Validation_Gap = Test - Validation
  ) %>%
  arrange(Test_Gap)

print(generalization_comparison)

write_csv(
  generalization_comparison,
  "data/processed/generalization-comparison.csv"
)


# Training vs. test RMSE

p_generalization <- generalization_comparison %>%
  select(Model, Training, Test) %>%
  pivot_longer(
    cols = c(Training, Test),
    names_to = "Dataset",
    values_to = "RMSE"
  ) %>%
  ggplot(
    aes(x = Model, y = RMSE, fill = Dataset)
  ) +
  geom_col(position = "dodge") +
  labs(
    title = "Training vs. Test RMSE",
    subtitle = "Larger differences may indicate weaker generalization",
    x = NULL,
    y = "RMSE (£)",
    fill = "Dataset"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(
      angle = 20,
      hjust = 1
    )
  )

print(p_generalization)

ggsave(
  "img/fig-training-test-comparison.png",
  p_generalization,
  width = 9,
  height = 6,
  dpi = 300
)


# Improvement over linear regression

baseline_rmse <- test_comparison %>%
  filter(Model == "Linear Regression") %>%
  pull(RMSE)

improvement_comparison <- test_comparison %>%
  mutate(
    RMSE_Improvement = baseline_rmse - RMSE,
    RMSE_Improvement_Percent =
      (baseline_rmse - RMSE) / baseline_rmse * 100
  ) %>%
  select(
    Model,
    RMSE,
    MAE,
    R2,
    RMSE_Improvement,
    RMSE_Improvement_Percent
  )

print(improvement_comparison)

write_csv(
  improvement_comparison,
  "data/processed/model-improvement-comparison.csv"
)


# Final model ranking

final_ranking <- test_comparison %>%
  select(
    Model,
    RMSE,
    MAE,
    R2,
    RMSE_Rank,
    MAE_Rank,
    R2_Rank
  ) %>%
  mutate(
    Average_Rank = (
      RMSE_Rank + MAE_Rank + R2_Rank
    ) / 3
  ) %>%
  arrange(Average_Rank)

print(final_ranking)

write_csv(
  final_ranking,
  "data/processed/final-model-ranking.csv"
)


# Summary

best_rmse_model <- test_comparison %>%
  slice_min(RMSE, n = 1)

best_mae_model <- test_comparison %>%
  slice_min(MAE, n = 1)

best_r2_model <- test_comparison %>%
  slice_max(R2, n = 1)

cat("\nFINAL SUMMARY\n")

cat(
  "Best model by RMSE:",
  as.character(best_rmse_model$Model),
  "\n"
)

cat(
  "Best model by MAE:",
  as.character(best_mae_model$Model),
  "\n"
)

cat(
  "Best model by R²:",
  as.character(best_r2_model$Model),
  "\n"
)


# Best model performance summary

best_model <- test_comparison %>%
  slice_min(RMSE, n = 1)

best_model_rmse_improvement <- (
  baseline_rmse - best_model$RMSE
) / baseline_rmse * 100

cat("\nBEST MODEL SUMMARY\n")

cat(
  "Model:",
  as.character(best_model$Model),
  "\n"
)

cat(
  "Test RMSE:",
  round(best_model$RMSE, 2),
  "\n"
)

cat(
  "Test MAE:",
  round(best_model$MAE, 2),
  "\n"
)

cat(
  "Test R²:",
  round(best_model$R2, 4),
  "\n"
)

cat(
  "RMSE improvement over linear regression:",
  round(best_model_rmse_improvement, 2),
  "%\n"
)
