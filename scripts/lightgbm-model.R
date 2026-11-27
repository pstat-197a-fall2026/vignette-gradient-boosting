#LightGBM Regression Model
library(tidyverse)
library(lightgbm)

source("scripts/model-preparation.R")
source("scripts/evaluation-metrics.R")

set.seed(197)

#training data -> LightGBM format
dtrain_lgb <- lgb.Dataset(
  data = X_train,
  label = y_train
)

#validation dataset using training data
dvalidation_lgb <- lgb.Dataset.create.valid(
  dataset = dtrain_lgb,
  data = X_validation,
  label = y_validation
)

#hyperparameters
lgb_parameters <- list(
  objective = "regression",
  metric = "rmse",
  learning_rate = 0.05,
  num_leaves = 31,
  max_depth = 8,
  min_data_in_leaf = 20,
  feature_fraction = 0.8,
  bagging_fraction = 0.8,
  bagging_freq = 1,
  num_threads = 4,
  seed = 197,
  verbosity = -1
  
)

#LightGBM early stopping 
lgb_model <- lgb.train(
  params = lgb_parameters,
  data = dtrain_lgb,
  nrounds = 2500,
  valids = list(validation = dvalidation_lgb),
  early_stopping_rounds = 30,
  verbose = 1,
  eval_freq = 100
)

#predictions using best boosting iteration
train_pred_lgb <- predict(
  lgb_model, X_train,
  num_iteration = lgb_model$best_iter
)

validation_pred_lgb <- predict(
  lgb_model, X_validation,
  num_iteration = lgb_model$best_iter
)

test_pred_lgb <- predict(
  lgb_model, X_test,
  num_iteration = lgb_model$best_iter
)

#evaluation of LightGBM performance
lgb_results <- bind_rows(
  evaluate_model(y_train, train_pred_lgb) |>
    mutate(Dataset = "training"),
  
  evaluate_model(y_validation, validation_pred_lgb) |>
    mutate(Dataset = "validation"),
  
  evaluate_model(y_test, test_pred_lgb) |>
    mutate(Dataset = "test")
) |>
  mutate(Model = "LightGBM") |>
  select(Model, Dataset, everything()
)
print(lgb_results)

#results for model comparison
write_csv(
  lgb_results,
  "data/processed/lightgbm-results.csv"
)

#validation predictions for visualization
lgb_predictions <- tibble(
  actual = y_validation,
  predicted = validation_pred_lgb
)

set.seed(197)

lgb_plot_data <- lgb_predictions |>
  slice_sample(n = 5000)

#actual vs. predicted
p_lgb <- ggplot(
  lgb_plot_data,
  aes(x = actual, y = predicted)
) +
  geom_point(alpha = 0.2, color = "cyan4") +
  geom_abline(
    intercept = 0,
    slope = 1,
    color = "lightpink",
    linetype = "dashed"
  ) +
  scale_x_continuous(
    labels = scales::label_dollar(prefix = "£")
  ) + 
  scale_y_continuous(
    labels = scales::label_dollar(prefix = "£")
  ) + 
  coord_equal() +
  theme_minimal() +
  labs(title = "LightGBM: Actual(x) vs. Predicted Prices(y)",
       subtitle = "Validation Dataset",
       x = "Actual Price",
       y = "Predicted Price"
  )

print(p_lgb)

#plot save for future reference
ggsave(
  "img/fig-lightgbm-predictions.png",
  plot = p_lgb,
  width = 8,
  height = 6,
  dpi = 600
)

