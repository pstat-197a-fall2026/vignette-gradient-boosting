#handling missing data 
library(tidyverse)
library(xgboost)
library(lightgbm)

source("scripts/model-preparation.R")
source("scripts/evaluation-metrics.R")

set.seed(197)

#verification that original data has no missing vals
sum(is.na(X_train))
sum(is.na(X_validation))

X_train_missing <- X_train
X_validation_missing <- X_validation

#randomly select 20% of training observations
missing_train <- sample(
  seq_len(nrow(X_train)),
  size = floor(0.20 * nrow(X_train))
)

#randomly select 20% of validation observations
missing_validation <- sample(
  seq_len(nrow(X_validation)),
  size = floor(0.20 * nrow(X_validation))
)

X_train_missing[missing_train, "mileage"] <- NA
X_validation_missing[missing_validation, "mileage"] <- NA

#number of NA mileage vals
sum(is.na(X_train_missing[, "mileage"]))
sum(is.na(X_validation_missing[, "mileage"]))

#proportion of NA mileage vals
mean(is.na(X_train_missing[, "mileage"]))
mean(is.na(X_validation_missing[, "mileage"]))

#fx to train and evaluate XGBoost
fit_xgb <- function(train_x, validation_x) {
  set.seed(197)
  
  #XGBoost Format
  dtrain <- xgb.DMatrix(
    data = train_x,
    label = y_train,
    missing = NA
  )
  
  dvalidation <- xgb.DMatrix(
    data = validation_x,
    label = y_validation,
    missing = NA
  )
  
  #train XGBoost
  model <- xgb.train(
    params = list(
      objective = "reg:squarederror",
      eval_metric = "rmse",
      max_depth = 6,
      eta = 0.1,
      subsample = 0.8,
      colsample_bytree = 0.8,
      tree_method = "hist",
      nthread = 4
    ),
    data = dtrain,
    nrounds = 300,
    verbose = 0
  )
  predictions <- predict(model, dvalidation)
  evaluate_model(y_validation, predictions)
}

#XGBoost complete/full data
xgb_complete <- fit_xgb(
  X_train,
  X_validation
)

#XGBoost w/ missing data
xgb_missing <- fit_xgb(
  X_train_missing,
  X_validation_missing
)

print(xgb_complete)
print(xgb_missing)


# training and evaluating LightGBM
fit_lgb <- function(train_x, validation_x) {
  set.seed(197)
  
  dtrain <- lgb.Dataset(
    data = train_x,
    label = y_train
  )
  #LightGBM train
  model <- lgb.train(
    params = list(
      objective = "regression",
      metric = "rmse",
      learning_rate = 0.1,
      num_leaves = 31,
      max_depth = 6,
      min_data_in_leaf = 20,
      feature_fraction = 0.8,
      bagging_fraction = 0.8,
      bagging_freq = 1,
      use_missing = TRUE,
      zero_as_missing = FALSE,
      num_threads = 4,
      seed = 197,
      verbosity = -1
    ),
    data = dtrain,
    nrounds = 300,
    verbose = 0
  )
  predictions <- predict(model, validation_x)
  evaluate_model(y_validation, predictions)
}

lgb_complete <- fit_lgb(
  X_train,
  X_validation
)

lgb_missing <- fit_lgb(
  X_train_missing,
  X_validation_missing
)

print(lgb_complete)
print(lgb_missing)


#combining results for report
missing_results <- bind_rows(
  xgb_complete |>
    mutate(Model = "XGBoost", Missing = "0%"),
  xgb_missing |>
    mutate(Model = "XGBoost", Missing = "20%"),
  lgb_complete |>
    mutate(Model = "LightGBM", Missing = "0%"),
  lgb_missing |>
    mutate(Model = "LightGBM", Missing = "20%") 
) |>
  select(Model, Missing, everything())

print(missing_results)

# missing experiment results
write_csv(
  missing_results,
  "data/processed/missing-data-results.csv"
)

#% increase in RMSE due to introduction of missingness
missing_summary <- missing_results |>
  group_by(Model) |>
  summarize(
    complete_rmse = RMSE[Missing == "0%"],
    missing_rmse = RMSE[Missing == "20%"],
    percent_change = 100 *
      (missing_rmse - complete_rmse) / complete_rmse,
    .groups = "drop"
  )

print(missing_summary)

#RMSE w and w/o missingness
p_missing <- ggplot(
  missing_results,
  aes(x = Model, y = RMSE, fill = Missing)
) +
  geom_col(
    position = position_dodge(width = 0.7),
    width = 0.6
  ) +
  geom_text(
    aes(label = round(RMSE, 0)),
    position = position_dodge(width = 0.7),
    vjust = -0.5,
    size = 4
  ) +
  scale_fill_manual(
    values = c(
     "0%" = "cyan4",
     "20%" = "lightpink"
    )
  ) +
  scale_y_continuous(
    labels = scales::label_dollar(prefix = "£"),
    expand = expansion(mult = c(0,0.1))
  ) +
  theme_minimal() +
  labs(title = "Impact of Missing Data on Gradient Boosting",
       subtitle = "Randomly Removing 20% of Mileage Values",
       x = "Model",
       y = "Validation RMSE",
       fill = "Missing Mileage"
)

print(p_missing)

#plot for future reference
ggsave(
  "img/fig-missing-data-comparison.png",
  plot = p_missing,
  width = 8,
  height = 6,
  dpi = 600
)
