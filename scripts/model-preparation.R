# PSTAT 197A: Gradient Boosting Vignette
# Model Preparation and Feature Encoding

library(tidyverse)
library(recipes)

# Load reproducible data partitions
source("scripts/data-split.R")

# Define preprocessing recipe
car_recipe <- recipe(price ~ ., data = train_data) %>%
  
  # Combine categorical levels representing less
  # than 0.1% of training observations
  step_other(
    all_nominal_predictors(),
    threshold = 0.001
  ) %>%
  
  # Handle previously unseen categories
  step_novel(all_nominal_predictors()) %>%
  
  # Convert categorical predictors to dummy variables
  step_dummy(all_nominal_predictors()) %>%
  
  # Remove predictors with no variation
  step_zv(all_predictors())

# Estimate preprocessing using training data only
prepared_recipe <- prep(
  car_recipe,
  training = train_data
)

# Apply identical preprocessing to each partition
train_processed <- bake(
  prepared_recipe,
  new_data = train_data
)

validation_processed <- bake(
  prepared_recipe,
  new_data = validation_data
)

test_processed <- bake(
  prepared_recipe,
  new_data = test_data
)

# Separate predictors and response variables
X_train <- train_processed %>%
  select(-price) %>%
  as.matrix()

X_validation <- validation_processed %>%
  select(-price) %>%
  as.matrix()

X_test <- test_processed %>%
  select(-price) %>%
  as.matrix()

y_train <- train_processed$price
y_validation <- validation_processed$price
y_test <- test_processed$price

# Verify matrix dimensions
dim(X_train)
dim(X_validation)
dim(X_test)

# Validate consistency across datasets
stopifnot(
  identical(colnames(X_train), colnames(X_validation)),
  identical(colnames(X_train), colnames(X_test)),
  nrow(X_train) == length(y_train),
  nrow(X_validation) == length(y_validation),
  nrow(X_test) == length(y_test),
  !anyNA(X_train),
  !anyNA(X_validation),
  !anyNA(X_test)
)

cat("All model preparation checks passed.\n")