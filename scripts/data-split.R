# PSTAT 197A: Gradient Boosting Vignette
# Reproducible Data Partitioning

library(tidyverse)

# Load cleaned dataset
cars <- read_csv(
  "data/processed/cars-clean.csv",
  show_col_types = FALSE
)

# Set seed for reproducibility
set.seed(197)

# Randomly shuffle row indices
n <- nrow(cars)
indices <- sample(seq_len(n))

# Determine partition sizes
n_train <- floor(0.70 * n)
n_validation <- floor(0.15 * n)

# Assign row indices
train_indices <- indices[seq_len(n_train)]

validation_indices <- indices[
  (n_train + 1):(n_train + n_validation)
]

test_indices <- indices[
  (n_train + n_validation + 1):n
]

# Create datasets
train_data <- cars[train_indices, ]
validation_data <- cars[validation_indices, ]
test_data <- cars[test_indices, ]

# Verify partition sizes
cat("Training observations:", nrow(train_data), "\n")
cat("Validation observations:", nrow(validation_data), "\n")
cat("Testing observations:", nrow(test_data), "\n")

# Validate partitions
stopifnot(
  nrow(train_data) +
    nrow(validation_data) +
    nrow(test_data) == nrow(cars),
  length(unique(c(
    train_indices,
    validation_indices,
    test_indices
  ))) == nrow(cars)
)
