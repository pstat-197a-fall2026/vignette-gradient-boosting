
# PSTAT 197A: Gradient Boosting Vignette
# Preliminary Data Exploration


library(tidyverse)

# Verify working directory
getwd()

# Identify the manufacturer datasets
files <- list.files(
  path = "data/raw",
  pattern = "\\.csv$",
  full.names = TRUE
)

# Display available files
print(files)

# Inspect each individual dataset
for (file in files) {
  
  dat <- read_csv(file, show_col_types = FALSE)
  
  cat("\n============================\n")
  cat("File:", basename(file), "\n")
  cat("Rows:", nrow(dat), "\n")
  cat("Columns:", ncol(dat), "\n")
  cat("Column names:", paste(names(dat), collapse = ", "), "\n")
  cat("Missing values:", sum(is.na(dat)), "\n")
}
