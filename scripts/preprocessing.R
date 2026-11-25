
# PSTAT 197A: Gradient Boosting Vignette
# Data Preprocessing

library(tidyverse)

# Identify datasets and corresponding manufacturers

manufacturers <- c(
  audi = "Audi",
  bmw = "BMW",
  ford = "Ford",
  hyundi = "Hyundai",
  merc = "Mercedes",
  skoda = "Skoda",
  toyota = "Toyota",
  vauxhall = "Vauxhall",
  vw = "Volkswagen"
)

# Read and standardize each manufacturer's dataset

read_car_data <- function(file_name) {
  
  # Read CSV file
  dat <- read_csv(
    file.path("data/raw", paste0(file_name, ".csv")),
    show_col_types = FALSE
  )
  
  # Standardize tax column name
  if ("tax(£)" %in% names(dat)) {
    dat <- dat %>%
      rename(tax = `tax(£)`)
  }
  
  # Add manufacturer information
  dat <- dat %>%
    mutate(
      make = unname(manufacturers[file_name]),
      model = str_trim(model)
    )
  
  return(dat)
}


# Combine all nine datasets


cars_raw <- map_dfr(
  names(manufacturers),
  read_car_data
)

# Inspect combined dataset
dim(cars_raw)
glimpse(cars_raw)
summary(cars_raw)

# Count observations by manufacturer
cars_raw %>%
  count(make, sort = TRUE)

# Check missing values
colSums(is.na(cars_raw))

# Count exact duplicate rows
sum(duplicated(cars_raw))


# Remove duplicates and implausible model years


cars <- cars_raw %>%
  distinct() %>%
  filter(year >= 1990, year <= 2020)

# Inspect potentially problematic observations


# Unusually old vehicles
cars %>%
  filter(year < 1990)

# Vehicles with zero engine size
cars %>%
  filter(engineSize == 0) %>%
  count(fuelType)

# Vehicles with unusually low fuel efficiency
cars %>%
  filter(mpg < 5)


# Verify preliminary cleaned dataset


dim(cars)
summary(cars)
colSums(is.na(cars))
sum(duplicated(cars))


# Export preliminary cleaned dataset


write_csv(
  cars,
  "data/processed/cars-clean.csv"
)

# Print final dimensions
cat("Original observations:", nrow(cars_raw), "\n")
cat("Cleaned observations:", nrow(cars), "\n")
cat("Removed observations:",
    nrow(cars_raw) - nrow(cars), "\n")


# Validate final dataset

stopifnot(
  nrow(cars) == 97709,
  ncol(cars) == 10,
  sum(duplicated(cars)) == 0,
  sum(is.na(cars)) == 0,
  all(cars$price > 0),
  all(cars$mileage >= 0),
  all(cars$year >= 1990),
  all(cars$year <= 2020),
  n_distinct(cars$make) == 9
)

cat("All data validation checks passed.\n")