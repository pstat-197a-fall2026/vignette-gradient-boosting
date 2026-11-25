# PSTAT 197A: Gradient Boosting Vignette
# Exploratory Data Analysis

library(tidyverse)

# Load reproducible data partitions
source("scripts/data-split.R")

# Create image directory if necessary
dir.create("img", showWarnings = FALSE)

# Set consistent plotting theme
theme_set(theme_minimal(base_size = 12))

# Format monetary values in British pounds
price_labels <- scales::label_dollar(prefix = "£")

# Inspect training dataset
glimpse(train_data)
summary(train_data)

# Calculate summary statistics for numerical variables
numeric_summary <- train_data %>%
  select(where(is.numeric)) %>%
  pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "value"
  ) %>%
  group_by(variable) %>%
  summarize(
    minimum = min(value),
    mean = mean(value),
    median = median(value),
    sd = sd(value),
    maximum = max(value),
    .groups = "drop"
  )

print(numeric_summary, width = Inf)

# Summarize price by manufacturer
manufacturer_summary <- train_data %>%
  group_by(make) %>%
  summarize(
    observations = n(),
    mean_price = mean(price),
    median_price = median(price),
    sd_price = sd(price),
    .groups = "drop"
  ) %>%
  arrange(desc(median_price))

print(manufacturer_summary)

# Examine categorical variables
train_data %>% count(transmission, sort = TRUE)
train_data %>% count(fuelType, sort = TRUE)
train_data %>% count(make, sort = TRUE)

# Distribution of vehicle prices
p1 <- ggplot(train_data, aes(x = price)) +
  geom_histogram(
    bins = 60,
    fill = "cyan",
    color = "white"
  ) +
  geom_vline(
    xintercept = median(train_data$price),
    color = "green",
    linetype = "dashed",
    linewidth = 1
  ) +
  scale_x_continuous(labels = price_labels) +
  labs(
    title = "Distribution of Used Car Prices",
    subtitle = "Dashed line represents the median price",
    x = "Vehicle price",
    y = "Number of vehicles"
  )

print(p1)

ggsave(
  "img/fig-price-distribution.png",
  plot = p1,
  width = 8,
  height = 5,
  dpi = 300
)

# Sample observations to reduce visual overcrowding
set.seed(198)

plot_data <- train_data %>%
  slice_sample(n = 5000)

# Calculate median prices within 10,000-mile intervals
mileage_summary <- train_data %>%
  mutate(mileage_bin = floor(mileage / 10000)) %>%
  group_by(mileage_bin) %>%
  summarize(
    mileage = median(mileage),
    price = median(price),
    observations = n(),
    .groups = "drop"
  ) %>%
  filter(observations >= 30)

# Price versus mileage
p2 <- ggplot(
  plot_data,
  aes(x = mileage, y = price)
) +
  geom_point(
    alpha = 0.15,
    color = "cyan",
    size = 1
  ) +
  geom_line(
    data = mileage_summary,
    color = "red",
    linewidth = 1.2
  ) +
  scale_x_continuous(labels = scales::label_comma()) +
  scale_y_continuous(labels = price_labels) +
  labs(
    title = "Relationship Between Mileage and Price",
    subtitle = "Red line shows median prices by mileage interval",
    x = "Mileage (miles)",
    y = "Vehicle price"
  )

print(p2)

ggsave(
  "img/fig-price-mileage.png",
  plot = p2,
  width = 8,
  height = 5,
  dpi = 300
)

# Summarize price by registration year
year_summary <- train_data %>%
  group_by(year) %>%
  summarize(
    median_price = median(price),
    lower_quartile = quantile(price, 0.25),
    upper_quartile = quantile(price, 0.75),
    observations = n(),
    .groups = "drop"
  ) %>%
  filter(observations >= 30)

# Plot median price and interquartile range
p3 <- ggplot(
  year_summary,
  aes(x = year, y = median_price)
) +
  geom_ribbon(
    aes(
      ymin = lower_quartile,
      ymax = upper_quartile
    ),
    fill = "purple",
    alpha = 0.2
  ) +
  geom_line(
    color = "purple",
    linewidth = 1
  ) +
  geom_point(color = "purple") +
  scale_y_continuous(labels = price_labels) +
  labs(
    title = "Vehicle Price by Registration Year",
    subtitle = "Shaded area represents the interquartile range",
    x = "Registration year",
    y = "Vehicle price"
  )

print(p3)

ggsave(
  "img/fig-price-year.png",
  plot = p3,
  width = 8,
  height = 5,
  dpi = 300
)

# Compare price distributions across manufacturers
p4 <- ggplot(
  train_data,
  aes(
    x = fct_reorder(make, price, .fun = median),
    y = price,
    fill = make
  )
) +
  geom_boxplot(outlier.alpha = 0.1) +
  coord_flip() +
  scale_y_continuous(labels = price_labels) +
  guides(fill = "none") +
  labs(
    title = "Vehicle Prices by Manufacturer",
    subtitle = "Manufacturers ordered by median price",
    x = "Manufacturer",
    y = "Vehicle price"
  )

print(p4)

ggsave(
  "img/fig-price-manufacturer.png",
  plot = p4,
  width = 8,
  height = 5,
  dpi = 300
)

# Calculate median prices by manufacturer and mileage
manufacturer_mileage <- train_data %>%
  mutate(mileage_bin = floor(mileage / 10000)) %>%
  group_by(make, mileage_bin) %>%
  summarize(
    mileage = median(mileage),
    median_price = median(price),
    observations = n(),
    .groups = "drop"
  ) %>%
  filter(observations >= 30)

# Compare mileage-price relationships
p5 <- ggplot(
  manufacturer_mileage,
  aes(
    x = mileage,
    y = median_price,
    color = make
  )
) +
  geom_line(linewidth = 1) +
  geom_point(size = 1.5) +
  facet_wrap(~ make, ncol = 3) +
  scale_x_continuous(labels = scales::label_comma()) +
  scale_y_continuous(labels = price_labels) +
  guides(color = "none") +
  labs(
    title = "Mileage and Price Across Manufacturers",
    subtitle = "Median prices within 10,000-mile intervals",
    x = "Mileage (miles)",
    y = "Median vehicle price"
  )

print(p5)

ggsave(
  "img/fig-mileage-manufacturer.png",
  plot = p5,
  width = 11,
  height = 8,
  dpi = 300
)

# Calculate correlations between numerical variables
correlation_matrix <- train_data %>%
  select(where(is.numeric)) %>%
  cor()

# Display rounded correlation matrix
print(round(correlation_matrix, 3))

# Rank numerical predictors by absolute correlation with price
price_correlations <- tibble(
  variable = rownames(correlation_matrix),
  correlation = correlation_matrix[, "price"]
) %>%
  filter(variable != "price") %>%
  arrange(desc(abs(correlation)))

print(price_correlations)