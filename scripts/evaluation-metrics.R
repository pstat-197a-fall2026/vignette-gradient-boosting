# PSTAT 197A: Gradient Boosting Vignette
# Shared Model Evaluation Metrics

evaluate_model <- function(actual, predicted) {
  
  # Root Mean Squared Error
  rmse <- sqrt(mean((actual - predicted)^2))
  
  # Mean Absolute Error
  mae <- mean(abs(actual - predicted))
  
  # Coefficient of determination
  r_squared <- 1 -
    sum((actual - predicted)^2) /
    sum((actual - mean(actual))^2)
  
  # Return evaluation metrics
  tibble::tibble(
    RMSE = rmse,
    MAE = mae,
    R2 = r_squared
  )
}