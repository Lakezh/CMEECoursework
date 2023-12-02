# Load necessary libraries
library(readr)
library(minpack.lm) 
library(dplyr)
library(ggplot2)

# Load the dataset and remove the NAs and negative value in the data
df <- read.csv('../data/LogisticGrowthData.csv')
#remove the NAs in the data
data <- na.omit(df)
write.csv(data, file = '../data/Modified_LogisticGrowthData.csv', row.names = FALSE)

#Replace 0 value as NA
df$PopBio[df$PopBio <= 0] <- NA
#remove the negative values when calculating logN
data <- na.omit(df)
data$LogN <- log(data$PopBio)
# Remove rows with NA, NaN, or Inf values in LogPopBio
data <- data[!is.na(data$LogN) & !is.nan(data$LogN) & !is.infinite(data$LogN), ]

#add a new columen which represent the ID
data <- data %>%
  mutate(Unique_ID = paste(Species, Temp, Medium, Citation, sep = "_"))
subsets <- split(data, data$Unique_ID)
any(is.infinite(data$LogN))
#remove the negatvie time value
create_LogPopBio <- function(subset) {
  if ("PopBio" %in% names(subset)) {
    # Filter out rows where Time <= 0
    subset <- subset[subset$Time > 0, ]
    # Apply log transformation and create new column
    subset$LogN<- log(subset$PopBio)
  } else {
    stop("Column 'PopBio' not found in the dataset")
  }
  return(subset)
}
#apply the function to each subset
subsets <- lapply(subsets, create_LogPopBio)



#fit the linear model
linear_results <- data.frame(Unique_ID = integer(),
                             Model = character(),
                             R2 = numeric(),
                             AIC = numeric(),
                             BIC = numeric(),
                             stringsAsFactors = FALSE)
#make a loop to fit all the subsets
for (id in seq_along(subsets)){
  subset <- subsets[[id]]
  fit_quadratic <- lm(LogN ~ poly(Time, 2, raw = TRUE), data = subset)
  residuals <-subset$LogN - predict(fit_quadratic) 
  ss_res <- sum(residuals^2)
  ss_tot <- sum((subset$LogN - mean(subset$LogN))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  # Calculate AIC and BIC
  n <- nrow(subset) 
  k <- length(coef(fit_quadratic)) 
  log_likelihood <- -n/2 * (log(2 * pi) + log(ss_res / n) + 1)
  aic_value <- 2 * k - 2 * log_likelihood
  bic_value <- log(n) * k - 2 * log_likelihood
  summary_quadratic <- data.frame(Unique_ID = id, Model = "Quadratic", 
                                  R2 = r_squared, 
                                  AIC = aic_value,
                                  BIC = bic_value)
  
  fit_cubic <- lm(LogN ~ poly(Time, 3, raw = TRUE), data = subset)
  residuals <-subset$LogN - predict(fit_cubic) 
  ss_res <- sum(residuals^2)
  ss_tot <- sum((subset$LogN - mean(subset$LogN))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  # Calculate AIC and BIC
  n <- nrow(subset) 
  k <- length(coef(fit_cubic)) 
  log_likelihood <- -n/2 * (log(2 * pi) + log(ss_res / n) + 1)
  aic_value <- 2 * k - 2 * log_likelihood
  bic_value <- log(n) * k - 2 * log_likelihood
  summary_cubic <- data.frame(Unique_ID = subset$Unique_ID, Model = "Cubic", 
                              R2 = r_squared, 
                              AIC = aic_value, 
                              BIC = bic_value)
  result <- rbind(summary_quadratic, summary_cubic)
  linear_results <- rbind(linear_results, result)
  
}


#fit the logestic model, using rolling regression
logistic_model <- function(t, r_max, K, N_0) {
  return(N_0 * K * exp(r_max * t)/(K + N_0 * (exp(r_max * t) - 1)))
}
fit_logistic_model <- function(subset,id){
  subset <- subset[!is.na(subset$LogN) & !is.nan(subset$LogN) & !is.infinite(subset$LogN), ]
  # Calculate starting values using LogN
  N_0_start <- min(subset$LogN)
  K_start <- max(subset$LogN)
  find_max_slope <- function(subset, window_size) {
    max_slope <- -Inf
    # Ensure that window_size is not larger than the number of rows in subset
    if (window_size > nrow(subset)) {
      stop("window_size is larger than the number of rows in the subset")
    }
    for (i in 1:(nrow(subset) - window_size + 1)) {
      logPopBio_window <- subset$LogN[i:(i + window_size - 1)]
      time_window <- subset$Time[i:(i + window_size - 1)]
      fit <- lm(logPopBio_window ~ time_window)
      slope <- coef(fit)["time_window"]
      if (!is.na(slope) && !is.null(slope) && slope > max_slope) {
        max_slope <- slope
      }
    }
    return(max_slope)
  }
  window_size <- 3
  r_max_start <- find_max_slope(subset, window_size)
  
  tryCatch({
    fit_logistic <- nlsLM(LogN ~ logistic_model(t = Time, r_max, K, N_0), subset,
                          start = list(r_max = r_max_start, N_0 = N_0_start, K = K_start),
                          control = nls.lm.control(maxiter = 500))
    return(fit_logistic)
  }, error = function(e) {
    warning("Error in fitting model for subset ", id, ": ", e$message)
    return(NULL)
  })
  
}
# intialize the result
logistic_results <- data.frame(Unique_ID = integer(),
                               Model = character(),
                               R2 = numeric(),
                               AIC = numeric(),
                               BIC = numeric(),
                               stringsAsFactors = FALSE)
for (id in seq_along(subsets)){
  subset <- subsets[[id]]
  subset <- subset[subset$Time > 0, ]
  subset_logisitc <- fit_logistic_model(subset,id)
  tryCatch({
    logistic_points <- logistic_model(t = subset$Time, 
                                      r_max = coef(subset_logisitc)["r_max.time_window"], 
                                      K = coef(subset_logisitc)["K"], 
                                      N_0 = coef(subset_logisitc)["N_0"])
    residuals <-subset$LogN - logistic_points 
    ss_res <- sum(residuals^2)
    ss_tot <- sum((subset$LogN - mean(subset$LogN))^2)
    r_squared <- 1 - (ss_res / ss_tot)
    # Calculate AIC and BIC
    n <- nrow(subset) 
    k <- length(coef(subset_logisitc)) 
    log_likelihood <- -n/2 * (log(2 * pi) + log(ss_res / n) + 1)
    aic_value <- 2 * k - 2 * log_likelihood
    bic_value <- log(n) * k - 2 * log_likelihood
    
    result <- data.frame(Unique_ID = subset$Unique_ID, Model = "Logistic", R2 = r_squared, AIC = aic_value, BIC = bic_value)
    logistic_results <- rbind(logistic_results, result)
  }, error = function(e) {
    warning("Error in calculating results for subset ", id, ": ", e$message)
  })
  
  print(logistic_results)
}



#fit the gompertz model
gompertz_model <- function(t, r_max, K, N_0, t_lag){
  return(N_0 + (K - N_0) * exp(-exp(r_max * exp(1) * (t_lag - t)/((K - N_0) * log(10)) + 1)))
} 
fit_gompertz_model <- function(subset,id){
  subset <- subset[!is.na(subset$LogN) & !is.nan(subset$LogN) & !is.infinite(subset$LogN), ]
  # Calculate starting values using Log population
  N_0_start <- min(subset$LogN)
  K_start <- max(subset$LogN)
  window_size <- 3
  find_max_slope <- function(subset, window_size) {
    max_slope <- -Inf
    # Ensure that window_size is not larger than the number of rows in subset
    if (window_size > nrow(subset)) {
      stop("window_size is larger than the number of rows in the subset")
    }
    for (i in 1:(nrow(subset) - window_size + 1)) {
      logPopBio_window <- subset$LogN[i:(i + window_size - 1)]
      time_window <- subset$Time[i:(i + window_size - 1)]
      fit <- lm(logPopBio_window ~ time_window)
      slope <- coef(fit)["time_window"]
      if (!is.na(slope) && !is.null(slope) && slope > max_slope) {
        max_slope <- slope
      }
    }
    return(max_slope)
  }
  
  r_max_start <- find_max_slope(subset, window_size)
  t_lag_start <- subset$Time[which.max(diff(diff(subset$LogN)))]
  #check if the starting values are available
  N_0_start <- ifelse(is.finite(N_0_start), N_0_start, 1)
  K_start <- ifelse(is.finite(K_start), K_start, max(subset$N, na.rm = TRUE))
  r_max_start <- ifelse(is.finite(r_max_start), r_max_start, 0.1)
  t_lag_start <- ifelse(is.finite(t_lag_start) && t_lag_start > 0, t_lag_start, min(subset$Time, na.rm = TRUE))
  #fit the gompertz model
  tryCatch({
    fit_gompertz <- nlsLM(LogN ~ gompertz_model(t = Time, r_max, K, N_0, t_lag), subset,
                          start = list(t_lag = t_lag_start, r_max = r_max_start, N_0 = N_0_start, K = K_start),
                          control = nls.lm.control(maxiter = 500))
    return(fit_gompertz)
  }, error = function(e) {
    warning("Error in fitting model for subset ", id, ": ", e$message)
  })
  
}
#intialize the result
gompertz_results <- data.frame(Unique_ID = integer(),
                               Model = character(),
                               R2 = numeric(),
                               AIC = numeric(),
                               BIC = numeric(),
                               stringsAsFactors = FALSE)
for (id in seq_along(subsets)){
  subset <- subsets[[id]]
  subset_gompertz <- fit_gompertz_model(subset,id)
  tryCatch({
    gompertz_points <- gompertz_model(t = subset$Time, 
                                      r_max = coef(subset_gompertz)["r_max.time_window"], 
                                      K = coef(subset_gompertz)["K"], 
                                      N_0 = coef(subset_gompertz)["N_0"],
                                      t_lag = coef(subset_gompertz)["t_lag"])
    residuals <-subset$LogN - gompertz_points 
    ss_res <- sum(residuals^2)
    ss_tot <- sum((subset$LogN - mean(subset$LogN))^2)
    r_squared <- 1 - (ss_res / ss_tot)
    # Calculate AIC and BIC
    n <- nrow(subset) 
    k <- length(coef(subset_gompertz)) 
    log_likelihood <- -n/2 * (log(2 * pi) + log(ss_res / n) + 1)
    aic_value <- 2 * k - 2 * log_likelihood
    bic_value <- log(n) * k - 2 * log_likelihood
    
    #output the result
    result <- data.frame(Unique_ID = subset$Unique_ID, Model = "Gompertz", R2 = r_squared, AIC = aic_value, BIC = bic_value)
    gompertz_results <- rbind(gompertz_results, result)
  }, error = function(e) {
    warning("Error in fitting model for subset ", id, ": ", e$message)
  })
  print(gompertz_results)
}

#plot all the results
for (id in seq_along(subsets)) {
  tryCatch({
    subset <- subsets[[id]]
    
    # Fit models and predict the fitted points
    subset_logistic <- fit_logistic_model(subset, id)
    
    fit_quadratic <- lm(LogN ~ poly(Time, 2, raw = TRUE), data = subset)
    quadratic_points <- predict(fit_quadratic)
    
    df_quadratic <- data.frame(subset$Time, quadratic_points)
    df_quadratic$model <- "Quadratic model"
    names(df_quadratic) <- c("Time", "N", "model")
    
    fit_cubic <- lm(LogN ~ poly(Time, 3, raw = TRUE), data = subset)
    cubic_points <- predict(fit_cubic)
    
    df_cubic <- data.frame(subset$Time, cubic_points)
    df_cubic$model <- "Cubic model"
    names(df_cubic) <- c("Time", "N", "model")
    
    if(!is.null(subset_logistic)){
      logistic_points <- logistic_model(t = subset$Time, 
                                        r_max = coef(subset_logistic)["r_max.time_window"], 
                                        K = coef(subset_logistic)["K"], 
                                        N_0 = coef(subset_logistic)["N_0"])
      
      df1 <- data.frame(subset$Time, logistic_points)
    }
    df1$model <- "Logistic equation"
    names(df1) <- c("Time", "N", "model")
    
    subset_gompertz <- fit_gompertz_model(subset, id)
    gompertz_points <- gompertz_model(t = subset$Time, 
                                      r_max = coef(subset_gompertz)["r_max.time_window"], 
                                      K = coef(subset_gompertz)["K"], 
                                      N_0 = coef(subset_gompertz)["N_0"],
                                      t_lag = coef(subset_gompertz)["t_lag"])
    
    df2 <- data.frame(subset$Time, gompertz_points)
    df2$model <- "Gompertz equation"
    names(df2) <- c("Time", "N", "model")
    
    # Create the plot
    p <- ggplot(subset, aes(x = Time, y = LogN)) +
      geom_point(size = 3) +
      geom_line(data = df1, aes(x = Time, y = N, col = model)) +
      geom_line(data = df2, aes(x = Time, y = N, col = model)) +
      geom_line(data = df_quadratic, aes(x = Time, y = N, col = model)) +
      geom_line(data = df_cubic, aes(x = Time, y = N, col = model)) +
      theme(aspect.ratio = 1) +  # Make the plot square
      labs(x = "Time", y = "Cell number")
    
    ggsave(paste("../results/", "subset_", id, "_plot.png"), plot = p, device = "png")
  }, error = function(e) {
    warning("Error in plotting for subset ", id, ": ", e$message)
  })
}



combined_results <- rbind(logistic_results, gompertz_results, linear_results)
write.csv(combined_results, file = '../results/Results.csv', row.names = FALSE)
# Compare models based on the lowest AIC
best_models_aic <- combined_results %>%
  group_by(Unique_ID) %>%
  arrange(AIC) %>%
  slice(1)

# Compare models based on the lowest BIC
best_models_bic <- combined_results %>%
  group_by(Unique_ID) %>%
  arrange(BIC) %>%
  slice(1)

# Compare models based on the highest R-squared
best_models_r2 <- combined_results %>%
  group_by(Unique_ID) %>%
  arrange(desc(R2)) %>%
  slice(1)


best_models_aic
best_models_bic
best_models_r2
table(best_models_aic$Model)
table(best_models_bic$Model)

#calculate the mean of R squared, AIC and BIC of different values
quadratic_result <- linear_results %>% 
  filter(Model == "Quadratic")
cubic_result <- linear_results %>% 
  filter(Model == "Cubic")
cleaned_logistic_results <- logistic_results %>%
  filter(!if_any(everything(), is.infinite))
quadratic_mean_aic <- mean(quadratic_result$AIC)
cubic_mean_aic <- mean(cubic_result$AIC)
logistic_mean_aic <- mean(cleaned_logistic_results$AIC)
gompertz_mean_aic <- mean(gompertz_results$AIC)

quadratic_mean_bic <- mean(quadratic_result$BIC)
cubic_mean_bic <- mean(cubic_result$BIC)
logistic_mean_bic <- mean(cleaned_logistic_results$BIC)
gompertz_mean_bic <- mean(gompertz_results$BIC)

quadratic_mean_r2 <- mean(quadratic_result$R2)
cubic_mean_r2 <- mean(cubic_result$R2)
logistic_mean_r2 <- mean(logistic_results$R2)
gompertz_mean_r2 <- mean(gompertz_results$R2)

