# Load required library
library(ggplot2)
library(caret)

# Load the data
crime_data <- read.csv("communities.csv")
#head(crime_data)

# Scale data
data_scaled <- scale(crime_data[, !names(crime_data) %in% "ViolentCrimesPerPop"])

# eigenvalues
eigen_res <- eigen(cor(data_scaled))

# Calculate the proportion of variation explained
cumulative_variance <- cumsum(eigen_res$values) / sum(eigen_res$values)

# Find the number of components needed for at least 95% variance
components_needed <- which(cumulative_variance >= 0.95)[1]
variance_first_two <- cumulative_variance[1:2]

cat("Number of components needed for at least 95% variance:", components_needed, "\n")
cat("Proportion of variation explained by each of the first two principal components:\n")
cat("PC1:", variance_first_two[1], "\n")
cat("PC2:", variance_first_two[2]-variance_first_two[1], "\n")



# Perform PCA using princomp()
pca_res <- princomp(data_scaled)

# Trace plot of the first principal component
# plot(pca_res)

trace_plot <- plot(pca_res$loadings[,1])
print(trace_plot)

# Features contributing to the first principal component
top_5_features <- sort(abs(pca_res$loadings[, 1]), decreasing = TRUE)[1:5]

# Print the top 5 contributing features
cat("Top 5 features contributing mostly to the first principal component:\n")
print(top_5_features)

# Plot PC scores in coordinates (PC1, PC2) with color given by ViolentCrimesPerPop
pc_scores <- predict(pca_res)
response <- crime_data$ViolentCrimesPerPop

ggplot(data.frame(PC1 = pc_scores[, 1], PC2 = pc_scores[, 2], CrimeLevel = response), aes(x = PC1, y = PC2, color = CrimeLevel)) +
  geom_point() +
  scale_color_gradient(low = "orange", high = "green") +
  labs(title = "PC Scores Plot", x = "PC1", y = "PC2")


# Split the data into training and test sets
n=dim(crime_data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5))
train_data=crime_data[id,]
test_data=crime_data[-id,]

# Use preProcess to scale the data and obtain a scaling model
scaling_model <- preProcess(train_data, method = c("center", "scale"))

# Apply the scaling to both training and test sets
scaled_train_data <- predict(scaling_model, train_data)
scaled_test_data <- predict(scaling_model, test_data)


# Fit linear regression model
lm_model <- lm(scaled_train_data$ViolentCrimesPerPop ~ ., data = scaled_train_data)

# Compute training and test errors
train_pred <- predict(lm_model, newdata = scaled_train_data)
test_pred <- predict(lm_model, newdata = scaled_test_data)

train_error <- mean((scaled_train_data$ViolentCrimesPerPop - train_pred)^2)
test_error <- mean((scaled_test_data$ViolentCrimesPerPop - test_pred)^2)

# Print the results
cat("Training Error:", train_error, "\n")
cat("Test Error:", test_error, "\n")


# Cost function for linear regression without intercept on training data set
cost_function <- function(theta, X, y) {
  h <- X %*% theta
  cost <- 1 / (2 * length(y)) * sum((h - y)^2)
  return(cost)
}


# Compute training and test errors for every iteration
max_iter <- 1000
scaled_train_features <- as.matrix(scaled_train_data[, -which(names(train_data) == "ViolentCrimesPerPop")])
scaled_test_features <- as.matrix(scaled_test_data[, -which(names(test_data) == "ViolentCrimesPerPop")])
scaled_train_response <- scaled_train_data$ViolentCrimesPerPop
scaled_test_response <- scaled_test_data$ViolentCrimesPerPop
training_errors <- numeric(length = max_iter)
test_errors <- numeric(length = max_iter)
theta_i <- rep(0, ncol(scaled_train_features))

for (i in 1:max_iter) {
  optim_res <- optim(par = theta_i,
                     fn = cost_function,
                     X = scaled_train_features,
                     y = scaled_train_response,
                     method = "BFGS",
                     control = list(maxit = i))
  theta_i <- optim_res$par
  training_errors[i] <- cost_function(theta_i, scaled_train_features, scaled_train_response)
  test_errors[i] <- cost_function(theta_i, scaled_test_features, scaled_test_response)

  # Add a convergence check, adjust the tolerance as needed
  if (optim_res$convergence == 0) {
    break
  }
}

# Plot dependence of both errors on the iteration number
plot(training_errors, type = "l", col = "blue", xlab = "Iteration", ylab = "Error", main = "Training and Test Errors vs Iteration")
lines(test_errors, col = "red")
legend("topright", legend = c("Training Error", "Test Error"), col = c("blue", "red"), lty = 1)

