# Load necessary libraries
library(glmnet)
library(ggplot2)

# Import data set
tecator_data <- read.csv('tecator.csv', header = TRUE)
#head(tecator_data, 5)

# Split the data into training and test sets
tecator_data <- subset(tecator_data, select = -c(1,103,104))
n=dim(tecator_data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5))
train_data=tecator_data[id,]
test_data=tecator_data[-id,]

# linear regression model
lm_model <- lm(Fat ~ ., data = train_data)

# Significant variables
summary(lm_model)

# Training and test MSE
train_pred <- predict(lm_model, newdata = train_data)
train_mse <- round(mean((train_data$Fat - train_pred)^2),5)

test_pred <- predict(lm_model, newdata = test_data)
test_mse <- round(mean((test_data$Fat - test_pred)^2),5)
results_df <- data.frame(dataset=c("train_data", "test_data"),
                         mse=c(train_mse, test_mse))

print(results_df)

# The cost function is ||y - Xb||^2 + lambda * ||b||_1

# LASSO model
x <- model.matrix(Fat ~ ., data = train_data)[,-1]
y <- train_data$Fat
lasso_model <- glmnet(x, y, alpha = 1)
plot(lasso_model, xvar = "lambda", label = TRUE)

coef_res <- coef(lasso_model, s = 10^(-0.1))

lasso_3_features <- coef_res[coef_res[,1]!=0,]
cat("When the penalty factor \$log(\lambda)\$ is -0.1, the LASSO model only has three feactures, \n")
cat("Three feactures are: \n")
print(lasso_3_features)


# Ridge model
ridge_model <- glmnet(x, y, alpha = 0)
plot(ridge_model, xvar = "lambda", label = TRUE)

# Cross-validation to optimal LASSO model
cv_model <- cv.glmnet(x, y, alpha = 1)
plot(cv_model)

# Optimal lambda for LASSO
opt_lambda <- cv_model$lambda.min

# Number of variables chosen in the model
num_variables <- sum(coef(cv_model, s = opt_lambda) != 0)

cat("Optimal LASSO Lambda:", opt_lambda, "\n")
cat("Number of Variables:", num_variables, "\n")


opt_model <- glmnet(x, y, alpha = 1, lambda = opt_lambda)
coef(opt_model)
test_predict <- predict(opt_model, newx = model.matrix(Fat ~ ., data = test_data)[,-1])
plot(test_data$Fat, test_predict, main = "LASSO Model: Original vs Predicted (Test Set)", xlab = "Original test values", ylab = "Predicted test values")




