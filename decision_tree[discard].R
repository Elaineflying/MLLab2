# Load necessary libraries
library(ggplot2)
library(rpart)
library(caret)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

# Import data set
bank_data <- read.csv('bank-full.csv', header = TRUE, sep = ';')
#head(bank_data, 5)

# Split the data into training and test sets
bank_data <- subset(bank_data, select = -c(duration))

#bank_data[sapply(bank_data, is.character)] <- lapply(bank_data[sapply(bank_data, is.character)], as.factor)
#str(bank_data)
bank_data$y <- as.factor(bank_data$y)
n=dim(bank_data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.4))
train_data=bank_data[id,]
id1=setdiff(1:n, id)
id2=sample(id1, floor(n*0.3))
valid_data=bank_data[id2,]
id3=setdiff(id1,id2)
test_data=bank_data[id3,]

# Decision Tree with default settings
default_tree_model <- rpart(y ~ ., data=train_data, method = "class")
plotcp(default_tree_model)

# Decision Tree with smallest allowed node size equal to 7000
nodesize_tree_model <- rpart(y ~ ., data = train_data, method = "class", control = rpart.control(minsplit = 7000))
plotcp(nodesize_tree_model)

# Decision trees minimum deviance to 0.0005
deviance_tree_model <- rpart(y ~ ., data = train_data, method = "class", control = rpart.control(cp = 0.0005))
plotcp(deviance_tree_model)

#fancyRpartPlot(deviance_tree_model, caption = NULL)

# Compute confusion matrices and misclassification rates for each model
cm_mc_function <- function(model, data) {
  predictions <- predict(model, data, type = "class")
  confusion_matrix <- confusionMatrix(predictions, data$y)
  misclassification_rate <- 1- confusion_matrix$overall["Accuracy"]
  return(list(confusion_matrix=confusion_matrix, misclassification_rate=misclassification_rate))
}

# Calculate the misclassification rates for the training data
misclassification_rate_train1 <- cm_mc_function(default_tree_model,train_data)$misclassification_rate
misclassification_rate_train2 <- cm_mc_function(nodesize_tree_model,train_data)$misclassification_rate
misclassification_rate_train3 <- cm_mc_function(deviance_tree_model,train_data)$misclassification_rate

# Calculate the misclassification rates for the validation data
misclassification_rate_validation1 <- cm_mc_function(default_tree_model,valid_data)$misclassification_rate
misclassification_rate_validation2 <- cm_mc_function(nodesize_tree_model,valid_data)$misclassification_rate
misclassification_rate_validation3 <- cm_mc_function(deviance_tree_model,valid_data)$misclassification_rate

comparasion_df <- data.frame(models=c("decision tree default", "decision tree with 7000 node size", "decision tree with 0.0005 deviance"),
                             train_data_misclassification_rate=c(misclassification_rate_train1,misclassification_rate_train2,misclassification_rate_train3),
                             validation_data_misclassification_rate=c(misclassification_rate_validation1,misclassification_rate_validation2,misclassification_rate_validation3))

print(comparasion_df)


# Create a sequence of cp values
cp_grid <- expand.grid(.cp = seq(0.001, 0.08, 0.001))

# Train the model with different cp values
fit <- train(y ~ ., data = train_data, method = "rpart", trControl = trainControl(method = "cv"), tuneGrid = cp_grid)


# Plot the results
plot(fit)


# Get the optimal cp value
optimalCp <- fit$bestTune$cp

# Prune the tree to the optimal cp value
prunedTree <- prune(fit$finalModel, cp = optimalCp)

# Plot the pruned tree
plot(prunedTree)
text(prunedTree)



# Predict the test data
predictions <- predict(fit, test_data, type = "class")

# Compute the confusion matrix
cm_test <- confusionMatrix(predictions, test_data$y)

# Compute the accuracy
accuracy <- cm$overall["Accuracy"]

# Compute the F1 score
f1_score <- 2 * (cm$byClass["Sensitivity"] * cm$byClass["Pos Pred Value"]) / (cm$byClass["Sensitivity"] + cm$byClass["Pos Pred Value"])



