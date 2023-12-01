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

bank_data[sapply(bank_data, is.character)] <- lapply(bank_data[sapply(bank_data, is.character)], as.factor)
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



n <- nrow(train_data)

# Decision Tree with default settings
tree_model_default <- tree(y ~ ., data=train_data)

# Decision Tree with smallest allowed node size equal to 7000
tree_model_nodesize <- tree(y ~ ., data=train_data, control = tree.control(nobs = n, minsize = 7000))

# Decision trees minimum deviance to 0.0005
tree_model_deviance <- tree(y ~ ., data=train_data, control = tree.control(nobs = n, mindev = 0.0005))
plot(tree_model_deviance)

# Compute confusion matrices and misclassification rates for each model
cm_mc_function <- function(model, data) {
  predictions <- predict(model, data, type = "class")
  confusion_matrix <- confusionMatrix(predictions, data$y)
  misclassification_rate <- 1- confusion_matrix$overall["Accuracy"]
  return(list(confusion_matrix=confusion_matrix, misclassification_rate=misclassification_rate))
}

# Calculate the misclassification rates for the training data
misclassification_rate_train1 <- cm_mc_function(tree_model_default,train_data)$misclassification_rate
misclassification_rate_train2 <- cm_mc_function(tree_model_nodesize,train_data)$misclassification_rate
misclassification_rate_train3 <- cm_mc_function(tree_model_deviance,train_data)$misclassification_rate

# Calculate the misclassification rates for the validation data
misclassification_rate_validation1 <- cm_mc_function(tree_model_default,valid_data)$misclassification_rate
misclassification_rate_validation2 <- cm_mc_function(tree_model_nodesize,valid_data)$misclassification_rate
misclassification_rate_validation3 <- cm_mc_function(tree_model_deviance,valid_data)$misclassification_rate

comparasion_df <- data.frame(models=c("decision tree default", "decision tree with 7000 node size", "decision tree with 0.0005 deviance"),
                             train_data_misclassification_rate=c(misclassification_rate_train1,misclassification_rate_train2,misclassification_rate_train3),
                             validation_data_misclassification_rate=c(misclassification_rate_validation1,misclassification_rate_validation2,misclassification_rate_validation3))

print(comparasion_df)


train_deviance=rep(0,50)
valid_deviance=rep(0,50)
for ( i in 2:50) {
  pruned_tree=prune.tree(tree_model_deviance, best = i)
  predictions = predict(pruned_tree, newdata=valid_data, type="tree")
  train_deviance[i]=deviance(pruned_tree)
  valid_deviance[i]=deviance(predictions)
}


# Plot deviances for the training and validation data on the number of leaves
plot(2:50, train_deviance[2:50], type = "b", col = "blue", ylim=c(7800,12000), xlab = "Number of Leaves", ylab = "Deviance", main = "Deviance vs Number of Leaves")
lines(2:50, valid_deviance[2:50], type = "b", col = "red")
legend("topright", legend = c("Training", "Validation"), fill = c("blue", "red"))


# Find the optimal number of leaves (up to 50 leaves)
optimal_num_leaves <- which.min(valid_deviance[2:50])
cat("Optimal Number of Leaves: ", optimal_num_leaves, "\n")

# Fit the optimal tree
optimal_tree_model_deviance <- prune.tree(tree_model_deviance, best = optimal_num_leaves)
summary(optimal_tree_model_deviance)

# Variable importance
##"poutcome" "month"    "contact"  "pdays"    "age"      "day"      "balance"  "housing"


# Predict the test data
predicted_test <- predict(optimal_tree_model_deviance, test_data, type = "class")

# Compute the confusion matrix
cm_test <- confusionMatrix(predicted_test, test_data$y)

# Compute the accuracy
accuracy_score <- cm_test$overall["Accuracy"]

# Compute the F1 score
# precision = TP / (TP + FP) ---- "Pos Pred Value"
# recall = TP / (TP + FN) ---- "Sensitivity"
# precision <- cm_test$table[[1,1]] / (cm_test$table[[1,1]] + cm_test$table[[1,2]])
# recall <- cm_test$table[[1,1]] / (cm_test$table[[1,1]] + cm_test$table[[2,1]])
# f1_score = 2 * (precision * recall) / (precision + recall)
precision_score <- cm_test$byClass["Pos Pred Value"]
recall_score <- cm_test$byClass["Sensitivity"]
f1_score <- 2 * (cm_test$byClass["Sensitivity"] * cm_test$byClass["Pos Pred Value"]) / (cm_test$byClass["Sensitivity"] + cm_test$byClass["Pos Pred Value"])

cat("Confusion Matrix:")
print(cm_test)
cat("Accuracy Score: ", accuracy_score, "\n")
cat("Precision Score: ", precision_score, "\n")
cat("Recall Score: ", recall_score, "\n")
cat("F1 Score: ", f1_score, "\n")


###
# the f1_score is quite higher than accuracy_score, this is because we have imbalanced classes.
# the accuracy is very close to the precision, and quite dissimilar to recall. This is means that precision
# (accuracy of positive predictions) is dominating the overall accuracy measure,
# so the accuracy among the predicted positive is almost equivalent to the accuracy among all the cases.

# f1_score is quite higher, because model performs better on the minority class than the majority class,
# which is evidenced by the nearly equivalent accuracy and precision, and much higher recall.

###


# Define the loss matrix
loss_matrix <- matrix(c(0, 1, 5, 0), nrow = 2, byrow = TRUE)

# Predict with the loss matrix
predicted_test_loss <- predict(optimal_tree_model_deviance, newdata = test_data, type = "class", loss.matrix = loss_matrix)

# Compute the confusion matrix
cm_test_loss <- confusionMatrix(predicted_test_loss, test_data$y)

loss_accuracy_score <- cm_test_loss$overall["Accuracy"]
loss_precision_score <- cm_test_loss$byClass["Pos Pred Value"]
loss_recall_score <- cm_test_loss$byClass["Sensitivity"]
loss_f1_score <- 2 * (cm_test_loss$byClass["Sensitivity"] * cm_test_loss$byClass["Pos Pred Value"]) / (cm_test_loss$byClass["Sensitivity"] + cm_test_loss$byClass["Pos Pred Value"])


score_comparsion_df <- data.frame(scores=c("without loss matrix", "with loss matrix"),
                                  accuracy_scores=c(accuracy_score, loss_accuracy_score),
                                  precision_scores=c(precision_score, loss_precision_score),
                                  recall_scores=c(recall_score, loss_recall_score),
                                  f1_scores=c(f1_score,loss_f1_score))

print(score_comparsion_df)


# Fit a logistic regression model
logreg_model <- glm(y ~ ., data = train_data, family = binomial())

predicted_test_optimal <- predict(optimal_tree_model_deviance, newdata = test_data, type = "class")
predicted_test_logreg <- predict(fit_logistic, newdata = test_data, type = "response") > 0.5


# Define the threshold values
thresholds <- seq(0.05, 0.95, 0.05)

# Initialize vectors to store the TPR and FPR values
tpr_tree <- numeric(length(thresholds))
fpr_tree <- numeric(length(thresholds))
tpr_logistic <- numeric(length(thresholds))
fpr_logistic <- numeric(length(thresholds))


# Compute the TPR and FPR values for each threshold
for (i in seq_along(thresholds)) {
  threshold <- thresholds[i]
  predicted_test_optimal_thresholded <- ifelse(predict(optimal_tree_model_deviance, newdata = test_data)[,2] > threshold, "yes", "no")
  predicted_test_logreg_thresholded <- ifelse(predict(logreg_model, newdata = test_data, type = "response") > threshold, "yes", "no")

  #tpr_values[i] <- sum(predicted_test_optimal_thresholded == 1 & test_data$y == 1) / sum(test_data$y == 1)
  #fpr_values[i] <- sum(predicted_test_optimal_thresholded == 1 & test_data$y == 0) / sum(test_data$y == 0)

  #predictionsTree <- ifelse(predict(fit, testData, type = "prob")[, "yes"] > thresholds[i], "yes", "no")
  #predictionsLogistic <- ifelse(predict(fitLogistic, testData, type = "response") > thresholds[i], "yes", "no")
  all_level <- c("no","yes")
  cm_tree <- table(factor(predicted_test_optimal_thresholded, levels = all_level), test_data$y)
  cm_logistic <- table(factor(predicted_test_logreg_thresholded, levels = all_level), test_data$y)

  # tpr = tp/(tp+fn)
  # fpr = fp/(fp+tn)
  tpr_tree[i] <- cm_tree[[1,1]] / (cm_tree[[1,1]] + cm_tree[[2,1]])
  fpr_tree[i] <- cm_tree[[1,2]] / (cm_tree[[1,2]] + cm_tree[[2,2]])
  tpr_logistic[i] <- cm_logistic[[1,1]] / (cm_logistic[[1,1]] + cm_logistic[[2,1]])
  fpr_logistic[i] <- cm_logistic[[1,2]] / (cm_logistic[[1,2]] + cm_logistic[[2,2]])
}


# Plot the ROC curves
plot(fpr_tree, tpr_tree, type = "l", xlab = "FPR", ylab = "TPR", main = "ROC Curves")
lines(fpr_logistic, tpr_logistic, col = "red")
legend("bottomright", legend = c("Tree", "Logistic"), col = c("black", "red"), lty = 1)


