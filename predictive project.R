#CA-3 PROJECT WORK 
#.............. DATASET.....................
# Install and load necessary libraries
install.packages(c("e1071", "class", "rpart", "ggplot2", "caTools"))
# Load required libraries
# Load required libraries
library(caret)
library(class)
library(e1071)
library(rpart)
library(ggplot2)
library(dplyr)
library(tidyr)

# Load the dataset
data <- read.csv(file.choose())

# Preprocessing
data$diagnosis <- as.factor(ifelse(data$diagnosis == "M", 1, 0))  # Convert to binary
X <- data[, !(names(data) %in% c("id", "diagnosis"))]  # Features
y <- data$diagnosis  # Target

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
trainData
testData <- data[-trainIndex, ]
testData

# Standardize features
scaler <- preProcess(trainData[, -which(names(trainData) == "diagnosis")], method = c("center", "scale"))
trainData[, -which(names(trainData) == "diagnosis")] <- predict(scaler, trainData[, -which(names(trainData) == "diagnosis")])
testData[, -which(names(testData) == "diagnosis")] <- predict(scaler, testData[, -which(names(testData) == "diagnosis")])

# Function to calculate evaluation metrics
calculate_metrics <- function(predictions, actual) {
  accuracy <- mean(predictions == actual)
  precision <- posPredValue(predictions, actual, positive = "1")
  recall <- sensitivity(predictions, actual, positive = "1")
  f1 <- (2 * precision * recall) / (precision + recall)
  error_rate <- 1 - accuracy
  return(data.frame(Accuracy = accuracy, Precision = precision, Recall = recall, F1_Score = f1, Error_Rate = error_rate))
}

# Models and Evaluation
results <- list()
results

# 1. KNN
# Performs well if the dataset is standardized and has a clear separation between classes.
#Sensitive to the choice of k and the feature scaling.
knn_model <- knn(train = trainData[, -which(names(trainData) == "diagnosis")],
                 test = testData[, -which(names(testData) == "diagnosis")],
                 cl = trainData$diagnosis, k = 5)
results$KNN <- calculate_metrics(knn_model, testData$diagnosis)

# 2. Naive Bayes
#Assumes independence between features.
#Works well for datasets with categorical and numerical features but may struggle with complex relationships.

nb_model <- naiveBayes(diagnosis ~ ., data = trainData)
nb_predictions <- predict(nb_model, testData)
results$Naive_Bayes <- calculate_metrics(nb_predictions, testData$diagnosis)

# 3. Decision Tree
#Handles non-linearity well.
#Risk of overfitting if the tree is too deep.
dt_model <- rpart(diagnosis ~ ., data = trainData, method = "class")
dt_predictions <- predict(dt_model, testData, type = "class")
results$Decision_Tree <- calculate_metrics(dt_predictions, testData$diagnosis)

# 4. Logistic Regression
#Simple yet effective for linearly separable data.
#Struggles with non-linear relationships.
log_reg_model <- glm(diagnosis ~ ., data = trainData, family = binomial)
log_reg_pred <- predict(log_reg_model, testData, type = "response")
log_reg_class <- as.factor(ifelse(log_reg_pred > 0.5, 1, 0))
results$Logistic_Regression <- calculate_metrics(log_reg_class, testData$diagnosis)

# 5. K-Means Clustering
#Primarily an unsupervised algorithm, often less accurate for classification.
kmeans_model <- kmeans(trainData[, -which(names(trainData) == "diagnosis")], centers = 2)
trainData$Cluster <- as.factor(ifelse(kmeans_model$cluster == which.max(table(kmeans_model$cluster)), 1, 0))
results$K_Means <- calculate_metrics(trainData$Cluster, trainData$diagnosis)

# Combine results into a single data frame
results_df <- do.call(rbind, results)
results_df$Model <- rownames(results_df)
results_df <- results_df %>% gather(key = "Metric", value = "Value", -Model)

# Visualization
ggplot(results_df, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  theme_minimal() +
  ggtitle("Model Comparison Across Evaluation Metrics") +
  ylab("Metric Value") +
  xlab("Evaluation Metrics") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

