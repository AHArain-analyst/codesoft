#Importing Libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(tidyverse)

# Load the Titanic data set
Titanic <- read.csv("C:\\Users\\user\\Desktop\\DSF Project\\Titanic.csv")



# Data pre-processing

# Examining the data set
# Number of rows and columns
dim(Titanic)

# Displaying the first few rows of the data
head(Titanic) 

# Displaying the column names
colnames(Titanic) 

# Counting the number of missing values
sum(is.na(Titanic))

# There are a large number of missing values (87) compared to the dataset, so we can't omit them.
# Counting the number of missing values in each column
colSums(is.na(Titanic))

# We can see that the number of missing values for the "Age" column (86) is relatively high.
# Replacing missing values in the "Age" column with the median, ignoring NA values
Titanic$Age[is.na(Titanic$Age)] <- median(Titanic$Age, na.rm = TRUE)

# Removing rows with any missing values
Titanic <- na.omit(Titanic)

# Checking the number of missing values after removing them
colSums(is.na(Titanic))

# Displaying the structure of the modified data set
str(Titanic)

# Displaying summary statistics of the modified data set
summary(Titanic)



# Prepare the Data for Modeling
# Select the relevant features and target variable
features <- Titanic %>%
  select(Age, Sex, Pclass, SibSp, Survived ,Parch)

# Convert categorical variables to factor
features$Sex <- as.factor(features$Sex)

# Convert categorical variables to dummy variables
features <- features %>%
  mutate(Sex = as.numeric(factor(Sex, levels = c("female", "male"))))

# Step 3: Split the Data into Training and Testing Sets
set.seed(123)
train_indices <- sample(nrow(features), nrow(features) * 0.7)
train_data <- features[train_indices, ]
test_data <- features[-train_indices, ]
train_target <- Titanic$Survived[train_indices]
test_target <- Titanic$Survived[-train_indices]



# Logistic Regression
logistic_model <- glm(Survived ~ ., data = train_data, family = "binomial", maxit = 100)
logistic_predictions <- predict(logistic_model, newdata = test_data, type = "response")
logistic_predictions <- ifelse(logistic_predictions > 0.5, 1, 0)

# Calculate the confusion matrix
confusion_matrix_log <- table(logistic_predictions, test_target)
# Calculate accuracy
log_accuracy <- sum(diag(confusion_matrix_log)) / sum(confusion_matrix_log)

# Print the evaluation metric
print(paste("Accuracy:", log_accuracy))
summary(logistic_model)

# Create a data frame with predicted and actual values
plot_data <- data.frame(Actual = test_target, Predicted = logistic_predictions)

# Create a count plot
ggplot(Titanic, aes(x = factor(Pclass), fill = factor(Survived))) +
  geom_bar() +
  labs(x = "Pclass", y = "Count") +
  ggtitle("Count of Passengers by Pclass and Survival in Titanic Dataset") +
  scale_fill_manual(values = c("#FF0000", "#0000FF"), labels = c("No", "Yes"))

# Create a box plot
ggplot(Titanic, aes(x = factor(Survived), y = Age, fill = factor(Survived))) +
  geom_boxplot() +
  labs(x = "Survived", y = "Age") +
  ggtitle("Distribution of Age by Survival in Titanic Dataset") +
  scale_fill_manual(values = c("#FF0000", "#0000FF"), labels = c("No", "Yes"))



# Linear Regression
linear_model <- lm(Survived ~ ., data = train_data)
linear_predictions <- predict(logistic_model, newdata = test_data, type = "response")
linear_predictions <- ifelse(logistic_predictions > 0.5, 1, 0)

# Evaluate the model
mse <- mean((test_target - linear_predictions)^2)
rmse <- sqrt(mse)
r_squared <- cor(test_target, linear_predictions)^2

# Print the evaluation metrics
print(paste("Mean Squared Error (MSE):", mse))
print(paste("Root Mean Squared Error (RMSE):", rmse))
print(paste("R-squared:", r_squared))

# Display the summary of the linear regression model
summary(linear_model)

# Create a data frame with predicted and actual values
plot_data <- data.frame(Actual = test_target, Predicted = linear_predictions)

# Create a scatter plot with a best-fit line
ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  labs(x = "Actual Fare", y = "Predicted Fare") +
  ggtitle("Linear Regression Model: Predicted vs Actual Fare")

# Calculate accuracy
lin_accuracy <- mean(linear_predictions == test_target)


# K-Means Clustering
k <- 3  # Number of clusters
kmeans_model <- kmeans(features, centers = k)
kmeans_clusters <- kmeans_model$cluster
kmeans_y_pred_class <- ifelse(kmeans_clusters == 2, 0, 1)

# Add cluster labels to the Titanic dataset
Titanic$cluster <- as.factor(kmeans_clusters)

# Add cluster labels to the original dataset
kmeans_y_pred_class <- ifelse(kmeans_clusters == 2, 0, 1)

# Ensure that kmeans_y_pred_class and test_target have the same length
if (length(kmeans_y_pred_class) != length(test_target)) {
  # Adjust the length of kmeans_y_pred_class or test_target as needed
  kmeans_y_pred_class <- kmeans_y_pred_class[1:length(test_target)]
}

# Plot the clusters
ggplot(Titanic, aes(x = Age, y = Fare, color = cluster)) +
  geom_point() +
  labs(x = "Age", y = "Fare") +
  ggtitle("K-means Clustering: Age vs Fare")

# Plot histogram of age
ggplot(Titanic, aes(x = Age)) +
  geom_histogram(binwidth = 5, fill = "skyblue", color = "black") +
  labs(x = "Age", y = "Count") +
  ggtitle("Histogram of Age")

# Plot box plot of age by passenger class
ggplot(Titanic, aes(x = factor(Survived), y = Age)) +
  geom_boxplot(fill = "skyblue", color = "black") +
  labs(x = "Survived", y = "Age") +
  ggtitle("Box Plot of Age by Survived")

# Calculate accuracy
kmeans_accuracy <- mean(kmeans_y_pred_class == test_target)



# Model Comparison Visualization
model_names <- c("Logistic Regression", "Linear Regression", "K-Means Clustering")
accuracies <- c(log_accuracy, lin_accuracy, kmeans_accuracy)

result_df <- data.frame(Model = model_names, Accuracy = accuracies)

ggplot(result_df, aes(x = Model, y = Accuracy)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Model Comparison", y = "Accuracy", x = "Model") +
  theme_minimal()
theme_minimal()
