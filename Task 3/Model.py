import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load and Prepare Data
data = pd.read_csv('D:\\Data Analyst\\CodSoft\\Task 3\\advertising.csv') 

# Step 2: Data Wrangling and Cleaning
# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Handle missing values (if necessary)
# Example: data['Feature'].fillna(data['Feature'].mean(), inplace=True)

# Step 3: Data Analysis and Visualization
# Explore the data
print(data.describe())  # Summary statistics
print(data.head())      # Display the first few rows

# Data Visualization
# Pairplot to visualize relationships between variables
sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=1)
plt.suptitle("Pairplot of Sales vs. Advertising Channels", y=1.02)
plt.show()

# Heatmap to visualize correlations
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Feature Selection
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Visualization of Predicted vs. Actual Sales
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual Sales vs. Predicted Sales")
plt.show()

# Use the Model for Predictions (Example)
new_data = pd.DataFrame({
    'TV': [100],
    'Radio': [25],
    'Newspaper': [10]
})

predicted_sales = model.predict(new_data)
print("\nPredicted Sales:", predicted_sales[0])
