import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the titanic set
titanic = pd.read_csv('D:\\Data Analyst\\CodSoft\\Task 1\\tested.csv')

# titanic preprocessing
titanic.fillna(method='ffill', inplace=True)
titanic = pd.get_dummies(titanic, columns=['Sex', 'Pclass', 'Embarked'])
titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
titanic.head(5)

# Split into features (X) and target (y)
X = titanic.drop('Survived', axis=1)
y = titanic['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

# Get feature importance
coef = model.coef_[0]
feature_names = X.columns
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(coef)})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Visualization

# Confusion Matrix
sns.set(font_scale=1.2)
plt.figure(figsize=(6, 5))
sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
coef = model.coef_[0]
feature_names = X.columns
feature_importance = pd.titanicFrame({'Feature': feature_names, 'Importance': coef})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Feature Importance
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Logistic Regression Feature Importance')
plt.show()

# Distribution of Ages for Survived and Not Survived
plt.figure(figsize=(8, 6))
sns.histplot(titanic, x='Age', hue='Survived', element='step', stat='density', common_norm=False, bins=30)
plt.title('Age Distribution by Survival')
plt.legend(labels=['Not Survived', 'Survived'])
plt.show()