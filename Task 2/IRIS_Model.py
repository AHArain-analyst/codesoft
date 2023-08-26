import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from scipy import interp

# Load the Iris dataset from a CSV file
iris_data = pd.read_csv('D:\\Data Analyst\\CodSoft\\Task 2\\IRIS.csv')

# Split the data into features (X) and target labels (y)
X = iris_data.drop('Species', axis=1)
y = iris_data['Species']

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Train a Support Vector Machine (SVM) model
svm_model = SVC(kernel='linear', C=1, probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Calculate the confusion matrix for SVM
confusion_matrix_svm = confusion_matrix(y_test, y_pred_svm)

# Plot a confusion matrix heatmap for SVM
plt.figure()
sns.heatmap(confusion_matrix_svm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('SVM Confusion Matrix')
plt.show()

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=123)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Calculate the confusion matrix for Random Forest
confusion_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Plot a confusion matrix heatmap for Random Forest
plt.figure()
sns.heatmap(confusion_matrix_rf, annot=True, fmt="d", cmap="Greens", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Random Forest Confusion Matrix')
plt.show()

# Create ROC curve for SVM
y_score_svm = svm_model.decision_function(X_test)
fpr_svm = dict()
tpr_svm = dict()
roc_auc_svm = dict()
for i, class_name in enumerate(np.unique(y)):
    fpr_svm[i], tpr_svm[i], _ = roc_curve(y_test == class_name, y_score_svm[:, i])
    roc_auc_svm[i] = auc(fpr_svm[i], tpr_svm[i])

# Plot ROC curve for SVM
plt.figure()
colors = ['blue', 'red', 'green']
for i, color in zip(range(len(np.unique(y))), colors):
    plt.plot(fpr_svm[i], tpr_svm[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(np.unique(y)[i], roc_auc_svm[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for SVM')
plt.legend(loc="lower right")
plt.show()

# Create ROC curve for Random Forest
y_score_rf = rf_model.predict_proba(X_test)
fpr_rf = dict()
tpr_rf = dict()
roc_auc_rf = dict()
for i, class_name in enumerate(np.unique(y)):
    fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test == class_name, y_score_rf[:, i])
    roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])

# Plot ROC curve for Random Forest
plt.figure()
for i, color in zip(range(len(np.unique(y))), colors):
    plt.plot(fpr_rf[i], tpr_rf[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(np.unique(y)[i], roc_auc_rf[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.legend(loc="lower right")
plt.show()
