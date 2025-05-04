import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

# 1. Generate sample data (replace with your actual data)
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Train a Decision Tree classifier (replace with your trained model if you have one)
DT = DecisionTreeClassifier(max_depth=3, random_state=42)
DT.fit(x_train, y_train)

probabilities = DT.predict_proba(x_test)[:, 1]  # Probability estimates for class 1

# 4. Calculate ROC Curve metrics
fpr, tpr, thresholds = roc_curve(y_test, probabilities)

# 5. Calculate ROC AUC Score
roc_auc = roc_auc_score(y_test, probabilities)

# 6. Plot ROC curve with shaded area under the curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.fill_between(fpr, tpr, color='skyblue', alpha=0.4)  # Shade the area under curve
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random classifier
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with AUC Area')
plt.legend(loc='lower right')
plt.show()