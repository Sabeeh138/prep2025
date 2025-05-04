"""Practice Question: Task 4 – Credit Risk Classification
Objective: Develop a supervised learning model to predict whether a customer is a high-risk borrower (1=high risk, 0=low risk)
based on their financial and personal characteristics. Use a simulated dataset representing customer data and follow the supervised learning 
pipeline as outlined in Lab 09.

Dataset:

Simulated Dataset: Generate a dataset with 1000 customers containing the following columns:
income: Annual income in dollars (numerical, uniform between 20,000 and 100,000).
credit_score: Credit score (numerical, uniform between 300 and 850).
loan_amount: Requested loan amount in dollars (numerical, uniform between 5,000 and 50,000).
employment_status: Employment type (categorical: Employed, Unemployed, Self-Employed).
risk: Risk level (target, categorical: 0=low risk, 1=high risk, with 70% low risk, 30% high risk).
Missing Values: Introduce 5% missing values in income, credit_score, and employment_status.
Note: Simulate the dataset using numpy.random with random_state=42 for reproducibility.
Requirements:

Exploratory Data Analysis (EDA):
Report missing values for all features.
Plot histograms for numerical features (income, credit_score, loan_amount).
Create boxplots for numerical features by risk to identify differences.
Plot a count plot for employment_status by risk.
Compute and visualize a correlation matrix for numerical features.
Provide at least three insights based on the EDA (e.g., “Low credit scores may correlate with high risk”).
Data Preprocessing:
Fill missing values: Use mean for numerical features (income, credit_score), mode for categorical (employment_status).
Encode employment_status using One-Hot Encoding.
Scale numerical features (income, credit_score, loan_amount) using StandardScaler.
Use an 80-20 train-test split (test_size=0.2).
Model Training:
Train a LogisticRegression model (Lab 09, Part 2, Page 7).
Perform 5-fold cross-validation on the training set to report average accuracy and ROC-AUC.
Evaluation:
Evaluate the model on the test set using accuracy, ROC-AUC, and confusion matrix.
Plot the ROC curve with AUC value displayed.
Prediction:
Predict the risk for a new customer with: income=50,000, credit_score=600, loan_amount=20,000, employment_status=Employed.
Deliverables:
Save the preprocessed dataset as preprocessed_credit.csv.
Print EDA insights, CV results, test set metrics, confusion matrix, and prediction result.
Display the ROC curve plot with proper labels and title.
Constraints:

Follow Lab 09 guidelines (Part 1, Pages 4–15; Part 2, Pages 5–10).
Ensure no missing values in the final dataset used for modeling.
Use random_state=42 for all random operations (data generation, splitting, model).
Include axis labels and titles for all plots (Part 1, Page 4).
Use only techniques and models explicitly mentioned in the Lab 09 manual (e.g., no feature engineering unless stated)."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# General Supervised Learning Template for Lab 09
# Set random seed for reproducibility
np.random.seed(42)
data = {
    'income': np.random.uniform(20000, 100000, 1000),
    'credit_score': np.random.randint(300, 850, 1000),
    'loan_amount': np.random.uniform(5000,50000,1000),
    'employment_status': np.random.choice(['Employed', 'Unemployed','Self-Employed'], 1000),
    'risk': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
}
df = pd.DataFrame(data)

# Introduce 5% missing values
df.loc[np.random.choice(df.index, 50), 'income'] = np.nan #1000 5% is 50
df.loc[np.random.choice(df.index, 50), 'credit_score'] = np.nan
df.loc[np.random.choice(df.index, 50), 'employment_status'] = np.nan

# Step 1: Load Dataset
# Replace 'your_dataset.csv' with your dataset file
# For classification: target is categorical (e.g., 0/1, 'Yes'/'No')
# For regression: target is continuous (e.g., price, score)
#df = pd.read_csv('your_dataset.csv')  # Example: df = sns.load_dataset('titanic')

# Step 2: Exploratory Data Analysis (EDA)
# Check missing values
print("Missing Values:\n", df.isnull().sum())

# Define features and target
# Replace with your feature columns and target column
numerical_features = ['income', 'credit_score','loan_amount']  # Example numerical features
categorical_features = ['employment_status',]  # Example categorical features
target = 'risk'  # Example target (classification: 'survived', regression: 'price')

# Histogram for numerical features
df[numerical_features].hist(bins=20, figsize=(8, 4))
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Boxplot for numerical features by target (for classification)
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=target, y=feature, data=df)
    plt.title(f'{feature} by risk')
    plt.show()

# Count plot for categorical features by target (for classification)
for feature in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=feature, hue=target, data=df)
    plt.title(f'{feature} by risk')
    plt.show()

# Correlation matrix for numerical features
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

print("EDA Insights:")
print("- Income, credit score, and loan amount are uniformly distributed across their ranges (histograms).")
print("- Numerical features (income, credit score, loan amount) show no clear difference between low-risk and high-risk groups (boxplots).")
print("- The lack of variation in numerical features by risk suggests they may have limited predictive power for this classification task.")

# Step 3: Data Preprocessing
# Handle missing values
for feature in numerical_features:
    df[feature] = df[feature].fillna(df[feature].mean())  # Fill numerical with mean
for feature in categorical_features:
    df[feature] = df[feature].fillna(df[feature].mode()[0])  # Fill categorical with mode

# Encode categorical features
#le = LabelEncoder()
#for feature in categorical_features:
#   df[feature] = le.fit_transform(df[feature])  # Label Encoding for binary/multiclass

# Optional: One-Hot Encoding for multiclass categorical features
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

categorical_features = [col for col in df.columns if col.startswith('employment_status')]

# Features and target
X = df[numerical_features + categorical_features]
y = df[target]

# Scale numerical features
scaler = StandardScaler()
X = X.copy()  # Fix: Avoid SettingWithCopyWarning
X.loc[:, numerical_features] = scaler.fit_transform(X[numerical_features])

# Step 4: Train-Test Split
# 80-20 split (adjust test_size if needed, e.g., 0.3 for 70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Selection
# Choose one model based on task (uncomment the desired model)
# For classification:
# model = DecisionTreeClassifier(random_state=42)  # Decision Tree
# model = SVC(kernel='rbf', C=1, gamma='scale', probability=True)  # SVM with RBF kernel
# model = GaussianNB()  # Naive Bayes
model = LogisticRegression(random_state=42)  # Logistic Regression
# For regression:
# model = LinearRegression()  # Linear Regression

# Step 6: Train Model
model.fit(X_train, y_train)

# Step 7: K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracy = []
cv_roc_auc = []  # For classification only
for train_idx, val_idx in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    cv_accuracy.append(accuracy_score(y_val, y_pred))  # For classification
    y_proba = model.predict_proba(X_val)[:, 1]
    cv_roc_auc.append(roc_auc_score(y_val, y_proba))

print("K-Fold CV Average Accuracy:", np.mean(cv_accuracy))
print("K-Fold CV Average ROC-AUC:", np.mean(cv_roc_auc))

# Step 9: Test Set Evaluation
y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)[:, 1]
test_accuracy = accuracy_score(y_test, y_pred_test)
test_roc_auc = roc_auc_score(y_test, y_proba_test)
cm = confusion_matrix(y_test, y_pred_test)
print("Test Set Accuracy:", test_accuracy)
print("Test Set ROC-AUC:", test_roc_auc)
print("Confusion Matrix:\n", cm)

# Step 10: Plot ROC Curve (for classification only)

fpr, tpr, _ = roc_curve(y_test, y_proba_test)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {test_roc_auc:.2f})')
plt.fill_between(fpr, tpr, color='skyblue', alpha=0.4)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Test Set)')
plt.legend(loc='lower right')
plt.show()


new_risk = pd.DataFrame({
    'income': [50000],
    'credit_score': [600],
    'loan_amount': [20000],
    'employment_status_Self-Employed': [0],
    'employment_status_Unemployed': [0]
})
new_risk[numerical_features] = scaler.transform(new_risk[numerical_features])
predicted_risk = model.predict(new_risk)
print("Predicted RISK (1=risk, 0=Not risk):", predicted_risk[0])


# Step 11: Save Preprocessed Data (optional)
df.to_csv('preprocessed_data.csv', index=False)
print("Preprocessed data saved to 'preprocessed_data.csv'")
