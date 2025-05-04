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

# Step 1: Load Dataset
# Replace 'your_dataset.csv' with your dataset file
# For classification: target is categorical (e.g., 0/1, 'Yes'/'No')
# For regression: target is continuous (e.g., price, score)
df = pd.read_csv('your_dataset.csv')  # Example: df = sns.load_dataset('titanic')

# Step 2: Exploratory Data Analysis (EDA)
# Check missing values
print("Missing Values:\n", df.isnull().sum())

# Define features and target
# Replace with your feature columns and target column
numerical_features = ['age', 'fare']  # Example numerical features
categorical_features = ['gender', 'embarked']  # Example categorical features
target = 'survived'  # Example target (classification: 'survived', regression: 'price')

# Histogram for numerical features
df[numerical_features].hist(bins=20, figsize=(8, 4))
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Boxplot for numerical features by target (for classification)
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=target, y=feature, data=df)
    plt.title(f'{feature} by {target}')
    plt.show()

# Count plot for categorical features by target (for classification)
for feature in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=feature, hue=target, data=df)
    plt.title(f'{feature} by {target}')
    plt.show()

# Correlation matrix for numerical features
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# EDA Insights (customize based on observations)
print("EDA Insights:")
print("- Check for skewed distributions in histograms.")
print("- Look for outliers in boxplots.")
print("- Note categorical feature imbalances in count plots.")
print("- Identify strong correlations in the matrix.")

# Step 3: Data Preprocessing
# Handle missing values
for feature in numerical_features:
    df[feature] = df[feature].fillna(df[feature].mean())  # Fill numerical with mean
for feature in categorical_features:
    df[feature] = df[feature].fillna(df[feature].mode()[0])  # Fill categorical with mode

# Encode categorical features
le = LabelEncoder()
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])  # Label Encoding for binary/multiclass

# Optional: One-Hot Encoding for multiclass categorical features
# df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

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
model = DecisionTreeClassifier(random_state=42)  # Decision Tree
# model = SVC(kernel='rbf', C=1, gamma='scale', probability=True)  # SVM with RBF kernel
# model = GaussianNB()  # Naive Bayes
# model = LogisticRegression(random_state=42)  # Logistic Regression
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
    # For regression: cv_accuracy.append(r2_score(y_val, y_pred))
    if hasattr(model, 'predict_proba'):  # For classification with probabilities
        y_proba = model.predict_proba(X_val)[:, 1]
        cv_roc_auc.append(roc_auc_score(y_val, y_proba))

print("K-Fold CV Average Accuracy:", np.mean(cv_accuracy))
if cv_roc_auc:
    print("K-Fold CV Average ROC-AUC:", np.mean(cv_roc_auc))

# Step 8: Leave-One-Out Cross-Validation (LOOCV, optional)
# Use for small datasets; computationally expensive
loo = LeaveOneOut()
loo_scores = []
for train_idx, test_idx in loo.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[test_idx]
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    loo_scores.append(accuracy_score(y_val, y_pred))  # For classification
    # For regression: loo_scores.append(r2_score(y_val, y_pred))
print("LOOCV Average Accuracy:", np.mean(loo_scores))

# Step 9: Test Set Evaluation
y_pred_test = model.predict(X_test)
if hasattr(model, 'predict_proba'):  # For classification
    y_proba_test = model.predict_proba(X_test)[:, 1]
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_roc_auc = roc_auc_score(y_test, y_proba_test)
    cm = confusion_matrix(y_test, y_pred_test)
    print("Test Set Accuracy:", test_accuracy)
    print("Test Set ROC-AUC:", test_roc_auc)
    print("Confusion Matrix:\n", cm)
else:  # For regression
    test_r2 = r2_score(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    print("Test Set R-squared:", test_r2)
    print("Test Set MSE:", test_mse)

# Step 10: Plot ROC Curve (for classification only)
if hasattr(model, 'predict_proba'):
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

# Step 11: Save Preprocessed Data (optional)
df.to_csv('preprocessed_data.csv', index=False)
print("Preprocessed data saved to 'preprocessed_data.csv'")

# SOLVED TASK 1: House Price Prediction
# Step 1: Simulate Dataset
data = {
    'square_footage': np.random.uniform(1000, 4000, 1000),
    'bedrooms': np.random.randint(1, 6, 1000),
    'bathrooms': np.random.uniform(1, 4, 1000),
    'age': np.random.uniform(0, 50, 1000),
    'neighborhood': np.random.choice(['Urban', 'Suburban', 'Rural'], 1000),
    'price': np.random.uniform(100000, 1000000, 1000)
}
df = pd.DataFrame(data)
# Introduce 5% missing values
df.loc[np.random.choice(df.index, 50), 'square_footage'] = np.nan
df.loc[np.random.choice(df.index, 50), 'age'] = np.nan
df.loc[np.random.choice(df.index, 50), 'neighborhood'] = np.nan

# Step 2: Exploratory Data Analysis (EDA)
# Check missing values
print("Missing Values:\n", df.isnull().sum())

# Define features and target
numerical_features = ['square_footage', 'bedrooms', 'bathrooms', 'age']
categorical_features = ['neighborhood']
target = 'price'

# Histogram for numerical features
df[numerical_features].hist(bins=20, figsize=(8, 4))
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Correlation matrix for numerical features
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_features + [target]].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# EDA Insights
print("EDA Insights:")
print("- Square footage and bathrooms likely have higher correlation with price.")
print("- Age may have negative correlation with price.")
print("- Neighborhood distribution should be checked for balance.")

# Step 3: Data Preprocessing
# Handle missing values
for feature in numerical_features:
    df[feature] = df[feature].fillna(df[feature].mean())
for feature in categorical_features:
    df[feature] = df[feature].fillna(df[feature].mode()[0])

# One-Hot Encoding for neighborhood
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Update features after one-hot encoding
categorical_features = [col for col in df.columns if col.startswith('neighborhood_')]
X = df[numerical_features + categorical_features]
y = df[target]

# Scale numerical features
scaler = StandardScaler()
X = X.copy()  # Fix: Avoid SettingWithCopyWarning
X.loc[:, numerical_features] = scaler.fit_transform(X[numerical_features])

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Selection
model = LinearRegression()

# Step 6: Train Model
model.fit(X_train, y_train)

# Step 7: K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracy = []
for train_idx, val_idx in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    cv_accuracy.append(r2_score(y_val, y_pred))  # R-squared for regression
print("K-Fold CV Average R-squared:", np.mean(cv_accuracy))

# Step 8: Test Set Evaluation
y_pred_test = model.predict(X_test)
test_r2 = r2_score(y_test, y_pred_test)
test_mse = mean_squared_error(y_test, y_pred_test)
print("Test Set R-squared:", test_r2)
print("Test Set MSE:", test_mse)

# Step 9: Predict for New House
new_house = pd.DataFrame({
    'square_footage': [2000],
    'bedrooms': [3],
    'bathrooms': [2],
    'age': [10],
    'neighborhood_Suburban': [0],
    'neighborhood_Urban': [1]
})
new_house[numerical_features] = scaler.transform(new_house[numerical_features])
predicted_price = model.predict(new_house)
print("Predicted Price for New House:", predicted_price[0])

# Step 10: Save Preprocessed Data
df.to_csv('preprocessed_houses.csv', index=False)
print("Preprocessed data saved to 'preprocessed_houses.csv'")

# SOLVED TASK 2: Email Spam Classification
# Step 1: Simulate Dataset
data = {
    'word_freq': np.random.uniform(0, 0.5, 1000),
    'email_length': np.random.randint(50, 1000, 1000),
    'has_hyperlink': np.random.choice([0, 1], 1000),
    'sender_domain': np.random.choice(['Free', 'Paid'], 1000),
    'spam': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
}
df = pd.DataFrame(data)
# Introduce 5% missing values
df.loc[np.random.choice(df.index, 50), 'word_freq'] = np.nan
df.loc[np.random.choice(df.index, 50), 'sender_domain'] = np.nan

# Step 2: Exploratory Data Analysis (EDA)
# Check missing values
print("Missing Values:\n", df.isnull().sum())

# Define features and target
numerical_features = ['word_freq', 'email_length']
categorical_features = ['has_hyperlink', 'sender_domain']
target = 'spam'

# Histogram for numerical features
df[numerical_features].hist(bins=20, figsize=(8, 4))
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Boxplot for numerical features by target
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=target, y=feature, data=df)
    plt.title(f'{feature} by {target}')
    plt.show()

# Count plot for categorical features by target
for feature in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=feature, hue=target, data=df)
    plt.title(f'{feature} by {target}')
    plt.show()

# Correlation matrix for numerical features
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# EDA Insights
print("EDA Insights:")
print("- High word frequency may indicate spam.")
print("- Emails with hyperlinks are more likely spam.")
print("- Free sender domains may correlate with spam.")

# Step 3: Data Preprocessing
# Handle missing values
for feature in numerical_features:
    df[feature] = df[feature].fillna(df[feature].mean())
for feature in categorical_features:
    df[feature] = df[feature].fillna(df[feature].mode()[0])

# Encode sender_domain
le = LabelEncoder()
df['sender_domain'] = le.fit_transform(df['sender_domain'])  # Free=0, Paid=1

# Features and target
X = df[numerical_features + categorical_features]
y = df[target]

# Scale numerical features
scaler = StandardScaler()
X = X.copy()  # Fix: Avoid SettingWithCopyWarning
X.loc[:, numerical_features] = scaler.fit_transform(X[numerical_features])

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Selection
model = DecisionTreeClassifier(random_state=42)

# Step 6: Train Model
model.fit(X_train, y_train)

# Step 7: K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracy = []
cv_roc_auc = []
for train_idx, val_idx in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    cv_accuracy.append(accuracy_score(y_val, y_pred))
    y_proba = model.predict_proba(X_val)[:, 1]
    cv_roc_auc.append(roc_auc_score(y_val, y_proba))
print("K-Fold CV Average Accuracy:", np.mean(cv_accuracy))
print("K-Fold CV Average ROC-AUC:", np.mean(cv_roc_auc))

# Step 8: Test Set Evaluation
y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)[:, 1]
test_accuracy = accuracy_score(y_test, y_pred_test)
test_roc_auc = roc_auc_score(y_test, y_proba_test)
cm = confusion_matrix(y_test, y_pred_test)
print("Test Set Accuracy:", test_accuracy)
print("Test Set ROC-AUC:", test_roc_auc)
print("Confusion Matrix:\n", cm)

# Step 9: Plot ROC Curve
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

# Step 10: Predict for New Email
new_email = pd.DataFrame({
    'word_freq': [0.1],
    'email_length': [200],
    'has_hyperlink': [1],
    'sender_domain': [0]  # Free
})
new_email[numerical_features] = scaler.transform(new_email[numerical_features])
predicted_spam = model.predict(new_email)
print("Predicted Spam (1=Spam, 0=Not Spam):", predicted_spam[0])

# Step 11: Save Preprocessed Data
df.to_csv('preprocessed_emails.csv', index=False)
print("Preprocessed data saved to 'preprocessed_emails.csv'")

# SOLVED TASK 3: Customer Value Classification
# Step 1: Simulate Dataset
data = {
    'total_spending': np.random.uniform(100, 5000, 1000),
    'age': np.random.uniform(18, 80, 1000),
    'visits': np.random.randint(1, 50, 1000),
    'purchase_frequency': np.random.uniform(0.1, 5, 1000),
    'value': np.random.choice([0, 1], 1000, p=[0.6, 0.4])
}
df = pd.DataFrame(data)
# Introduce 5% missing values
df.loc[np.random.choice(df.index, 50), 'total_spending'] = np.nan
df.loc[np.random.choice(df.index, 50), 'visits'] = np.nan

# Step 2: Exploratory Data Analysis (EDA)
# Check missing values
print("Missing Values:\n", df.isnull().sum())

# Define features and target
numerical_features = ['total_spending', 'age', 'visits', 'purchase_frequency']
categorical_features = []
target = 'value'

# Histogram for numerical features
df[numerical_features].hist(bins=20, figsize=(8, 4))
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Boxplot for numerical features by target
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=target, y=feature, data=df)
    plt.title(f'{feature} by {target}')
    plt.show()

# Correlation matrix for numerical features
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# EDA Insights
print("EDA Insights:")
print("- High total spending and purchase frequency likely indicate high-value customers.")
print("- Age may have less impact based on boxplots.")
print("- Check for outliers in spending and visits.")

# Step 3: Data Preprocessing
# Handle outliers (cap at 99th percentile)
for feature in numerical_features:
    df[feature] = df[feature].clip(upper=df[feature].quantile(0.99))

# Handle missing values
for feature in numerical_features:
    df[feature] = df[feature].fillna(df[feature].mean())

# Features and target
X = df[numerical_features]
y = df[target]

# Scale numerical features
scaler = StandardScaler()
X = X.copy()  # Fix: Avoid SettingWithCopyWarning
X.loc[:, numerical_features] = scaler.fit_transform(X[numerical_features])

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Selection (SVM for hyperplane)
model = SVC(kernel='rbf', C=1, gamma='scale', probability=True)

# Step 6: Train Model
model.fit(X_train, y_train)

# Step 7: K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracy = []
cv_roc_auc = []
for train_idx, val_idx in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    cv_accuracy.append(accuracy_score(y_val, y_pred))
    y_proba = model.predict_proba(X_val)[:, 1]
    cv_roc_auc.append(roc_auc_score(y_val, y_proba))
print("K-Fold CV Average Accuracy:", np.mean(cv_accuracy))
print("K-Fold CV Average ROC-AUC:", np.mean(cv_roc_auc))

# Step 8: Test Set Evaluation
y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)[:, 1]
test_accuracy = accuracy_score(y_test, y_pred_test)
test_roc_auc = roc_auc_score(y_test, y_proba_test)
cm = confusion_matrix(y_test, y_pred_test)
print("Test Set Accuracy:", test_accuracy)
print("Test Set ROC-AUC:", test_roc_auc)
print("Confusion Matrix:\n", cm)

# Step 9: Plot ROC Curve
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

# Step 10: Decision Tree for Rules
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
print("Feature Importances (Rules):")
for feature, importance in zip(numerical_features, dt_model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# Step 11: Save Preprocessed Data
df.to_csv('preprocessed_customers.csv', index=False)
print("Preprocessed data saved to 'preprocessed_customers.csv'")