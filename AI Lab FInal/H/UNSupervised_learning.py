import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# ------------------------------
# Step 1: Load or Simulate Dataset
# ------------------------------

# Option 1: Load from CSV (uncomment if using a real dataset)
# Replace 'your_dataset.csv' with the actual file name (e.g., 'Mall_Customers.csv')
# df = pd.read_csv('your_dataset.csv')

# Option 2: Simulate data (uncomment and modify for simulated datasets)
# For 100 entries: range(1, 101); for 150 entries: range(1, 151); adjust as needed
data = {
    'id': range(1, 151),  # Identifier (e.g., 'shopper_id', 'student_id'); adjust range (e.g., range(1, 101) for 100 entries)
    'feature1': np.random.uniform(0, 100, 150),  # Numerical feature (e.g., 'average_order_value', range 20-300)
    'feature2': np.random.uniform(0, 50, 150),   # Numerical feature (e.g., 'browsing_time', range 5-60)
    'feature3': np.random.uniform(0, 100, 150),  # Numerical feature (e.g., 'review_rating', range 1-5)
    # 'categorical_feature': np.random.choice(['Category1', 'Category2', 'Category3'], 150),  # Categorical (e.g., 'preferred_device')
}
df = pd.DataFrame(data)

# ------------------------------
# Step 2: Exploratory Data Analysis (EDA)
# ------------------------------

# Check missing values
print("Missing Values:\n", df.isnull().sum())

# Define numerical and categorical features
# Replace with actual feature names (e.g., ['average_order_value', 'browsing_time', 'review_rating'])
numerical_features = ['feature1', 'feature2', 'feature3']
# Uncomment and update if there are categorical features (e.g., ['preferred_device'])
# categorical_features = ['categorical_feature']
# Combine features for clustering (uncomment if categorical features exist)
# features = numerical_features + categorical_features
features = numerical_features  # Default: only numerical features

# Plot histograms for numerical features
df[numerical_features].hist(bins=20, figsize=(8, 4))
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Plot count plots for categorical features (uncomment if applicable)
# if 'categorical_features' in locals():
#     for feature in categorical_features:
#         plt.figure(figsize=(8, 4))
#         sns.countplot(x=feature, data=df)
#         plt.title(f'Distribution of {feature}')
#         plt.xlabel(feature)
#         plt.ylabel('Count')
#         plt.show()

# Plot correlation matrix for numerical features (uncomment if required, e.g., Lab 09 Task 4)
# plt.figure(figsize=(8, 6))
# sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()

# ------------------------------
# Step 3: Data Preprocessing
# ------------------------------

# Handle missing values
# Numerical: fill with mean
for feature in numerical_features:
    df[feature] = df[feature].fillna(df[feature].mean())
# Categorical: fill with mode (uncomment if categorical features exist)
# if 'categorical_features' in locals():
#     for feature in categorical_features:
#         df[feature] = df[feature].fillna(df[feature].mode()[0])

# Encode categorical features (uncomment if applicable)
# if 'categorical_features' in locals():
#     df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
#     # Update features list after encoding
#     features = [col for col in df.columns if col not in ['id']]

# Extract features for clustering
X = df[features].values

# ------------------------------
# Step 4: Feature Scaling
# ------------------------------

# Option 1: Scale all features (default for most tasks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Option 2: Scale specific features (uncomment and modify if needed)
# scale_features = ['feature1', 'feature2']  # Specify features to scale (e.g., exclude 'Age')
# scale_indices = [features.index(f) for f in scale_features]
# X_scaled = X.copy()
# X_scaled[:, scale_indices] = scaler.fit_transform(X[:, scale_indices])

# Option 3: No scaling (uncomment if explicitly stated)
# X_scaled = X

# ------------------------------
# Step 5: Determine Optimal Number of Clusters (Elbow Method)
# ------------------------------

# Set K range (default: 1 to 10; adjust as needed, e.g., range(2, 9) for K=2 to 8)
k_range = range(1, 11)
wcss = []
for i in k_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 4))
plt.plot(k_range, wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()

# ------------------------------
# Step 6: Apply K-Means Clustering
# ------------------------------

# Choose optimal K (replace with elbow point, e.g., 3, 4)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the DataFrame
df['Cluster'] = clusters

# ------------------------------
# Step 7: Visualize Clusters
# ------------------------------

# Choose two features for visualization (replace with actual names, e.g., 'browsing_time', 'average_order_value')
feature_x = 'feature1'
feature_y = 'feature2'

plt.figure(figsize=(8, 6))
for cluster in range(optimal_k):
    plt.scatter(
        df[feature_x][df['Cluster'] == cluster],
        df[feature_y][df['Cluster'] == cluster],
        s=100,
        label=f'Cluster {cluster + 1}'
    )

# Plot centroids (handles both scaled and unscaled cases)
if 'scaler' in locals():
    centroids_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(
        centroids_unscaled[:, features.index(feature_x)],
        centroids_unscaled[:, features.index(feature_y)],
        s=300,
        c='yellow',
        marker='*',
        label='Centroids'
    )
else:
    plt.scatter(
        kmeans.cluster_centers_[:, features.index(feature_x)],
        kmeans.cluster_centers_[:, features.index(feature_y)],
        s=300,
        c='yellow',
        marker='*',
        label='Centroids'
    )

plt.title('Clusters Visualization')  # Customize title (e.g., 'Shopper Clusters')
plt.xlabel(feature_x)  # Customize label (e.g., 'Browsing Time (Minutes)')
plt.ylabel(feature_y)  # Customize label (e.g., 'Average Order Value (Dollars)')
plt.legend()
plt.show()

# ------------------------------
# Step 8: Deliverables
# ------------------------------

# Display final dataset (adjust columns, e.g., ['shopper_id', 'Cluster'])
print("Final Dataset with Clusters:\n", df[['id', 'Cluster']])

# Save to CSV (replace filename, e.g., 'clustered_shoppers.csv')
df.to_csv('clustered_data.csv', index=False)
print("Clustered data saved to 'clustered_data.csv'")

# Comparison Insights (uncomment for scaling vs. no scaling tasks)
# print("Comparison Insights:")
# print("- Without scaling, features with larger ranges dominate clustering.")
# print("- With scaling, clusters are more balanced across all features.")

# ------------------------------
# Solved Example: Online Shoppers Clustering
# ------------------------------

# Simulate dataset
data = {
    'shopper_id': range(1, 151),
    'average_order_value': np.random.uniform(20, 300, 150),
    'browsing_time': np.random.uniform(5, 60, 150),
    'review_rating': np.random.uniform(1, 5, 150),
    'preferred_device': np.random.choice(['Desktop', 'Mobile', 'Tablet'], 150),
}
df = pd.DataFrame(data)

# Define features
numerical_features = ['average_order_value', 'browsing_time', 'review_rating']
categorical_features = ['preferred_device']
features = numerical_features + categorical_features

# EDA (optional)
print("Missing Values:\n", df.isnull().sum())
df[numerical_features].hist(bins=20, figsize=(8, 4))
plt.suptitle('Histograms of Numerical Features')
plt.show()
plt.figure(figsize=(8, 4))
sns.countplot(x='preferred_device', data=df)
plt.title('Distribution of Preferred Device')
plt.xlabel('Preferred Device')
plt.ylabel('Count')
plt.show()

# Preprocessing
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
features = [col for col in df.columns if col not in ['shopper_id']]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Elbow Method
k_range = range(2, 9)
wcss = []
for i in k_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(8, 4))
plt.plot(k_range, wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()

# Clustering
optimal_k = 4  # Adjust based on elbow plot
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# Visualization
feature_x = 'browsing_time'
feature_y = 'average_order_value'
plt.figure(figsize=(8, 6))
for cluster in range(optimal_k):
    plt.scatter(
        df[feature_x][df['Cluster'] == cluster],
        df[feature_y][df['Cluster'] == cluster],
        s=100,
        label=f'Cluster {cluster + 1}'
    )
centroids_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centroids_unscaled[:, features.index(feature_x)],
    centroids_unscaled[:, features.index(feature_y)],
    s=300,
    c='yellow',
    marker='*',
    label='Centroids'
)
plt.title('Shopper Clusters Based on Browsing Time and Average Order Value')
plt.xlabel('Browsing Time (Minutes)')
plt.ylabel('Average Order Value (Dollars)')
plt.legend()
plt.show()

# Deliverables
print("Final Dataset with Clusters:\n", df[['shopper_id', 'Cluster']])
df.to_csv('clustered_shoppers.csv', index=False)
print("Clustered data saved to 'clustered_shoppers.csv'")