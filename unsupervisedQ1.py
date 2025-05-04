"""Practice Question: Clustering Online Shoppers for Marketing Strategy
Objective: An e-commerce company wants to segment its online shoppers to design targeted marketing campaigns. 
The company has collected data on shoppers, including their shopper_id, average_order_value (in dollars), browsing_time (average minutes spent per session), 
review_rating (average rating given, 1–5), and preferred_device (Desktop, Mobile, Tablet). 
The goal is to group shoppers into clusters based on these attributes to identify distinct customer segments for personalized marketing.

Dataset:

Simulated dataset with 150 shoppers.
Columns:
shopper_id: Unique identifier (1 to 150).
average_order_value: Numerical, uniform between 20 and 300 dollars.
browsing_time: Numerical, uniform between 5 and 60 minutes.
review_rating: Numerical, uniform between 1 and 5.
preferred_device: Categorical (Desktop, Mobile, Tablet).
No missing values (to keep it simple for practice).
Requirements:

Feature Selection and Scaling:
Use average_order_value, browsing_time, review_rating, and preferred_device for clustering.
Apply feature scaling to all numerical features (average_order_value, browsing_time, review_rating).
Determine Optimal Number of Clusters (K):
Use the Elbow Method to determine the optimal number of clusters (K) in the range of 2 to 8.
Perform Clustering:
Apply K-Means clustering using the optimal K and assign a cluster label to each shopper.
Visualization:
Create a scatter plot to visualize the clusters using browsing_time (x-axis) and average_order_value (y-axis).
Color each point based on its cluster.
Add centroids and an informative title and labels.
Deliverables:
Display the final dataset showing shopper_id along with their assigned cluster.
Present the scatter plot.
Save the clustered dataset as clustered_shoppers.csv.
Constraints:

Follow Lab 10 guidelines (Sections 4.1, 4.1.1, 4.1.2).
Use random_state=42 for all random operations.
Include axis labels and titles for all plots."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Load or Simulate Dataset
# If the question provides a dataset file (e.g., Mall_Customers.csv), uncomment the line below
# df = pd.read_csv('your_dataset.csv')

# If the question requires simulating data (e.g., Task 3 student data), use the block below
data = {
    'shopper_id': range(1, 151),  # Replace 'id' with the identifier (e.g., student_id, vehicle_serial_no)
    'average_order_value': np.random.uniform(20, 300, 150),  # Replace with your feature (e.g., GPA, mileage)
    'browsing_time': np.random.uniform(5, 60, 150),   # Replace with your feature (e.g., study_hours, fuel_efficiency)
    'review_rating': np.random.uniform(1, 5, 150),  # Replace with your feature (e.g., attendance_rate, maintenance_cost)
    'preferred_device': np.random.choice(['Desktop', 'Mobile', 'Tablet'], 150),  # Uncomment if the dataset has categorical features (e.g., vehicle_type)
}
df = pd.DataFrame(data)

# Step 2: Exploratory Data Analysis (EDA)

# Define features for clustering
# Replace with your features (exclude IDs or non-relevant columns)
numerical_features = ['average_order_value', 'browsing_time', 'review_rating']
# If the dataset has categorical features, include them before encoding
categorical_features = ['preferred_device']
features = numerical_features + categorical_features

# Histogram for numerical features
df[features].hist(bins=20, figsize=(8, 4))
plt.suptitle('Histograms of Features')
plt.show()

# Count plot for categorical features
for feature in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=feature, data=df)
    plt.title('Distribution of Preferred Device')
    plt.xlabel('Preferred Device')
    plt.ylabel('Count')
    plt.show()

# Correlation matrix for numerical features
# If the question asks for a correlation matrix (e.g., Lab 09 Task 4), keep this; otherwise, comment it out
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Step 3: Data Preprocessing

# Encode categorical features (if any)
# If the dataset has categorical features (e.g., vehicle_type, Gender), uncomment the block below
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
# Update features list after encoding
features = [col for col in df.columns if col not in ['id']]  # Exclude ID column

# Extract features for clustering
X = df[features].values

# Step 4: Feature Scaling
# If the question requires scaling (default for most tasks, e.g., Task 3), keep this block
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# If the question requires scaling only specific features (e.g., Task 1: scale all except Age), uncomment the block below
# X_scaled = X.copy()
# scale_features = [f for f in features if f != 'feature_to_exclude']  # Replace 'feature_to_exclude' with the feature to exclude (e.g., Age)
# scale_indices = [features.index(f) for f in scale_features]
# X_scaled[:, scale_indices] = scaler.fit_transform(X[:, scale_indices])

# If the question explicitly says NO scaling (e.g., Task 1/2 first part), use X directly and comment out the scaling block
# X_scaled = X  # Uncomment this line and comment the scaling block above

# Step 5: Determine Optimal Number of Clusters (Elbow Method)
# If the question specifies a K range (e.g., Task 3: K=2 to 6), adjust the range below
wcss = []
for i in range(2, 9):  # Default range 1 to 10; adjust as needed (e.g., range(2, 7) for Task 3)
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)  # Use X_scaled or X based on scaling
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 4))
plt.plot(range(2, 9), wcss, marker='o')  # Adjust range to match the loop above
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()

# Step 6: Apply K-Means Clustering
# Choose optimal K based on elbow plot (replace with your chosen K)
optimal_k = 3  # Example: Replace with the elbow point (e.g., 4 for Task 1, 3 for Task 2/3)
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)  # Use X_scaled or X based on scaling

# Add cluster labels to the DataFrame
df['Cluster'] = clusters

# Step 7: Visualize Clusters
# Choose two features for visualization (replace with the features specified in the question, e.g., Task 3: study_hours vs. GPA)
feature_x = 'browsing_time'  # Replace (e.g., study_hours, mileage)
feature_y = 'average_order_value'  # Replace (e.g., GPA, fuel_efficiency)
plt.figure(figsize=(8, 6))
for cluster in range(optimal_k):
    plt.scatter(
        df[feature_x][df['Cluster'] == cluster],
        df[feature_y][df['Cluster'] == cluster],
        s=100,
        label=f'Cluster {cluster + 1}'
    )

# If scaling was applied, unscale centroids for visualization; otherwise, use kmeans.cluster_centers_ directly
# If NO scaling was used, uncomment the line below and comment the scaled block
# plt.scatter(
#     kmeans.cluster_centers_[:, features.index(feature_x)],
#     kmeans.cluster_centers_[:, features.index(feature_y)],
#     s=300,
#     c='yellow',
#     marker='*',
#     label='Centroids'
# )

# If scaling was applied (default), keep this block
plt.scatter(
    scaler.inverse_transform(kmeans.cluster_centers_)[:, features.index(feature_x)],
    scaler.inverse_transform(kmeans.cluster_centers_)[:, features.index(feature_y)],
    s=300,
    c='yellow',
    marker='*',
    label='Centroids'
)

plt.title('Clusters Visualization')  # Adjust title as needed (e.g., 'Student Clusters Based on Study Hours and GPA')
plt.xlabel(feature_x)  # Replace with feature name (e.g., 'Study Hours (Weekly)')
plt.ylabel(feature_y)  # Replace with feature name (e.g., 'GPA')
plt.legend()
plt.show()

# Step 8: Deliverables
# Display the final dataset with cluster labels
# If the question asks for specific columns (e.g., Task 3: student_id and Cluster), adjust the columns below
print("Final Dataset with Clusters:\n", df[['shopper_id', 'Cluster']])

# If the question requires saving the dataset (e.g., Task 3), keep this; otherwise, comment it out
df.to_csv('clustered_data.csv', index=False)
print("Clustered data saved to 'clustered_data.csv'")

# If the question requires comparison insights (e.g., Task 1/2: scaling vs. no scaling), uncomment and run the clustering twice
# First run: Comment out scaling (use X), run Steps 5-7, save clusters as 'Cluster (No Scaling)'
# Second run: Uncomment scaling (use X_scaled), run Steps 5-7, save clusters as 'Cluster (Scaled)'
# Then uncomment the block below to print comparison insights
# print("Comparison Insights:")
# print("- Without scaling, features with larger ranges (e.g., mileage) dominate clustering.")
# print("- With scaling, clusters are more balanced across all features.")
# print("- Scaling often leads to tighter, more meaningful clusters, as seen in the scatter plots.")
