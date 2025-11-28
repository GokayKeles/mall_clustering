import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Set style for better visualizations
plt.style.use('ggplot')  # Using a valid matplotlib style
sns.set_theme(style="whitegrid")  # Setting seaborn theme

# Read the dataset
print("Loading dataset...")
df = pd.read_csv('Mall_Customers.csv')

# Data Analysis
print("\nDataset Information:")
print(df.info())
print("\nDataset Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Convert categorical data to numbers
le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])

# Fill missing values if any
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Data Visualization
# 1. Distribution of Age
plt.figure(figsize=(12, 8))
sns.histplot(data=df, x='Age', bins=30, kde=True)
plt.title('Distribution of Customer Ages', pad=20, fontsize=14)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.show()

# 2. Distribution of Annual Income
plt.figure(figsize=(12, 8))
sns.histplot(data=df, x='Annual Income (k$)', bins=30, kde=True)
plt.title('Distribution of Annual Income', pad=20, fontsize=14)
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.show()

# Plot 3: Spending Score by Genre
# This boxplot shows how spending scores differ between male and female customers.
plt.figure(figsize=(6,4))
sns.boxplot(x='Genre', y='Spending Score (1-100)', data=df)
plt.title('Spending Score by Genre')
plt.show()

# Correlation Analysis
# Calculate correlation matrix
correlation_matrix = df.corr()

# Plot correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Matrix of Customer Data', pad=20, fontsize=14)
plt.tight_layout()
plt.show()

# Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method for finding optimal K
print("\nCalculating optimal number of clusters...")
inertias = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(12, 8))
plt.plot(K, inertias, 'bx-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.title('Elbow Method For Optimal k', pad=20, fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Perform K-means clustering with optimal k=5
print("\nPerforming K-means clustering...")
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Create scatter plot of clusters
plt.figure(figsize=(14, 10))
scatter = plt.scatter(X['Annual Income (k$)'], 
                     X['Spending Score (1-100)'], 
                     c=df['Cluster'], 
                     cmap='viridis',
                     s=100,
                     alpha=0.6)
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.title('Customer Segments', pad=20, fontsize=14)
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate cluster statistics
print("\nCalculating cluster statistics...")
cluster_stats = df.groupby('Cluster').agg({
    'Annual Income (k$)': ['mean', 'std', 'min', 'max'],
    'Spending Score (1-100)': ['mean', 'std', 'min', 'max'],
    'Age': ['mean', 'std', 'min', 'max']
}).round(2)

# Save cluster statistics to CSV
cluster_stats.to_csv('cluster_statistics.csv')

# Print cluster centers
print("\nCluster Centers:")
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=['Annual Income (k$)', 'Spending Score (1-100)']
)
print(cluster_centers)

# Print cluster sizes
print("\nCluster Sizes:")
cluster_sizes = df['Cluster'].value_counts().sort_index()
print(cluster_sizes)

# Interpret the clusters
print("\nCluster Interpretation:")
for i in range(5):
    income_mean = cluster_stats.loc[i, ('Annual Income (k$)', 'mean')]
    spending_mean = cluster_stats.loc[i, ('Spending Score (1-100)', 'mean')]
    size = cluster_sizes[i]
    
    print(f"\nCluster {i}:")
    print(f"Size: {size} customers")
    print(f"Average Annual Income: ${income_mean:.2f}k")
    print(f"Average Spending Score: {spending_mean:.2f}")
    
    if income_mean > 70 and spending_mean > 50:
        print("Segment: High Income, High Spending")
    elif income_mean > 70 and spending_mean <= 50:
        print("Segment: High Income, Low Spending")
    elif income_mean <= 70 and spending_mean > 50:
        print("Segment: Low Income, High Spending")
    else:
        print("Segment: Low Income, Low Spending")
