# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 21:42:12 2024

@author: mcall
"""
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

                     # Load dataset
data = pd.read_csv('Mall_Customers.csv')

                      # Select relevant features for clustering
features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

                          # Scale features to standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

                               #Code for EDA
            # Visualizing the distribution of annual income and spending score
#visualizations: Use histograms, pair plots, and scatter plots to visualize relationships between features (e.g., Annual Income vs. Spending Score).
#Statistics: Calculate basic statistics to identify trends (e.g., mean, median, range) in customer behavior.

sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=data)
plt.title('Annual Income vs. Spending Score')
plt.show()

# Pairplot to visualize relationships
sns.pairplot(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
plt.show()


               #Code for Model Building


             # Determine the optimal number of clusters using the Elbow Method
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

            # Apply K-Means clustering with the selected number of clusters
optimal_k = 5  # From the elbow graph, you might select 5 clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

                             #Visualizing Clusters
   #Code for Cluster Visualization
   # Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=data, palette='viridis', s=100)
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
