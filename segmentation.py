import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os

def main():
    # 1. Load Data
    try:
        df = pd.read_csv('customers.csv')
        print("Data loaded successfully. Shape:", df.shape)
    except FileNotFoundError:
        print("Error: customers.csv not found. Run generate_data.py first.")
        return

    # Create output directory
    os.makedirs('output', exist_ok=True)

    # 2. Preprocessing & Feature Engineering
    # We will use numerical behavioral and demographic features for clustering
    # Encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['Gender', 'Location'], drop_first=True)
    
    # Select features for clustering
    # We focus on RFM and demographics
    features = ['Age', 'Income', 'PurchaseFrequency', 'Recency', 'TotalSpend', 'WebsiteVisits']
    X = df_encoded[features]

    # Handle missing values if any (though synthetic data shouldn't have any)
    X = X.fillna(X.median())

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Model Evaluation - Elbow Method
    print("Running Elbow Method...")
    inertia = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Within-cluster Sum of Squares)')
    plt.grid(True)
    plt.savefig('output/elbow_curve.png')
    plt.close()
    print("Saved Elbow Curve to output/elbow_curve.png")

    # 4. Clustering - Choosing k=4 (common for customer segmentation)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # 5. Evaluate with Silhouette Score
    sil_score = silhouette_score(X_scaled, df['Cluster'])
    print(f"Silhouette Score for K={optimal_k}: {sil_score:.4f}")

    # 6. Visualization & Insights
    
    # PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='viridis', data=df, alpha=0.7)
    plt.title('Customer Segments (PCA Reduced)')
    plt.savefig('output/cluster_pca.png')
    plt.close()
    print("Saved PCA Scatter Plot to output/cluster_pca.png")

    # Pairplot for key features to see differences
    sample_features = ['Age', 'Income', 'TotalSpend', 'Recency', 'Cluster']
    sns.pairplot(df[sample_features], hue='Cluster', palette='viridis', corner=True)
    plt.savefig('output/cluster_pairplot.png')
    plt.close()
    print("Saved Pairplot to output/cluster_pairplot.png")

    # Calculate cluster profiles (means for each feature)
    cluster_profile = df.groupby('Cluster')[features].mean().round(2)
    print("\n--- Cluster Profiles ---")
    print(cluster_profile)
    
    # Save the profiled data summary
    cluster_profile.to_csv('output/cluster_profiles.csv')
    print("Saved Cluster Profiles to output/cluster_profiles.csv")
    
    # Save the original dataframe with cluster labels
    df.drop(['PCA1', 'PCA2'], axis=1).to_csv('output/customers_segmented.csv', index=False)
    print("Saved Full Segmented Data to output/customers_segmented.csv")

if __name__ == "__main__":
    main()
