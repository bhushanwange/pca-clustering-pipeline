import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Step 1: Load a high-dimensional dataset (simulated high-dim feature space, e.g., gene expression style)
print("Loading high-dimensional dataset...")
X_high_dim, _ = make_blobs(n_samples=400, n_features=50, centers=4, random_state=42)

# Step 2: Scale data and fit PCA retaining 95% variance
print("Scaling data and fitting PCA (retaining 95% variance)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_high_dim)

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"Original features: {X_scaled.shape[1]}")
print(f"Reduced features after PCA (95% variance): {X_pca.shape[1]}")

# Step 3: Transform data to principal components (already done by pca.fit_transform)
# Step 4: Run K-Means on the reduced data
print("Running K-Means clustering on reduced data...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca)

# Step 5: Plot clusters on PC1 vs PC2 with matplotlib
print("Generating PC1 vs PC2 cluster plot...")
plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    cmap="viridis",
    edgecolors="k",
    alpha=0.8,
)
plt.title("K-Means Clusters on PCA Reduced Data (PC1 vs PC2)")
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.grid(True)
plt.show()