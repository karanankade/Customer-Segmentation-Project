# 🎯 Customer Segmentation Project

An end-to-end Machine Learning project using **K-Means Clustering**, **Principal Component Analysis (PCA)**, and **RFM (Recency, Frequency, Monetary)** behavioral features to group customers into actionable business segments.

---

## 📌 Features

- **Synthetic Data Generation (`generate_data.py`)**: Simulates 1,000 realistic customer profiles containing demographics (Age, Gender, Location) and purchasing metrics (Income, Purchase Frequency, Recency, Total Spend, Website Visits, Campaign Response).
- **Data Preprocessing & Scaling**: Encodes categorical attributes and standardizes numerical features using `StandardScaler`.
- **Optimal Cluster Evaluation**:
  - **Elbow Method**: Evaluates inertia across different values of $K$ (2 to 10) to find the optimal number of clusters.
  - **Silhouette Score**: Quantitative verification of cluster separation.
- **Dimensionality Reduction**: Utilizes PCA to map high-dimensional customer features into 2D space for intuitive visualization.
- **Automated Profiling & Reporting**: Computes mean feature profiles per cluster and exports clean CSV summaries alongside high-resolution plots.

---

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn` (`KMeans`, `StandardScaler`, `PCA`, `silhouette_score`)
- **Data Visualization**: `matplotlib`, `seaborn`

---

## 📁 Directory Structure

```text
Customer Segmentation Project/
├── generate_data.py        # Generates synthetic customers.csv dataset
├── segmentation.py         # Main ML pipeline (Preprocessing, K-Means, PCA, Visualizations)
├── customers.csv           # Generated customer dataset (1,000 rows)
├── requirements.txt        # Python dependency list
├── output/                 # Generated artifacts
│   ├── elbow_curve.png     # K-Means Elbow curve chart
│   ├── cluster_pca.png     # 2D PCA visual scatter plot of clusters
│   ├── cluster_pairplot.png# Feature pairplot colored by cluster ID
│   ├── cluster_profiles.csv# Mean metrics table per segment
│   └── customers_segmented.csv # Full dataset with assigned cluster labels
└── README.md               # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python generate_data.py
```
*Output*: Generates `customers.csv` in the project root.

### 3. Run Segmentation Pipeline
```bash
python segmentation.py
```

*Output*:
- Prints dataset shape, Elbow curve progress, Silhouette score for $K=4$, and cluster mean profiles.
- Populates the `output/` folder with charts and CSV results.

---

## 📊 Output & Results Overview

1. **Elbow Curve (`output/elbow_curve.png`)**: Helps select $K=4$ based on within-cluster sum of squares.
2. **PCA Visualization (`output/cluster_pca.png`)**: Displays distinct customer clusters in 2D space.
3. **Pairplot (`output/cluster_pairplot.png`)**: Explores relationships between `Age`, `Income`, `TotalSpend`, and `Recency` across segments.
4. **Segmented Customer Data (`output/customers_segmented.csv`)**: Full original dataset augmented with a `Cluster` column for downstream targeted marketing.

---
