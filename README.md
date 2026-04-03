# 📊 Customer Segmentation using Clustering Algorithms

## 📌 Project Overview

This project focuses on **customer segmentation** using three different clustering algorithms:

- **K-Means**
- **AGNES (Hierarchical Clustering)**
- **OPTICS (Density-Based Clustering)**

The goal is to analyze different datasets and identify meaningful customer groups based on behavioral and numerical features.

---

## 🎯 Objectives

- Apply multiple clustering algorithms on different datasets  
- Compare clustering performance using evaluation metrics  
- Analyze cluster structures and interpret customer segments  
- Understand how different algorithms behave on various data types  

---

## 📁 Datasets

The project includes three different datasets:

### 1️⃣ Dataset 1 – Bank Marketing
- Customer demographic and campaign data  
- Includes economic indicators (euribor, employment rate, etc.)

### 2️⃣ Dataset 2 – Mall Customers
- Age, Annual Income, Spending Score  
- Used for customer behavior segmentation  

### 3️⃣ Dataset 3 – Wholesale Customers
- Annual spending on different product categories:
  - Fresh
  - Milk
  - Grocery
  - Frozen
  - Detergents
  - Delicassen  

---

## ⚙️ Methods Used

### 🔹 K-Means
- Centroid-based clustering  
- Fast and efficient  
- Sensitive to outliers  

### 🔹 AGNES (Hierarchical Clustering)
- Bottom-up clustering approach  
- Uses dendrogram for cluster visualization  
- Reveals hierarchical structure of data  

### 🔹 OPTICS (Density-Based Clustering)
- Density-based clustering algorithm  
- Does not require a predefined K value 
- Can automatically detect: clusters and noise(outliers)  
- Works well for datasets with irregular shapes and density variations.

---

## 📊 Evaluation Metrics

To evaluate clustering performance, the following metrics were used:

- **Silhouette Score** → Measures cluster separation (higher is better)  
- **Davies-Bouldin Index** → Measures cluster compactness (lower is better)  

---

## 📌 Algorithm Comparison

### 🔹 K-Means
- Performs well on balanced datasets  
- Struggles with outliers  

### 🔹 AGNES (Hierarchical Clustering)
- Reveals hierarchical relationships  
- Helps understand natural cluster formation  

### 🔹 OPTICS (Density-Based Clustering)
- Automatically detects cluster structures  
- Identifies noise (outliers)  
- May fail when data lacks clear density differences   

---

## 📈 Visualization

- PCA (Principal Component Analysis) was used to reduce dimensionality  
- Clusters were visualized in 2D space for better interpretation  
- Reachability plots were used for OPTICS analysis

---

## 🧪 Project Structure

```bash
kumeleme-doktora-kmeans/
│
├── data/
│   ├── dataset1.csv
│   ├── dataset2.csv
│   └── dataset3.csv
│
├── results/
│   ├── agnes/
│   │   ├── dataset1/
│   │   │   ├── dendrogram-agnes-dataset1.png
│   │   │   └── pca-agnes-dataset1.png
│   │   │
│   │   ├── dataset2/
│   │   │   ├── dendrogram-agnes-dataset2.png
│   │   │   └── pca-agnes-dataset2.png
│   │   │
│   │   └── dataset3/
│   │       ├── dendrogram-agnes-dataset3.png
│   │       └── pca-agnes-dataset3.png
│   │
│   ├── kmeans/
│   │   ├── dataset1/
│   │   │   ├── elbow-kmeans-dataset1.png
│   │   │   └── pca-kmeans-dataset1.png
│   │   │
│   │   ├── dataset2/
│   │   │   ├── elbow-kmeans-dataset2.png
│   │   │   └── pca-kmeans-dataset2.png
│   │   │
│   │   └── dataset3/
│   │       ├── elbow-kmeans-dataset3.png
│   │       └── pca-kmeans-dataset3.png
│   │
│   └── optics/
│       ├── dataset1/
│       │   ├── pca-optics-dataset1.png
│       │   └── reachability-optics-dataset1.png
│       │
│       ├── dataset2/
│       │   ├── pca-optics-dataset2.png
│       │   └── reachability-optics-dataset2.png
│       │
│       └── dataset3/
│           ├── pca-optics-dataset3.png
│           └── reachability-optics-dataset3.png
│
├── src/
│   ├── agnes/
│   │   ├── agnes_dataset1.py
│   │   ├── agnes_dataset2.py
│   │   └── agnes_dataset3.py
│   │
│   ├── kmeans/
│   │   ├── kmeans_dataset1.py
│   │   ├── kmeans_dataset2.py
│   │   └── kmeans_dataset3.py
│   │
│   └── optics/
│       ├── optics_dataset1.py
│       ├── optics_dataset2.py
│       └── optics_dataset3.py
│
└── README.md
```

## 📂 Results

All generated visual outputs are stored in the `results/` directory, organized by algorithm and dataset.

- **K-Means**
  - Elbow plots
  - PCA visualizations

- **AGNES**
  - Dendrograms
  - PCA visualizations

- **OPTICS**
  - Reachability plots
  - PCA visualizations

---

## 🚀 Installation

```bash
pip install pandas numpy matplotlib scikit-learn scikit-learn-extra

cd src 
cd kmeans/agnes/optics
python kmeans_dataset1.py/agnes_dataset1.py/optics_dataset1.py