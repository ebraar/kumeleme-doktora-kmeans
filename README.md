# 📊 Customer Segmentation using Clustering Algorithms

## 📌 Project Overview

This project focuses on **customer segmentation** using three different clustering algorithms:

- **K-Means**
- **AGNES (Hierarchical Clustering)**
- **K-Medoids**

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

### 🔹 K-Medoids
- Similar to K-Means but uses real data points as centers  
- More robust to outliers  
- Provides interpretable cluster representatives (medoids)  

---

## 📊 Evaluation Metrics

To evaluate clustering performance, the following metrics were used:

- **Silhouette Score** → Measures cluster separation (higher is better)  
- **Davies-Bouldin Index** → Measures cluster compactness (lower is better)  

---

## 🔍 Key Insights

- Different algorithms can produce different cluster structures on the same dataset  
- K-Means works well for balanced datasets but is sensitive to outliers  
- AGNES reveals hierarchical relationships and can detect natural cluster counts  
- K-Medoids performs better on datasets with **outliers** and provides more stable clusters  

### 📌 Example Insight (Dataset 3):
- Customers were grouped into:
  - Fresh-product focused buyers  
  - Grocery & detergent-heavy buyers  
  - Low-volume customers  

---

## 📈 Visualization

- PCA (Principal Component Analysis) was used to reduce dimensionality  
- Clusters were visualized in 2D space for better interpretation  

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
├── src/
│   ├── kmeans/
│   │   ├── kmeans_dataset1.py
│   │   ├── kmeans_dataset2.py
│   │   └── kmeans_dataset3.py
│   │
│   ├── agnes/
│   │   ├── agnes_dataset1.py
│   │   ├── agnes_dataset2.py
│   │   └── agnes_dataset3.py
│   │
│   └── kmedoids/
│       ├── kmedoids_dataset1.py
│       ├── kmedoids_dataset2.py
│       └── kmedoids_dataset3.py
│
└── README.md
```

---

## 🚀 Installation

```bash
pip install pandas numpy matplotlib scikit-learn scikit-learn-extra

cd src 
cd kmeans/agnes/kmedoids
python kmeans_dataset1.py/agnes_dataset1.py/kmedoids_dataset1.py