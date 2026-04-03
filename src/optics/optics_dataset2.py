import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA


# 1. VERIYI YUKLE
df = pd.read_csv("../../data/dataset2.csv")

print("\n" + "=" * 60)
print("DATASET 2 - MALL CUSTOMERS (OPTICS)")
print("=" * 60)

print("\n--- ILK 5 SATIR ---")
print(df.head())

print("\n--- VERI BILGISI ---")
df.info()

print("\n--- ISTATISTIK ---")
print(df.describe())


# 2. GEREKSIZ KOLONLARI CIKAR
if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])

if "Gender" in df.columns:
    df = df.drop(columns=["Gender"])

print("\n--- KALAN KOLONLAR ---")
print(df.columns.tolist())


# 3. STANDARDIZATION
scaler = StandardScaler()
data = scaler.fit_transform(df)

print("\nOlceklenmis veri boyutu:", data.shape)


# 4. OPTICS MODEL
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)
labels = optics.fit_predict(data)

print("\n--- OPTICS SONUCLARI ---")
unique_labels = np.unique(labels)
print("Bulunan cluster etiketleri:", unique_labels)
print("Cluster sayisi (noise haric):", len(set(labels)) - (1 if -1 in labels else 0))
print("Noise sayisi:", np.sum(labels == -1))


# 5. METRIKLER
mask = labels != -1

print("\n--- DEGERLENDIRME METRIKLERI ---")
if len(set(labels[mask])) > 1:
    sil = silhouette_score(data[mask], labels[mask])
    db = davies_bouldin_score(data[mask], labels[mask])

    print("Silhouette Score:", round(sil, 4))
    print("Davies-Bouldin Index:", round(db, 4))
else:
    print("Yeterli sayida cluster bulunamadigi icin metrik hesaplanamadi.")


# 6. CLUSTER DAGILIMI
df_result = df.copy()
df_result["cluster"] = labels

print("\n--- CLUSTER DAGILIMI ---")
print(df_result["cluster"].value_counts().sort_index())


# 7. CLUSTER ORTALAMALARI
print("\n--- CLUSTER BAZLI ORTALAMALAR ---")
if len(df_result[df_result["cluster"] != -1]) > 0:
    print(df_result[df_result["cluster"] != -1].groupby("cluster").mean())
else:
    print("Cluster bulunamadi.")


# 8. REACHABILITY PLOT
plt.figure(figsize=(10, 5))
space = np.arange(len(data))
reachability = optics.reachability_[optics.ordering_]

plt.plot(space, reachability, marker=".", linestyle="none", alpha=0.7)
plt.title("Reachability Plot - Dataset 2 (OPTICS)")
plt.xlabel("Veri Noktalari")
plt.ylabel("Reachability Distance")
plt.grid(True)
plt.show()


# 9. PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

print("\n--- PCA ACIKLAMA ORANI ---")
print("1. bilesen aciklanan varyans:", round(pca.explained_variance_ratio_[0], 4))
print("2. bilesen aciklanan varyans:", round(pca.explained_variance_ratio_[1], 4))
print("Toplam aciklanan varyans:", round(sum(pca.explained_variance_ratio_), 4))

plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels)
plt.title("PCA ile 2 Boyutlu OPTICS Gorsellestirme")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()