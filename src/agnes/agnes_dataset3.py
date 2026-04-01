import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage


# 1. VERIYI YUKLE
df = pd.read_csv("../../data/dataset3.csv")

print("\n" + "=" * 60)
print("DATASET 3 - WHOLESALE CUSTOMERS (AGNES)")
print("=" * 60)

print("\n--- ILK 5 SATIR ---")
print(df.head())

print("\n--- VERI BILGISI ---")
df.info()

print("\n--- ISTATISTIK ---")
print(df.describe())


# 2. GEREKSIZ KOLONLAR
if "Channel" in df.columns:
    df = df.drop(columns=["Channel"])
if "Region" in df.columns:
    df = df.drop(columns=["Region"])

print("\n--- KALAN KOLONLAR ---")
print(df.columns.tolist())


# 3. SCALE
scaler = StandardScaler()
data = scaler.fit_transform(df)

print("\nOlceklenmis veri boyutu:", data.shape)


# 4. DENDROGRAM
linked = linkage(data, method="ward")

plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title("Dendrogram - Dataset 3")
plt.xlabel("Veri Noktalari")
plt.ylabel("Mesafe")
plt.show()


# 5. FARKLI K DEGERLERI
print("\n--- FARKLI K DEGERLERI ---")
for k in range(2, 7):
    agnes = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = agnes.fit_predict(data)

    sil = silhouette_score(data, labels)
    db = davies_bouldin_score(data, labels)

    print(f"K={k} -> Silhouette Score: {sil:.4f} | Davies-Bouldin Index: {db:.4f}")


# 6. FINAL MODEL
final_k = 3

agnes = AgglomerativeClustering(n_clusters=final_k, linkage="ward")
labels = agnes.fit_predict(data)

sil = silhouette_score(data, labels)
db = davies_bouldin_score(data, labels)

print("\n--- FINAL SONUC ---")
print("Secilen K:", final_k)
print("Silhouette Score:", round(sil, 4))
print("Davies-Bouldin Index:", round(db, 4))


# 7. CLUSTER DAGILIMI
df_result = df.copy()
df_result["cluster"] = labels

print("\n--- CLUSTER DAGILIMI ---")
print(df_result["cluster"].value_counts().sort_index())


# 8. CLUSTER ORTALAMALARI
print("\n--- CLUSTER ORTALAMALARI ---")
cluster_means = df_result.groupby("cluster").mean()
print(cluster_means)


# 9. PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

print("\n--- PCA ---")
print("1. bilesen aciklanan varyans:", round(pca.explained_variance_ratio_[0], 4))
print("2. bilesen aciklanan varyans:", round(pca.explained_variance_ratio_[1], 4))
print("Toplam varyans:", round(sum(pca.explained_variance_ratio_), 4))

plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels)
plt.title("PCA - Dataset 3 (AGNES)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()