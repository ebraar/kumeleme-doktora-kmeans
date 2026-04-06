import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage


# 1. VERIYI YUKLE
df = pd.read_csv("../../data/dataset2.csv")

print("\n" + "=" * 60)
print("DATASET 2 - MALL CUSTOMERS (AGNES)")
print("=" * 60)

print("\n--- ILK 5 SATIR ---")
print(df.head())

print("\n--- VERI BILGISI ---")
df.info()

print("\n--- ISTATISTIK ---")
print(df.describe())


# 2. GEREKSIZ KOLONLAR
if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])

# Gender → kategorik → çıkarıyoruz
if "Gender" in df.columns:
    df = df.drop(columns=["Gender"])

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
plt.title("Dendrogram - Dataset 2")
plt.xlabel("Veri Noktalari")
plt.ylabel("Mesafe")
plt.show()


# 5. METRIKLER
print("\n--- FARKLI K DEGERLERI ---")
for k in range(2, 7):
    agnes = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = agnes.fit_predict(data)

    sil = silhouette_score(data, labels)
    db = davies_bouldin_score(data, labels)

    print(f"K={k} -> Silhouette Score: {sil:.4f} | Davies-Bouldin Index: {db:.4f}")


# 6. FINAL MODEL (sonra karar veririz)
final_k = 6

agnes = AgglomerativeClustering(n_clusters=final_k, linkage="ward")
labels = agnes.fit_predict(data)

sil = silhouette_score(data, labels)
db = davies_bouldin_score(data, labels)

print("\n--- FINAL SONUC ---")
print("Secilen K:", final_k)
print("Silhouette Score:", round(sil, 4))
print("Davies-Bouldin Index:", round(db, 4))


# 7. CLUSTER
df_result = df.copy()
df_result["cluster"] = labels

print("\n--- CLUSTER DAGILIMI ---")
print(df_result["cluster"].value_counts().sort_index())


# 8. ORTALAMALAR
print("\n--- CLUSTER BAZLI ORTALAMALAR ---")
print(df_result.groupby("cluster").mean())


# 9. PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

print("\n--- PCA ---")
print("Toplam varyans:", round(sum(pca.explained_variance_ratio_), 4))

plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels)
plt.title("PCA - Dataset 2 (AGNES)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

output_dir = "../../results/agnes/dataset2"
os.makedirs(output_dir, exist_ok=True)

summary_path = os.path.join(output_dir, "summary-agnes-dataset2.txt")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("DATASET 2 - MALL CUSTOMERS (AGNES)\n")
    f.write("=" * 60 + "\n\n")

    f.write("KULLANILAN PARAMETRELER\n")
    f.write("linkage: ward\n")
    f.write(f"final_k: {final_k}\n\n")

    f.write("FINAL SONUCLAR\n")
    f.write(f"Silhouette Score: {round(sil, 4)}\n")
    f.write(f"Davies-Bouldin Index: {round(db, 4)}\n\n")

    f.write("CLUSTER DAGILIMI\n")
    f.write(df_result["cluster"].value_counts().sort_index().to_string())
    f.write("\n\n")

    f.write("CLUSTER BAZLI ORTALAMALAR\n")
    f.write(df_result.groupby("cluster").mean().to_string())
    f.write("\n")

print(f"\nOzet dosyasi kaydedildi: {summary_path}")