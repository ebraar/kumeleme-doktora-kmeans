import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA


# 1. VERIYI YUKLE
df = pd.read_csv("../../data/dataset2.csv")

print("\n" + "=" * 60)
print("DATASET 2 - MALL CUSTOMERS")
print("=" * 60)

print("\n--- ILK 5 SATIR ---")
print(df.head())

print("\n--- VERI BILGISI ---")
df.info()

print("\n--- ISTATISTIK ---")
print(df.describe())


# 2. GEREKSIZ KOLONLARI SIL
if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])

# Gender sayisal degil → drop (istersen encode da edebilirdik)
if "Gender" in df.columns:
    df = df.drop(columns=["Gender"])

print("\n--- KALAN KOLONLAR ---")
print(df.columns.tolist())


# 3. SAYISAL VERI
df_numeric = df.copy()

# 4. SCALE
scaler = StandardScaler()
data = scaler.fit_transform(df_numeric)

print("\nOlceklenmis veri boyutu:", data.shape)


# 5. ELBOW METHOD
inertia = []

for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(data)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker="o")
plt.title("Elbow Method - Dataset 2")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()


# 6. METRIKLER
print("\n--- FARKLI K DEGERLERI ---")
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(data)

    sil = silhouette_score(data, labels)
    db = davies_bouldin_score(data, labels)

    print(f"K={k} -> Silhouette Score: {sil:.4f} | Davies-Bouldin Index: {db:.4f}")

# Not:
# K=6 metrik olarak daha iyi olsa da,
# yorumlanabilirlik açısından K=5 seçilmiştir.

# 7. FINAL MODEL (sonra güncelleyebilirsin)
final_k = 5

kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(data)

sil = silhouette_score(data, labels)
db = davies_bouldin_score(data, labels)

print("\n--- FINAL SONUC ---")
print("Secilen K:", final_k)
print("Silhouette Score:", round(sil, 4))
print("Davies-Bouldin Index:", round(db, 4))


# 8. CLUSTER EKLE
df_result = df_numeric.copy()
df_result["cluster"] = labels

print("\n--- CLUSTER DAGILIMI ---")
print(df_result["cluster"].value_counts().sort_index())


# 9. ORTALAMALAR
print("\n--- CLUSTER BAZLI ORTALAMALAR ---")
cluster_means = df_result.groupby("cluster").mean()
print(cluster_means)


# 10. PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

print("\n--- PCA ACIKLAMA ORANI ---")
print("Toplam:", round(sum(pca.explained_variance_ratio_), 4))


plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels)
plt.title("PCA - Dataset 2")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()