import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA


# 1. VERIYI YUKLE
df = pd.read_csv("../../data/dataset3.csv")

print("\n" + "=" * 60)
print("DATASET 3 - WHOLESALE CUSTOMERS")
print("=" * 60)

print("\n--- ILK 5 SATIR ---")
print(df.head())

print("\n--- VERI BILGISI ---")
df.info()

print("\n--- ISTATISTIK ---")
print(df.describe())


# 2. GEREKSIZ KOLONLAR
# Channel ve Region categorical gibi → çıkarıyoruz
df = df.drop(columns=["Channel", "Region"])

print("\n--- KALAN KOLONLAR ---")
print(df.columns.tolist())


# 3. SCALE
scaler = StandardScaler()
data = scaler.fit_transform(df)

print("\nOlceklenmis veri boyutu:", data.shape)


# 4. ELBOW
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(data)
    inertia.append(km.inertia_)

plt.plot(range(1, 11), inertia, marker="o")
plt.title("Elbow Method - Dataset 3")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()


# 5. METRIKLER
print("\n--- FARKLI K DEGERLERI ---")
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(data)

    sil = silhouette_score(data, labels)
    db = davies_bouldin_score(data, labels)

    print(f"K={k} -> Silhouette Score: {sil:.4f} | Davies-Bouldin Index: {db:.4f}")


# 6. FINAL MODEL (sonra belirleyeceğiz)
final_k = 3

kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(data)

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
print(df_result["cluster"].value_counts())


print("\n--- CLUSTER ORTALAMALARI ---")
print(df_result.groupby("cluster").mean())


# 8. PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

print("\n--- PCA ---")
print("Toplam varyans:", round(sum(pca.explained_variance_ratio_), 4))

plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels)
plt.title("PCA - Dataset 3")
plt.grid(True)
plt.show()