import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids


# 1. VERIYI YUKLE
df = pd.read_csv("../../data/dataset1.csv", sep=";")

print("\n" + "=" * 60)
print("DATASET 1 - BANK MARKETING (K-MEDOIDS)")
print("=" * 60)

print("\n--- ILK 5 SATIR ---")
print(df.head())

print("\n--- VERI BILGISI ---")
df.info()

print("\n--- ISTATISTIK ---")
print(df.describe())


# 2. SADECE SAYISAL KOLONLARI AL
df_numeric = df.select_dtypes(include=["int64", "float64"]).copy()

print("\n--- SAYISAL KOLONLAR ---")
print(df_numeric.columns.tolist())


# 3. EKSIK VERILERI KONTROL ET
print("\n--- EKSIK DEGER SAYISI ---")
print(df_numeric.isnull().sum())

df_numeric = df_numeric.fillna(df_numeric.median())

print("\nVeri boyutu:", df_numeric.shape)


# 4. OPSIYONEL: DURATION KOLONUNU CIKAR
use_duration = True

if not use_duration and "duration" in df_numeric.columns:
    df_numeric = df_numeric.drop(columns=["duration"])
    print("\nDuration kolonu modelden cikarildi.")
else:
    print("\nDuration kolonu modelde tutuldu.")

print("\n--- MODELE GIRECEK SON KOLONLAR ---")
print(df_numeric.columns.tolist())


# 5. DATASET BUYUK OLDUGU ICIN ORNEKLEME
df_sample = df_numeric.sample(n=2000, random_state=42)

print("\n--- ORNEKLEM SONRASI VERI BOYUTU ---")
print(df_sample.shape)


# 6. STANDARDIZATION
scaler = StandardScaler()
data = scaler.fit_transform(df_sample)

print("\nOlceklenmis veri boyutu:", data.shape)


# 7. FARKLI K DEGERLERI ICIN METRIKLER
print("\n--- FARKLI K DEGERLERI ---")
for k in range(2, 7):
    kmedoids = KMedoids(n_clusters=k, random_state=42, method="pam")
    labels = kmedoids.fit_predict(data)

    sil = silhouette_score(data, labels)
    db = davies_bouldin_score(data, labels)

    print(f"K={k} -> Silhouette Score: {sil:.4f} | Davies-Bouldin Index: {db:.4f}")


# 8. FINAL MODEL
final_k = 3

kmedoids = KMedoids(n_clusters=final_k, random_state=42, method="pam")
labels = kmedoids.fit_predict(data)

sil = silhouette_score(data, labels)
db = davies_bouldin_score(data, labels)

print("\n--- FINAL SONUC ---")
print("Secilen K:", final_k)
print("Silhouette Score:", round(sil, 4))
print("Davies-Bouldin Index:", round(db, 4))


# 9. CLUSTER ETIKETLERI
df_result = df_sample.copy()
df_result["cluster"] = labels

print("\n--- CLUSTER DAGILIMI ---")
print(df_result["cluster"].value_counts().sort_index())


# 10. CLUSTER ORTALAMALARI
print("\n--- CLUSTER BAZLI ORTALAMALAR ---")
cluster_means = df_result.groupby("cluster").mean()
print(cluster_means)


# 11. MEDOID NOKTALARI
print("\n--- MEDOID INDEXLERI ---")
print(kmedoids.medoid_indices_)

print("\n--- MEDOID VERI NOKTALARI ---")
print(df_sample.iloc[kmedoids.medoid_indices_])


# 12. PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

print("\n--- PCA ACIKLAMA ORANI ---")
print("1. bilesen aciklanan varyans:", round(pca.explained_variance_ratio_[0], 4))
print("2. bilesen aciklanan varyans:", round(pca.explained_variance_ratio_[1], 4))
print("Toplam aciklanan varyans:", round(sum(pca.explained_variance_ratio_), 4))

plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels)
plt.title("PCA ile 2 Boyutlu K-Medoids Gorsellestirme")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()