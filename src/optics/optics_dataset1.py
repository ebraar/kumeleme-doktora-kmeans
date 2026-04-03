import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA


# 1. VERIYI YUKLE
df = pd.read_csv("../../data/dataset1.csv", sep=";")

print("\n" + "=" * 60)
print("DATASET 1 - BANK MARKETING (OPTICS)")
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


# 5. ORNEKLEME (DATASET BUYUK OLDUGU ICIN)
df_sample = df_numeric.sample(n=2000, random_state=42)

print("\n--- ORNEKLEM SONRASI VERI BOYUTU ---")
print(df_sample.shape)


# 6. STANDARDIZATION
scaler = StandardScaler()
data = scaler.fit_transform(df_sample)

print("\nOlceklenmis veri boyutu:", data.shape)


# 7. OPTICS MODEL
optics = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05)
labels = optics.fit_predict(data)

print("\n--- OPTICS SONUCLARI ---")
unique_labels = np.unique(labels)
print("Bulunan cluster etiketleri:", unique_labels)
print("Cluster sayisi (noise haric):", len(set(labels)) - (1 if -1 in labels else 0))
print("Noise sayisi:", np.sum(labels == -1))


# 8. METRIKLER
# Noise hariç hesaplamak daha doğru
mask = labels != -1

if len(set(labels[mask])) > 1:
    sil = silhouette_score(data[mask], labels[mask])
    db = davies_bouldin_score(data[mask], labels[mask])

    print("\n--- DEGERLENDIRME METRIKLERI ---")
    print("Silhouette Score:", round(sil, 4))
    print("Davies-Bouldin Index:", round(db, 4))
else:
    print("\n--- DEGERLENDIRME METRIKLERI ---")
    print("Yeterli sayida cluster bulunamadigi icin metrik hesaplanamadi.")


# 9. CLUSTER DAGILIMI
df_result = df_sample.copy()
df_result["cluster"] = labels

print("\n--- CLUSTER DAGILIMI ---")
print(df_result["cluster"].value_counts().sort_index())


# 10. CLUSTER ORTALAMALARI
print("\n--- CLUSTER BAZLI ORTALAMALAR ---")
print(df_result[df_result["cluster"] != -1].groupby("cluster").mean())


# 11. REACHABILITY PLOT
plt.figure(figsize=(10, 5))
space = np.arange(len(data))
reachability = optics.reachability_[optics.ordering_]
labels_ordered = labels[optics.ordering_]

plt.plot(space, reachability, marker=".", linestyle="none", alpha=0.7)
plt.title("Reachability Plot - Dataset 1 (OPTICS)")
plt.xlabel("Veri Noktalari")
plt.ylabel("Reachability Distance")
plt.grid(True)
plt.show()


# 12. PCA
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