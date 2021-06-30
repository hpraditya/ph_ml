#import package yang dibutuhkan
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.cluster import KMeans # K-means algorithm


##import csv
raw = pd.read_csv(r'ML_2_Fellow.csv',encoding='windows-1252')
print(raw.head(5))
print(raw.dtypes)


## pre-processing dat

df = raw
#membuat column baru "Nilai_Transaksi" dari "Harga_Satuan" * "Jumlah"
df['Nilai_Transaksi'] = df.apply(lambda row: row['Harga_Satuan'] * row['Jumlah'], axis=1)

#drop row yang column Jumlah atau Harga_Satuannya <= 0
index_drop = df[(df['Jumlah'] <= 0) | (df['Harga_Satuan'] <=0)].index
df.drop(index_drop, inplace=True)

#drop row Kode_Bayar yang bukan tipe data numeric
df['Kode_Bayar'] = pd.to_numeric(df['Kode_Bayar'],errors='coerce')
df.dropna(inplace=True)

print(df.head(5))


## untuk menjawab soal no 1

#mengelompokkan column Barang berdasarkan Jumlah untuk mendapatkan mana Barang yang Jumlah demandnya paling tingi
df.groupby('Barang').agg({'Jumlah':sum}).sort_values(['Jumlah'],ascending=False).head(5)

#mengelompokkan barang yang Nilai_Transaksinya terendah untuk mengetahui mana Barang yang bisa diabaikan
df.groupby('Barang').agg({'Nilai_Transaksi':sum}).sort_values(['Nilai_Transaksi'],ascending=True).head(5)


## untuk menjawab soal no 2

#mengelompokkan negara berdasarkan Nilai_Transaksi untuk mengetahui negara mana yang paling besar transaksinya
df.groupby('Negara').agg({'Nilai_Transaksi':sum}).sort_values(['Nilai_Transaksi'],ascending=False).head(10)

#Mengeliminasi duplikat customer yang transaksi berulang kali
Customer_perCountry = df[['Negara','Kode_Pelanggan']].drop_duplicates()

#Mengelompokkan jumlah customer di tiap  negara
Customer_perCountry.groupby(['Negara']).agg({'Kode_Pelanggan':'count'}).sort_values(['Kode_Pelanggan'],ascending=False)


# untuk menjawab soal no 3

#buat data clustering yang hanya berisi column Harga_Satuan dan Jumlah
cd = df[['Harga_Satuan','Jumlah']]

# transform data dengan cara di normalisasi  dengan StandardScaler()
X = cd.values
X = np.nan_to_num(X)
sc = StandardScaler()

cluster_data = sc.fit_transform(X)

# Modeling data dengan KMeans Clustering

clusters = 3
model = KMeans(init = 'k-means++', 
               n_clusters = clusters, 
               n_init = 10)
model.fit(X)

labels = model.labels_

# Menambahkan label no cluster atau Cluster_Num ke column df
df['Cluster_Num'] = labels
print(df.head())

#melihat jumlah anggota tiap cluster
df.groupby('Cluster_Num').count()

#melihat rata-rata nilai column Jumlah, Harga_Satuan, dan Nilai Transaksi tiap cluster
df.groupby('Cluster_Num').mean()

#plot hasil clustering ke dalam scatterplot
sb.scatterplot(x='Jumlah', y='Harga_Satuan', hue='Cluster_Num', data=df)
plt.title('Distribusi Jumlah dan Harga Satuan berdasarkan Cluster')


## untuk menjawab soal no 4

#mengelompokkan data untuk melihat berapa jumlah customer di tiap negara dan tiap cluster
df.groupby(['Negara','Cluster_Num']).agg({'Kode_Pelanggan':'count'}).sort_values(by='Kode_Pelanggan',ascending=False)

#melihat jumlah nilai transaksi dari cluster 0
df_clus_0 = df[df['Cluster_Num']==0]
df_clus_0.groupby(['Negara']).agg({'Nilai_Transaksi':sum}).sort_values(['Nilai_Transaksi'],ascending=False)

#melihat jumlah nilai transaksi dari cluster 1
df_clus_1 = df[df['Cluster_Num']==1]
df_clus_1.groupby(['Negara']).agg({'Nilai_Transaksi':sum}).sort_values(['Nilai_Transaksi'],ascending=False)

#melihat data transaksi customer di cluster 1
print(df_clus_1)

#melihat jumlah nilai transaksi dari cluster 2
df_clus_2 = df[df['Cluster_Num']==2]
df_clus_2.groupby(['Negara']).agg({'Nilai_Transaksi':sum}).sort_values(['Nilai_Transaksi'],ascending=False)