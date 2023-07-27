# Laporan Proyek Machine Learning - Joseph Setiawan Hardadi

## Domain Proyek

Pada tanggal 10 Oktober, diperingati hari kesehatan mental di seluruh dunia. Tentunya di hari tersebut, orang-orang ingin meningkatkan kesadaran akan kesehatan mental. Kesehatan mental dapat juga dipengaruhi pekerjaannya. Tenaga kerja dapat mengalami burnout, suatu keadaan mental yang ditandai dengan kelelahan, perasaan sinis, rasa terpisah dengan pekerjaan, dan rasa tidak efektif dan tidak adanya pencapaian. Proyek ini berisi prediksi tingkat burnout berdasarkan faktor-faktor tertentu seperti lama kerja, gender, fasilitas kerja dari rumah, alokasi sumber daya, pangkat, tipe perusahaan, dan tingkat kelelahan mental mereka. [Are Your Employees Burning Out?](https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out).[Understanding the burnout experience: recent research and its implications for psychiatry](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4911781/)

## Business Understanding

Proyek ini berisi perumusan masalah dan penyelesaian tingkat burnout dari para pegawai.

### Problem Statements

Masalah yang akan coba diselesaikan:
- Apa faktor-faktor penyebab burnout yang sangat berpengaruh?
- Bagaimana cara mendeteksi burnout?

### Goals

Tujuan proyek ini adalah:
- Mencari faktor-faktor burnout dari yang paling berpengaruh sampai yang tidak begitu berpengaruh
- Mendeteksi burnout dengan algoritma machine learning dengan pendekatan terhadap masalah regresi

## Data Understanding
Data yang digunakan diambil dari Kaggle. Data tersebut diambil dari suatu kompetisi dengan judul "HackerEarth Machine Learning Challenge: Are your employees burning out?". Data ini berformat csv, sehingga dapat ditangani oleh library Pandas dengan mudah. Data dalam kaggle tersebut memiliki 9 kolom fitur dan 22,750 baris data, meski tidak dijamin semua datanya lengkap. Data yang akan dipakai adalah file untuk training, karena data test tidak memiliki hasil burn rate yang dapat digunakan untuk mengecek hasil akurasi. [Are Your Employees Burning Out?](https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out).

### Variabel-variabel pada dataset adalah sebagai berikut:
- Employee ID : ID setiap pegawai dalam data ini
- Date of Joining : Tanggal pegawai direkrut (Berbentuk tanggal berformat YYYY-MM-DD seperti 2008-12-30)
- Gender : Jenis kelamin pegawai (Pria atau Wanita)
- Company Type : Tipe perusahaan tempat pegawai bekerja. Jawabannya hanya 2, yaitu Service atau Product
- WFH Setup Available : Apakah untuk pegawai ini, ada fasilitas Kerja Dari Rumah (WFH). Jawabannya Yes or No
- Designation : Tingkat Jabatan pegawai. Isi data berupa angka dari 1.0 sampai 5.0. Semakin besar angkanya, semakin tinggi jabatannya
- Resource Allocation : Fasilitas yang dimiliki pegawai, seperti jam kerja. Isi data berupa nilai dari 1.0 sampai 10.0. Semakin besar nilainya, semakin banyak fasilitasnya/sumber dayanya.
- Mental Fatigue Score : tingkat kelelahan mental pegawai. Isi data berupa nilai dari 0.0 sampai 10.0. Semakin besar nilainya, semakin lelah pegawainya.
- Burn Rate : Nilai target yang menggambarkan tingkat burnout karyawan. Isi data berupa nilai dari 0.0 sampai 1.0. Semakin besar nilainya, semakin beresiko seorang pegawai mengalami burnout.


**Missing Value**:
Berikut jumlah data yang memiliki missing value :

![Missing_Stat](https://i.postimg.cc/9XZq0t7g/missing-number.png)

Berikut gambaran data yang hilang :
[![missing-val.png](https://i.postimg.cc/R0QBkytz/missing-val.png)](https://postimg.cc/hJj6xCMC)

**Explorative Data Analysis**:
- (Date Of Joining diubah menjadi Join Days, dengan isi selisih tanggal tertinggi di data dengan tanggal pegawai. Hal ini memudahkan visualisasi data untuk pegawai mana yang paling lama ada di perusahaan)
- Join Days cukup terbagi rata. Pegawai-pegawai memiliki tanggal join perusahaan yang beragam.<br>
![Join Days](https://i.postimg.cc/MKW2QWck/join-days.png)

- Lebih dari 6000 pegawai berada di Designation 2.0 <br>
![designation.png](https://i.postimg.cc/522W0CgQ/designation.png)

- Gender yang dominan dalam data ini adalah Female.<br>
![gender](https://i.postimg.cc/kX9d0Ztc/gender.png)

- Dalam data ini, rerata Burn Rate gender Male lebih tinggi dari rerata Burn Rate Female.<br>
![gender-burn-rate.png](https://i.postimg.cc/zXS4L1xb/gender-burn-rate.png)

- Rerata Burn Rate dalam pegawai Company Type Service dan Product sama<br>
![type-burn-rate.png](https://i.postimg.cc/t4Bf5NQ9/type-burn-rate.png)

- Rerata Burn Rate Pegawai tanpa WFH Setup Available lebih tinggi daripada Rerata Burn Rate pegawai dengan WFH Setup Available <br>
![WFH-burn-rate.png](https://i.postimg.cc/htD6T3YC/WFH-burn-rate.png)

- Rata-Rata burn rate data adalah sekitar 0.452444. Banyak pegawai yang memiliki nilai burnout 0.4 - 0.5<br>
![Describe](https://i.postimg.cc/SsCFWR74/describe.png)
![Burn Rate](https://i.postimg.cc/4dhLcc54/burn-rate.png)

- Outlier dideteksi dengan menggunakan fungsi quantile panda untuk melihat quantile 25% dan 75%. Quantile ini disebut Q1 dan Q3. Sejauh ini tidak ditemukan data outlier. Berikut kode yang dipakai 
```
Q1, Q3 = (employees_train_nonNull.quantile(0.25),employees_train_nonNull.quantile(0.75))
IQR= Q3-Q1
employees_train_nonNull[((employees_train_nonNull>(Q3+1.5*IQR))).all(axis=1)]
```
![Outlier](https://i.postimg.cc/C5pQvFhg/ml-outlier.png)

- Dalam matriks korelasi ini, data numerik selain join days cukup berpengaruh kepada burn rate. Mental Fatigue Score memiliki korelasi yang dekat dengan Burn Rate dengan skor 0.96, diikuti dengan Resources Allocation engan nilai 0.86 dan Designation dengan nilai 0.74. Join Days tidak memiliki pengaruh tinggi terhada tingkat burnout.
![correlation.png](https://i.postimg.cc/L8s47XqR/correlation.png)


## Data Preparation
- Pertama, kita hapus dulu data dengan missing value, yaitu data yang memiliki nilai NaN. Untuk fitur numerikal seperti Designation, Mental Fatigue Score dan Burn Rate, nilai 0.0 memang dipakai sehingga bisa dibiarkan. Data dalam fitur Resource Allocation tidak memiliki nilai 0, karena nilai terkecilnya adalah 1. Akan tetapi, data-data tersebut masih mungkin memiliki missing value, atau nilai yang dibiarkan kosong begitu saja.
- Data yang bersih dari NaN berjumlah 18,590 data. Jumlah ini masih cukup banyak sehingga kita bisa memasuki tahap selanjutnya.
- Setelah menghapus missing value, kita dapat mengganti beberapa fitur menjadi angka agar datanya seragam, yaitu data angka.
- Sebelumnya telah dilakukan perubahan data dari Date Of Joining menjadi Join Days.
- Setelah itu data Gender, Company Type dan WFH Setup Available diubah menjadi data binary (0 dan 1) dengan nama IsMale, IsServiceCompany, dan WFH Setup Availlable. 0 untuk Female di Gender, Company Product, dan Tidak adanya WFH Setup. Teknik ini biasa disebut encoding dengan fungsi pandas get_dummies.
```
def feature_encode(pd_df):
 try:
    pd_df["IsMale"] = pd.get_dummies(pd_df["Gender"], drop_first=True)
    pd_df["IsServiceCompany"]  = pd.get_dummies(pd_df["Company Type"], drop_first=True)
    pd_df["WFH Setup Available"]  = pd.get_dummies(pd_df["WFH Setup Available"], drop_first=True)


    pd_df.drop(columns=["Gender", "Company Type"], axis=1, inplace=True)
 except:
    pass
feature_encode(employees_train_nonNull)
```

- Setelah penggantian data, data dipisah (Splitting) menjadi data training dan data valid menggunakan fungsi train_test_split dari sklearn. Rasio splitting yang dilakukan adalah 0.9 training dan 0.1 validasi. Sehingga akan ada 16731 data training dan 1859 data validasi ini, karena data training berjumlah banyak dan data validasi tidak terlalu sedikit.
- Setelah splitting, data train dan valid distandarisasi agar nilainya tidak terlalu tinggi. Standarisasi dilakukan dengan StandardScaler() 

## Modeling
Model yang dibuat akan mengunakan 3 algoritma yang dibahas dalam Machine Learning Terapan Dicoding. Algoritma tersebut adalah K-Nearest Neighbor (KNN), Random Forest, dan Ada Boosting. Disini model dengan akurasi validasi yang terbaik akan dipilih.

Ketiga algoritma tersebut sudah disediakan oleh library sklearn. Algoritma tersebut cukup dipanggil, diberi parameter yang sesuai, dilatih menggunakan data training. Berikut kode_kode pembuatan model :
```
#Training KNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, Y_train)

#Training RF
rf = RandomForestRegressor(n_estimators=40, max_depth=16, random_state=3, n_jobs=-1)
rf.fit(X_train, Y_train)

#Training Boosting
boosting=AdaBoostRegressor(learning_rate=0.01, random_state=3)
boosting.fit(X_train, Y_train)
```

## Evaluation
Metrik yang digunakan untuk kasus ini adalah MSE (Mean Squared Error). Metrik ini cukup sering dipakai untuk kasus regresi.
Berikut rumus MSE yang diambil dari Machine Learning Terapan Dicoding. [Machine Learning Terapan](https://www.dicoding.com/academies/319/tutorials/18595)

![MSE](https://i.postimg.cc/N0Wv9mXb/2021071619431112f1106e20559e77c855cea11d1b1479.jpg)

Inti rumus ini adalah menjumlahkan semua selisih hasil sesungguhnya (yi) dengan hasil prediksi(y_pred_i) yang telah dikuadratkan. Setelah semua dijumlahkan, hasil akan dibagi jumlah data (N). Semakin dekat prediksi ke hasil sesungguhnya, semakin kecil nilai MSE, dan semakin akurat prediksi model tersebut.

Berikut hasil dari model-model yang dicoba :
|             | KNN               | Random Forest       | Ada Boosting          |
| ------------- | ------------- | ------------- | ------------- | 
| param       |n_neighbor = 10 |n_estimators = 40 | learning_rate = 0.01|
|             |                   |max_depth = 16    | random_state = 3    |
|             |                   |random_state = 3  | n_neighbor = 10     |
|             |                   |n_jobs = -1       |                       |
| train_mse   | 0.003397          | 0.000823            | 0.004358              |
| val_mse     | 0.003397          | 0.003225            | 0.004572              |

![Hasil](https://i.postimg.cc/1XrkjdjY/result.png)

Berdasarkan hasil tersebut, Random Forest memiliki mean squared error yang lebih kecil dibanding algoritma lain. Maka dari itu, Random Forest dipilih menjadi model terbaik untuk kasus ini
