# %%
# ======================================
# 1. IMPORT LIBRARY
# ======================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from joblib import dump, load
from math import sqrt

# ======================================
# 2. LOAD DATA
# ======================================
df = pd.read_csv('Crop_recommendation.csv')

# ======================================
# 3. BERSIHKAN DAN CEK DATA
# ======================================
df = df.dropna()  # hapus baris kosong

# ======================================
# 4. BUAT SKOR SUBUR (DENGAN NORMALISASI CURAH HUJAN)
# ======================================

# Pertama, normalisasi semua fitur numerik antara 0-100
numeric_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

df_norm = df.copy()

for col in numeric_features:
    min_val = df[col].min()
    max_val = df[col].max()
    # normalisasi ke skala 0-100
    df_norm[col] = ((df[col] - min_val) / (max_val - min_val)) * 100

# Buat skor subur berdasarkan fitur yang sudah dinormalisasi
df_norm['score'] = (
    df_norm['N']*0.2 +
    df_norm['P']*0.2 +
    df_norm['K']*0.2 +
    df_norm['temperature']*0.1 +
    df_norm['humidity']*0.1 +
    df_norm['ph']*0.1 +
    df_norm['rainfall']*0.1
)

# ======================================
# 5. KLASIFIKASI SUBUR / KURANG / TIDAK SUBUR
# ======================================
def fertility_label(score):
    if score >= 70:
        return 'subur'
    elif score >= 40:
        return 'kurang_subur'
    else:
        return 'tidak_subur'

df_norm['fertility'] = df_norm['score'].apply(fertility_label)

# ======================================
# 6. PHASE 1 - PREDIKSI SCORE DENGAN RANDOM FOREST REGRESSOR
# ======================================
X = df_norm[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df_norm['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(n_estimators=200, random_state=42)
regressor.fit(X_train, y_train)
pred1 = regressor.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, pred1))

# ======================================
# 7. PHASE 2 - PREDIKSI KELAS FERTILITY
# ======================================
X2 = df_norm[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y2 = df_norm['fertility']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)
clf.fit(X2_train, y2_train)

pred2 = clf.predict(X2_test)
accuracy = accuracy_score(y2_test, pred2)

report = classification_report(y2_test, pred2)
matrix = confusion_matrix(y2_test, pred2)

print("Phase1 RMSE:", round(rmse, 4))
print("Phase2 Accuracy (Balanced):", round(accuracy, 4))
print("\n=== Classification Report ===")
print(report)
print("\n=== Confusion Matrix ===")
print(matrix)

# ======================================
# 8. SIMPAN MODEL
# ======================================
dump(regressor, 'model_phase1_rf.joblib')
dump(clf, 'model_phase2_rf.joblib')

print("Phase1 RMSE:", rmse)
print("Phase2 Accuracy:", accuracy)
print(report)

print("\nJumlah masing-masing kategori kesuburan:")
print(df_norm['fertility'].value_counts())

print("\nContoh data tanah:")
print("\nContoh subur:")
print(df_norm[df_norm['fertility'] == 'subur'].head())
print("\nContoh kurang subur:")
print(df_norm[df_norm['fertility'] == 'kurang_subur'].head())
print("\nContoh tidak subur:")
print(df_norm[df_norm['fertility'] == 'tidak_subur'].head())

# %%
# ======================================
# 9. REKOMENDASI TANAMAN
# ======================================

# Karena dataset tidak punya kolom lokasi, kita hilangkan bagian location_detail
recommendations_by_fertility = (
    df_norm.groupby('fertility')['label']
      .apply(lambda x: x.value_counts().index.tolist())
      .to_dict()
)

# Tambahkan RESTRIKSI lokasi → agar rekomendasi masuk akal
allowed_crops_by_location = {
    'pekarangan': ['banana', 'papaya', 'mung bean', 'blackgram', 'lentil', 'chickpea'],
    'sawah-irigasi': ['rice', 'maize', 'sugarcane'],
    'sawah-tadah_hujan': ['rice', 'maize', 'sorghum', 'groundnut'],
    'kebun-dataran-tinggi': ['coffee', 'tea', 'orange', 'apple', 'potato'],
    'kebun-dataran-rendah': ['banana', 'papaya', 'coconut', 'cassava', 'sweet potato']
}

# Tambahkan kamus terjemahan tanaman
translate_crop = {
    'rice': 'padi',
    'maize': 'jagung',
    'chickpea': 'kacang arab',
    'kidneybeans': 'kacang merah',
    'pigeonpeas': 'kacang gude',
    'mothbeans': 'kacang mot',
    'mung bean': 'kacang hijau',
    'blackgram': 'kacang hitam',
    'lentil': 'kacang lentil',
    'pomegranate': 'delima',
    'banana': 'pisang',
    'mango': 'mangga',
    'grapes': 'anggur',
    'watermelon': 'semangka',
    'muskmelon': 'blewah',
    'apple': 'apel',
    'orange': 'jeruk',
    'papaya': 'pepaya',
    'coconut': 'kelapa',
    'cotton': 'kapas',
    'jute': 'goni',
    'coffee': 'kopi',
    'tea': 'teh',
    'sugarcane': 'tebu',
    'sorghum': 'sorgum',
    'groundnut': 'kacang tanah',
    'potato': 'kentang',
    'sweet potato': 'ubi jalar',
    'cassava': 'singkong'
}

# ======================================
# 10. FUNGSI PREDIKSI
# ======================================
def prediksi_tanah(input_data):
    X1 = np.array([[ 
        input_data['N'], input_data['P'], input_data['K'],
        input_data['temperature'], input_data['humidity'],
        input_data['ph'], input_data['rainfall']
    ]])

    score = regressor.predict(X1)[0]

    if score >= 70:
        fertility_class = 'subur'
    elif score >= 40:
        fertility_class = 'kurang_subur'
    else:
        fertility_class = 'tidak_subur'

    lokasi = input_data['location_detail']

    # Ambil tanaman yang cocok berdasarkan kesuburan
    tanaman = recommendations_by_fertility.get(fertility_class, [])

    # Filter tanaman yang cocok dengan lokasi
    tanaman = [t for t in tanaman if t in allowed_crops_by_location[lokasi]]

    # Jika kosong, ambil tanaman default dari lokasi
    if not tanaman:
        tanaman = allowed_crops_by_location[lokasi]

    return score, fertility_class, tanaman

# ======================================
# 10.1. FUNGSI MENGAMBIL CURAH HUJAN DARI API
# ======================================
import requests, time

def get_rainfall_by_location(nama_daerah):
    try:
        # Ambil koordinat dari OpenStreetMap
        geo_url = f"https://nominatim.openstreetmap.org/search?format=json&q={nama_daerah}, Indonesia"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MLApp/1.0; +https://example.com)"}
        geo_data = requests.get(geo_url, headers=headers).json()

        if not geo_data:
            print("❌ Daerah tidak ditemukan.")
            return None

        lat = float(geo_data[0]['lat'])
        lon = float(geo_data[0]['lon'])
        print(f"📍 Lokasi ditemukan: {nama_daerah} (Lat: {lat}, Lon: {lon})")

        # Tunggu 1 detik agar tidak kena rate limit
        time.sleep(1)

        # Ambil data curah hujan dari Open-Meteo
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum&timezone=Asia%2FJakarta"
        data = requests.get(url).json()

        # Debugging output
        if "daily" not in data or "precipitation_sum" not in data["daily"]:
            print("⚠️ Data curah hujan tidak ditemukan:", data)
            return None

        rainfall_today = float(data["daily"]["precipitation_sum"][0])
        print(f"🌧️ Curah hujan hari ini di {nama_daerah}: {rainfall_today} mm")
        return rainfall_today

    except Exception as e:
        print("❌ Gagal mengambil data curah hujan:", e)
        return None


# %%
# ======================================
# 11. INPUT MANUAL (versi diperbarui)
# ======================================
def input_manual():
    print("\n=== INPUT DATA TANAH ===")
    N = float(input("N: "))
    P = float(input("P: "))
    K = float(input("K: "))
    temperature = float(input("Suhu (°C): "))
    humidity = float(input("Kelembapan (%): "))
    ph = float(input("pH: "))

    # Input koordinat lokasi (tidak dibulatkan)
    koordinat = input("\nMasukkan Koordinat (format: latitude,longitude), contoh: -8.190234,111.922583: ")
    try:
        lat, lon = map(float, koordinat.split(','))
        print(f"📍 Koordinat diterima: Latitude = {lat}, Longitude = {lon}")
    except:
        print("❌ Format koordinat salah. Gunakan format: -8.190234,111.922583")
        exit()

    # Ambil nama daerah detail dari koordinat (reverse geocoding)
    try:
        geo_url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MLApp/1.0; +https://example.com)"}
        geo_data = requests.get(geo_url, headers=headers).json()
        nama_daerah = geo_data.get('display_name', 'Tidak diketahui')
        print(f"📍 Lokasi terdeteksi: {nama_daerah}")
    except Exception as e:
        print("❌ Gagal mendapatkan nama daerah:", e)
        nama_daerah = "Tidak diketahui"

    # Ambil curah hujan otomatis berdasarkan koordinat
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum&timezone=Asia%2FJakarta"
        data = requests.get(url).json()
        rainfall = float(data["daily"]["precipitation_sum"][0])
        print(f"🌧️ Curah hujan hari ini di {nama_daerah}: {rainfall} mm")
    except Exception as e:
        print("❌ Gagal mengambil curah hujan otomatis:", e)
        rainfall = float(input("Masukkan curah hujan manual (mm): "))

    # Pilih jenis lokasi
    print("\nPilih Lokasi:")
    print("1. pekarangan")
    print("2. sawah-irigasi")
    print("3. sawah-tadah_hujan")
    print("4. kebun-dataran-tinggi")
    print("5. kebun-dataran-rendah")

    pilihan = input("Masukkan nomor: ")
    lokasi_map = {
        '1': 'pekarangan',
        '2': 'sawah-irigasi',
        '3': 'sawah-tadah_hujan',
        '4': 'kebun-dataran-tinggi',
        '5': 'kebun-dataran-rendah'
    }
    location_detail = lokasi_map.get(pilihan, 'pekarangan')

    return {
        'N': N, 'P': P, 'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall,
        'location_detail': location_detail
    }




data_saya = input_manual()
score, kategori, rekom = prediksi_tanah(data_saya)

print("\n=== HASIL ===")
print("Skor:", round(score, 2))
print("Lokasi:", data_saya['location_detail'])
print("Kategori:", kategori)
print("Rekomendasi tanaman yang cocok:")
for t in rekom:
    nama_id = translate_crop.get(t, t)
    print(f"- {nama_id} ({t})")

# %%
