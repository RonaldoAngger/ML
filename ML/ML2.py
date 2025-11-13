# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from joblib import dump, load
from math import sqrt

# ======================================
# 1. LOAD DATA
# ======================================
df = pd.read_csv('Crop_recommendation.csv')

# ======================================
# 2. HITUNG SKOR TANAH
# ======================================
def compute_soil_score(row):
    N,P,K,pH,temp,hum,rain = row['N'],row['P'],row['K'],row['ph'],row['temperature'],row['humidity'],row['rainfall']
    score = (N*0.15 + P*0.15 + K*0.15 + (100-abs(pH-7)*10)*0.1 + (100-temp)*0.1 + hum*0.1 + rain*0.15)
    return np.clip(score,0,100)

df['soil_score'] = df.apply(compute_soil_score,axis=1)

# ======================================
# 3. KATEGORI KESUBURAN
# ======================================
def classify_fertility(score):
    if score >= 70: return 'subur'
    elif score >= 40: return 'kurang_subur'
    return 'tidak_subur'

df['fertility'] = df['soil_score'].apply(classify_fertility)

# ======================================
# 4. TENTUKAN LOKASI LAHAN LOGIS
# ======================================
def assign_location(row):
    temp = row['temperature']
    rain = row['rainfall']
    ph = row['ph']

    if temp < 22:
        return 'kebun-dataran-tinggi'

    if temp >= 22 and rain < 100:
        return 'kebun-dataran-rendah'

    if 5.5 <= ph <= 7 and rain >= 200:
        return 'sawah-irigasi'

    if 100 <= rain < 200:
        return 'sawah-tadah_hujan'

    return 'pekarangan'

df['location_detail'] = df.apply(assign_location, axis=1)

# ======================================
# 5. TRAIN MODEL FASE 1
# ======================================
X1 = df[['N','P','K','temperature','humidity','ph','rainfall']]
y1 = df['soil_score']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1,test_size=0.2,random_state=42)

reg = RandomForestRegressor(n_estimators=200,random_state=42)
reg.fit(X1_train,y1_train)
pred1 = reg.predict(X1_test)
rmse = sqrt(mean_squared_error(y1_test, pred1))

# ======================================
# 6. ENCODE LOCATION + TRAIN FASE 2
# ======================================
enc = OneHotEncoder(sparse_output=False)
loc_encoded = enc.fit_transform(df[['location_detail']])

X2 = np.column_stack([df['soil_score'], loc_encoded])
y2 = df['fertility']

X2_train,X2_test,y2_train,y2_test = train_test_split(X2,y2,test_size=0.2,random_state=42)

clf = RandomForestClassifier(n_estimators=200,random_state=42)
clf.fit(X2_train,y2_train)
pred2 = clf.predict(X2_test)

accuracy = accuracy_score(y2_test,pred2)
report = classification_report(y2_test,pred2)

# ======================================
# 7. SIMPAN MODEL
# ======================================
dump(reg,'model_phase1_rf.joblib')
dump(enc,'encoder_loc.joblib')
dump(clf,'model_phase2_rf.joblib')

print("Phase1 RMSE:", rmse)
print("Phase2 Accuracy:", accuracy)
print(report)


print("\nJumlah masing-masing kategori kesuburan:")
print(df['fertility'].value_counts())


print("\nContoh data tanah:")
print("\nContoh subur:")
print(df[df['fertility'] == 'subur'].head())
print("\nContoh kurang subur:")
print(df[df['fertility'] == 'kurang_subur'].head())
print("\nContoh tidak subur:")
print(df[df['fertility'] == 'tidak_subur'].head())


# %%
# ======================================
# 8. REKOMENDASI TANAMAN
# ======================================

# Buat rekomendasi berdasarkan data
recommendations_by_location = (
    df.groupby(['fertility', 'location_detail'])['label']
      .apply(lambda x: x.value_counts().index.tolist())
      .to_dict()
)

recommendations_by_fertility = (
    df.groupby('fertility')['label']
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

# ======================================
# 9. FUNGSI PREDIKSI
# ======================================
def prediksi_tanah(input_data):
    X1 = np.array([[
        input_data['N'], input_data['P'], input_data['K'],
        input_data['temperature'], input_data['humidity'],
        input_data['ph'], input_data['rainfall']
    ]])

    score = reg.predict(X1)[0]
    loc = enc.transform([[input_data['location_detail']]])
    X2 = np.column_stack([score, loc])

    if score >= 70:
        fertility_class = 'subur'
    elif score >= 40:
        fertility_class = 'kurang_subur'
    else:
        fertility_class = 'tidak_subur'

    lokasi = input_data['location_detail']

    if (fertility_class, lokasi) in recommendations_by_location:
        tanaman = recommendations_by_location[(fertility_class, lokasi)]
    else:
        tanaman = recommendations_by_fertility[fertility_class]

    tanaman = [t for t in tanaman if t in allowed_crops_by_location[lokasi]]

    if not tanaman:
        tanaman = allowed_crops_by_location[lokasi]

    return score, fertility_class, tanaman


# %%
# ======================================
# 10. INPUT MANUAL
# ======================================
def input_manual():
    print("\n=== INPUT DATA TANAH ===")
    N = float(input("N: "))
    P = float(input("P: "))
    K = float(input("K: "))
    temperature = float(input("Suhu (°C): "))
    humidity = float(input("Kelembapan (%): "))
    ph = float(input("pH: "))
    rainfall = float(input("Curah Hujan (mm): "))

    print("\nPilih Lokasi:")
    print("1. pekarangan")
    print("2. sawah-irigasi")
    print("3. sawah-tadah_hujan")
    print("4. kebun-dataran-tinggi")
    print("5. kebun-dataran-rendah")

    pilihan = input("Masukkan nomor: ")
    lokasi_map = {
        '1':'pekarangan','2':'sawah-irigasi','3':'sawah-tadah_hujan',
        '4':'kebun-dataran-tinggi','5':'kebun-dataran-rendah'
    }
    location_detail = lokasi_map.get(pilihan,'pekarangan')

    return {
        'N':N,'P':P,'K':K,'temperature':temperature,'humidity':humidity,
        'ph':ph,'rainfall':rainfall,'location_detail':location_detail
    }

data_saya = input_manual()
score, kategori, rekom = prediksi_tanah(data_saya)

print("\n=== HASIL ===")
print("Skor:", round(score,2))
print("lokasi:", data_saya['location_detail'])
print("Kategori:", kategori)
print("Rekomendasi:")
for t in rekom:
    print("-", t)



# %%
