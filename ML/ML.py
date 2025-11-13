# contoh_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, classification_report
import joblib

# --- Contoh data sintetis (ganti dengan dataset nyata Anda) ---
np.random.seed(0)
n = 500
df = pd.DataFrame({
    'moisture': np.random.uniform(5, 40, n),
    'air_temp': np.random.uniform(15, 35, n),
    'soil_temp': np.random.uniform(12, 30, n),
    'pH': np.random.uniform(4.5, 8.5, n),
    'N_ppm': np.random.uniform(50, 300, n),
    'P_ppm': np.random.uniform(5, 100, n),
    'K_ppm': np.random.uniform(50, 400, n),
})
# buat label skor sintetis fase1 (mis. kombinasi weighted)
df['score_phase1'] = (
    0.2 * (df['N_ppm']/300) +
    0.15 * (df['P_ppm']/100) +
    0.15 * (df['K_ppm']/400) +
    0.2 * (1 - np.abs(df['pH'] - 6.5)/4) +
    0.2 * (df['moisture']/40)
) * 100 + np.random.normal(0, 5, n)
df['score_phase1'] = df['score_phase1'].clip(0,100)

# --- Fase 1: model regresi skor soil ---
features_phase1 = ['moisture','air_temp','soil_temp','pH','N_ppm','P_ppm','K_ppm']
X1 = df[features_phase1]
y1 = df['score_phase1']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1,test_size=0.2, random_state=42)

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])
preproc1 = ColumnTransformer([('num', num_pipe, features_phase1)])

pipe1 = Pipeline([
    ('pre', preproc1),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])
pipe1.fit(X1_train, y1_train)
pred1 = pipe1.predict(X1_test)
print("Fase1 RMSE:", mean_squared_error(y1_test, pred1, squared=False))

# simpan model fase1
joblib.dump(pipe1, 'model_phase1.joblib')

# --- Siapkan data untuk fase2: gabungkan output fase1 dengan fitur eksternal ---
# buat kolom eksternal sintetis
n2 = len(df)
ext = pd.DataFrame({
    'location_type': np.random.choice(['pekarangan','sawah','kebun'], n2),
    'irrigation': np.random.choice(['tadah_hujan','irigasi'], n2),
})
df2 = pd.concat([df, ext], axis=1)
# gunakan prediksi fase1 sebagai fitur
df2['score_phase1_pred'] = pipe1.predict(df2[features_phase1])

# buat label akhir (subur/kurang_subur/tidak_subur) sintetis
def map_label(row):
    s = row['score_phase1_pred']
    if s >= 65 and row['irrigation']=='irigasi':
        return 'subur'
    if s >= 50:
        return 'kurang_subur'
    return 'tidak_subur'

df2['final_label'] = df2.apply(map_label, axis=1)

# --- Fase 2: klasifikasi final_label ---
features_phase2_num = ['score_phase1_pred']
features_phase2_cat = ['location_type','irrigation']

num_pipe2 = Pipeline([('imputer', SimpleImputer(strategy='median')),
                      ('scaler', StandardScaler())])
cat_pipe2 = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                      ('ohe', OneHotEncoder(handle_unknown='ignore'))])

preproc2 = ColumnTransformer([
    ('num', num_pipe2, features_phase2_num),
    ('cat', cat_pipe2, features_phase2_cat),
])

X2 = df2[features_phase2_num + features_phase2_cat]
y2 = df2['final_label']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2,test_size=0.2, random_state=42)

pipe2 = Pipeline([
    ('pre', preproc2),
    ('model', RandomForestClassifier(n_estimators=200, random_state=42))
])
pipe2.fit(X2_train, y2_train)
pred2 = pipe2.predict(X2_test)
print("\nFase2 classification report:\n", classification_report(y2_test, pred2))

# simpan model fase2
joblib.dump(pipe2, 'model_phase2.joblib')

# --- Contoh inference function ---
def infer(sample):
    # sample: dict dengan keys: moisture,air_temp,soil_temp,pH,N_ppm,P_ppm,K_ppm, location_type, irrigation
    df_sample = pd.DataFrame([sample])
    # fase1
    score = joblib.load('model_phase1.joblib').predict(df_sample[features_phase1])[0]
    df_sample['score_phase1_pred'] = score
    # fase2
    final = joblib.load('model_phase2.joblib').predict(df_sample[['score_phase1_pred','location_type','irrigation']])[0]
    # rekomendasi sederhana rule-based
    if final == 'subur':
        rec = ['Padi', 'Jagung'] if sample['location_type']=='sawah' else ['Cabai','Tomat']
    elif final == 'kurang_subur':
        rec = ['Kacang-kacangan', 'Kedelai']
    else:
        rec = ['Tanam Penutup Tanah, Beri Pupuk Organik']
    return {'score_phase1': score, 'final_label': final, 'recommended': rec}

# contoh inference
samp = {
    'moisture': 30, 'air_temp': 28, 'soil_temp': 26, 'pH': 6.5, 'N_ppm': 180, 'P_ppm': 40, 'K_ppm': 200,
    'location_type':'sawah', 'irrigation':'irigasi'
}
print("\nContoh inference:", infer(samp))
