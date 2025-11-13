import streamlit as st
import numpy as np
from joblib import load
import requests
from datetime import datetime, timedelta

# ======================================
# LOAD MODEL
# ======================================
regressor = load('model_phase1_rf.joblib')
clf = load('model_phase2_rf.joblib')

# ======================================
# DATA PENDUKUNG
# ======================================
allowed_crops_by_location = {
    'pekarangan': ['banana', 'papaya', 'mung bean', 'blackgram', 'lentil', 'chickpea'],
    'sawah-irigasi': ['rice', 'maize', 'sugarcane'],
    'sawah-tadah_hujan': ['rice', 'maize', 'sorghum', 'groundnut'],
    'kebun-dataran-tinggi': ['coffee', 'tea', 'orange', 'apple', 'potato'],
    'kebun-dataran-rendah': ['banana', 'papaya', 'coconut', 'cassava', 'sweet potato']
}

translate_crop = {
    'rice': 'padi', 'maize': 'jagung', 'chickpea': 'kacang arab',
    'mung bean': 'kacang hijau', 'blackgram': 'kacang hitam',
    'banana': 'pisang', 'papaya': 'pepaya', 'coffee': 'kopi',
    'tea': 'teh', 'apple': 'apel', 'orange': 'jeruk',
    'potato': 'kentang', 'cassava': 'singkong', 'sweet potato': 'ubi jalar'
}

# ======================================
# FUNGSI: Ambil nama daerah dari koordinat
# ======================================
def get_location_name_from_coords(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=14&addressdetails=1"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MLApp/1.0; +https://example.com)"}
        response = requests.get(url, headers=headers).json()

        if "address" in response:
            addr = response["address"]

            parts = []
            for key in ["village", "suburb", "town", "city_district", "city", "county", "state"]:
                if addr.get(key):
                    parts.append(addr[key])

            nama_daerah = ", ".join(parts + ["Indonesia"])
            return nama_daerah
        else:
            return "Daerah tidak diketahui"
    except Exception:
        return "Gagal mengambil nama daerah"

# ======================================
# FUNGSI: Ambil rata-rata curah hujan 30 hari
# ======================================
def get_rainfall_avg_30days(lat, lon):
    try:
        today = datetime.utcnow().date()
        start_date = today - timedelta(days=30)

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=precipitation_sum"
            f"&timezone=Asia%2FJakarta"
            f"&start_date={start_date}&end_date={today}"
        )
        weather_data = requests.get(weather_url).json()

        if "daily" not in weather_data or "precipitation_sum" not in weather_data["daily"]:
            return None

        rainfall_list = weather_data["daily"]["precipitation_sum"]
        rainfall_avg = sum(rainfall_list) / len(rainfall_list)
        return float(rainfall_avg)

    except Exception as e:
        st.warning(f"Gagal mengambil data curah hujan: {e}")
        return None

# ======================================
# FUNGSI PREDIKSI
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
    tanaman = allowed_crops_by_location[lokasi]
    return score, fertility_class, tanaman

# ======================================
# STREAMLIT UI
# ======================================
st.set_page_config(page_title="Prediksi Kesuburan Tanah", page_icon="🌱")
st.title("🌾 Prediksi Kesuburan Tanah & Rekomendasi Tanaman")
st.markdown("Masukkan parameter tanah dan **koordinat (latitude, longitude)** untuk mengambil curah hujan dan nama daerah secara otomatis.")

# Input form
with st.form("tanah_form"):
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Nitrogen (N)", 0.0, 150.0, 50.0)
        P = st.number_input("Phosphor (P)", 0.0, 150.0, 50.0)
        K = st.number_input("Kalium (K)", 0.0, 150.0, 50.0)
        ph = st.number_input("pH Tanah", 0.0, 14.0, 6.5)
    with col2:
        temperature = st.number_input("Suhu (°C)", 0.0, 50.0, 25.0)
        humidity = st.number_input("Kelembapan (%)", 0.0, 100.0, 60.0)

        koordinat_str = st.text_input(
            "Masukkan Koordinat (format: latitude, longitude)",
            "-6.914744, 107.609810"
        )

    lokasi = st.selectbox(
        "Pilih Jenis Lahan",
        ['pekarangan', 'sawah-irigasi', 'sawah-tadah_hujan', 'kebun-dataran-tinggi', 'kebun-dataran-rendah']
    )

    submitted = st.form_submit_button("🔍 Prediksi")

if submitted:
    try:
        lat_str, lon_str = koordinat_str.split(",")
        latitude = float(lat_str.strip())
        longitude = float(lon_str.strip())
    except:
        st.error("⚠️ Format koordinat salah! Gunakan format: -6.914744, 107.609810")
        st.stop()

    rainfall = get_rainfall_avg_30days(latitude, longitude)
    nama_daerah = get_location_name_from_coords(latitude, longitude)

    if rainfall is None:
        st.error("Tidak dapat mengambil data curah hujan. Coba periksa koordinat yang dimasukkan.")
    else:
        st.info(f"📍 Lokasi: {nama_daerah} ({latitude}, {longitude})")
        st.write(f"🌧️ Rata-rata curah hujan 30 hari terakhir: **{rainfall:.2f} mm**")

        input_data = {
            'N': N, 'P': P, 'K': K,
            'temperature': temperature, 'humidity': humidity,
            'ph': ph, 'rainfall': rainfall,
            'location_detail': lokasi
        }

        score, kategori, rekom = prediksi_tanah(input_data)

        st.success("✅ Prediksi Berhasil!")
        st.subheader("📊 Hasil Prediksi")
        st.write(f"**Skor Kesuburan:** {score:.2f}")
        st.write(f"**Kategori Kesuburan:** `{kategori}`")
        st.write(f"**Jenis Lahan:** `{lokasi}`")

        st.subheader("🌱 Rekomendasi Tanaman")
        for t in rekom:
            st.write(f"- {translate_crop.get(t, t).capitalize()} ({t})")

st.markdown("---")
st.caption("© 2025 - Aplikasi Rekomendasi Tanaman berbasis Machine Learning + Open-Meteo API + Nominatim Reverse Geocoding")
