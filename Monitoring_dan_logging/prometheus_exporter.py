# prometheus_exporter.py

from flask import Flask, request, jsonify, Response
import requests
import time
import pandas as pd
import psutil  # Untuk monitoring sistem
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# --- 1. Inisialisasi Aplikasi dan Metrik ---
app = Flask(__name__)

# Metrik untuk API Gateway
REQUEST_COUNT = Counter(
    'gateway_requests_total', 
    'Total HTTP Requests received by the gateway'
)
REQUEST_LATENCY = Histogram(
    'gateway_request_duration_seconds', 
    'HTTP Request latency at the gateway'
)

# Metrik untuk monitoring sistem di mana gateway berjalan
CPU_USAGE = Gauge('gateway_system_cpu_usage_percent', 'Gateway CPU Usage Percentage')
RAM_USAGE = Gauge('gateway_system_ram_usage_percent', 'Gateway RAM Usage Percentage')

# --- 2. Fungsi Preprocessing (dari langkah sebelumnya) ---
# Definisikan pemetaan dan urutan kolom yang benar
GENDER_MAP = {'Male': 1, 'Female': 0}
DEVICE_MAP = {'Laptop': 0, 'Smartphone': 1, 'TV': 2, 'Tablet': 3}
LOCATION_MAP = {'Urban': 1, 'Rural': 0}

MODEL_FEATURES = [
    'Age', 'Gender', 'Avg_Daily_Screen_Time_hr', 'Primary_Device',
    'Educational_to_Recreational_Ratio', 'Urban_or_Rural', 'Anxiety',
    'Eye Strain', 'Obesity Risk', 'Poor Sleep'
]

def preprocess_input(data):
    """Mengubah data mentah menjadi format yang siap untuk model."""
    df = pd.DataFrame(data)
    df['Gender'] = df['Gender'].map(GENDER_MAP)
    df['Primary_Device'] = df['Primary_Device'].map(DEVICE_MAP)
    df['Urban_or_Rural'] = df['Urban_or_Rural'].map(LOCATION_MAP)
    
    if 'Health_Impacts' in df.columns:
        health_impacts_dummies = df['Health_Impacts'].str.get_dummies(sep=', ')
        df = pd.concat([df, health_impacts_dummies], axis=1)
        df = df.drop('Health_Impacts', axis=1)

    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0
            
    return df[MODEL_FEATURES]

# --- 3. Endpoint untuk Prometheus ---
@app.route('/metrics', methods=['GET'])
def metrics():
    """Menyajikan metrik untuk di-scrape oleh Prometheus."""
    # Update metrik sistem setiap kali endpoint ini diakses
    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# --- 4. Endpoint Prediksi (Gateway) ---
@app.route('/predict', methods=['POST'])
def predict_gateway():
    """Menerima data mentah, memprosesnya, mengirim ke model, dan mencatat metrik."""
    start_time = time.time()
    REQUEST_COUNT.inc()

    # URL dari server model MLflow Anda yang sebenarnya
    # Pastikan port-nya benar
    model_api_url = "http://127.0.0.1:5002/invocations"
    
    try:
        raw_data = request.get_json()
        if not raw_data or 'data' not in raw_data:
            return jsonify({"error": "Format JSON tidak valid. Harap sediakan key 'data'."}), 400

        # Langkah 1: Preprocess data mentah
        processed_df = preprocess_input(raw_data['data'])

        # Langkah 2: Siapkan data dalam format yang diterima MLflow (`dataframe_split`)
        mlflow_input = {
            "dataframe_split": {
                "columns": processed_df.columns.tolist(),
                "data": processed_df.values.tolist()
            }
        }

        # Langkah 3: Kirim data yang sudah bersih ke server model
        response = requests.post(model_api_url, json=mlflow_input)
        response.raise_for_status()  # Error jika respons bukan 2xx
        
        # Langkah 4: Catat latensi total (preprocessing + request model)
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)
        
        # Langkah 5: Kembalikan respons dari model ke pengguna
        return jsonify(response.json())

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Tidak dapat terhubung ke server model: {e}"}), 503
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan di gateway: {str(e)}"}), 500

# --- 5. Jalankan Aplikasi ---
if __name__ == '__main__':
    # Jalankan gateway di port 8001
    app.run(host='0.0.0.0', port=8001, debug=True)