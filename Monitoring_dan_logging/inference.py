# 7.inference.py
import requests
import json
import time

# Alamat endpoint server model MLflow
url = "http://127.0.0.1:5002/invocations"

# Header untuk permintaan POST
headers = {
    "Content-Type": "application/json"
}

# Membaca data input dari file
with open("input.json", "r") as f:
    input_data = json.load(f)

print("Starting inference script...")
print(f"Sending requests to {url}")

# Loop untuk mengirim permintaan secara terus-menerus
while True:
    try:
        response = requests.post(url, headers=headers, data=json.dumps(input_data))
        response.raise_for_status() # Akan error jika status code bukan 2xx
        
        predictions = response.json()
        print(f"Successfully received predictions: {predictions}")
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    # Jeda selama 2 detik sebelum mengirim permintaan berikutnya
    time.sleep(2)