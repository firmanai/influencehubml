from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model_path = 'my_model.h5'
model = load_model(model_path, compile=False)

# Jika Anda menggunakan scaler, aktifkan baris di bawah dan sesuaikan
# scaler = StandardScaler()

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            data = request.get_json(force=True)
            new_follower = float(data.get("followers"))
            new_er = float(data.get("er"))

            input_data = np.array([[new_follower, new_er]])

            # Normalisasi atau standarisasi (jika diperlukan)
            # input_data_scaled = scaler.transform(input_data)

            # Lakukan prediksi dengan model
            prediction = model.predict(input_data)

            # Inversi transformasi jika diperlukan
            # prediction = scaler.inverse_transform(prediction)

            # Konversi nilai float32 menjadi float64 untuk dapat di-serialize ke JSON
            harga_prediction = float(prediction[0][0])

            # Kembalikan hasil prediksi sebagai respons JSON
            return jsonify({'harga': harga_prediction})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None,
        }), 405

if __name__ == '__main__':
    app.run(debug=True)
