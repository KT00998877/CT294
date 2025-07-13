from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Đọc dữ liệu gốc để lấy tên cột
df = pd.read_csv('crx.csv')
columns = df.drop(columns=['A16']).columns.tolist()

# Lấy các giá trị gợi ý cho dropdown
unique_values = {}
for col in columns:
    if df[col].dtype == 'object':
        unique_values[col] = sorted(df[col].dropna().unique().tolist())
    else:
        unique_values[col] = None

# Load mô hình đã huấn luyện
data = joblib.load('best_model.pkl')
model = data['model']
scaler = data['scaler']
feature_names = data['feature_names']
le_y = data['label_encoder_y'] 

@app.route('/')
def index():
    return render_template('index.html', columns=columns, unique_values=unique_values)

@app.route('/predict', methods=['POST'])
def predict():
    input_raw = {}
    for col in columns:
        val = request.form.get(col)
        try:
            input_raw[col] = val if unique_values[col] else float(val)
        except:
            return render_template('result.html', prediction=f"❌ Giá trị không hợp lệ ở '{col}': {val}")

    try:
        # Tạo DataFrame từ input người dùng
        X_df = pd.DataFrame([input_raw])

        # Mã hóa one-hot
        X_encoded = pd.get_dummies(X_df)

        # Khớp lại cột với model
        X_encoded = X_encoded.reindex(columns=feature_names, fill_value=0)

        # Chuẩn hóa
        X_scaled = scaler.transform(X_encoded)

        # Dự đoán
        y_pred = model.predict(X_scaled)

        # ✅ GIẢI MÃ kết quả về '+' hoặc '-'
        predicted_label = le_y.inverse_transform(y_pred)[0]

        prediction = '✅ ĐƯỢC DUYỆT' if predicted_label == '+' else '❌ KHÔNG DUYỆT'

    except Exception as e:
        prediction = f"❌ Lỗi xử lý dữ liệu: {e}"

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
