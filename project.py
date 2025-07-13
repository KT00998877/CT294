import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Đọc dữ liệu
df = pd.read_csv('crx.csv', header=None, na_values='?')

# Loại bỏ header dòng đầu (nếu có)
if df.iloc[0].astype(str).str.contains('A2').any():
    df = df.iloc[1:].reset_index(drop=True)

# Gán tên cột
df.columns = ['A'+str(i) for i in range(1, 17)]

# Xử lý missing và kiểu dữ liệu
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = df[col].astype('float64')
        except:
            df[col] = df[col].fillna(df[col].mode()[0])
    if df[col].dtype == 'float64':
        df[col] = df[col].fillna(df[col].mean())

# Mã hóa nhãn
df['label'] = LabelEncoder().fit_transform(df['A16'])

# Mã hóa nhị phân
le = LabelEncoder()
for col in ['A1', 'A9', 'A10', 'A12']:
    df[col] = le.fit_transform(df[col])

# One-hot encoding các cột phân loại
df = pd.get_dummies(df, columns=['A4','A5','A6','A7','A13'], drop_first=False)

# Tách đặc trưng và nhãn
feature_cols = [c for c in df.columns if c not in ['A16', 'label']]
X_raw = df[feature_cols].copy()
y = df['label'].values

# Mô hình cần đánh giá
models = {
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(criterion='entropy', random_state=0)
}

results = {name: [] for name in models}
avg_results = {}

# Đánh giá 10 lần
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=i)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results[name].append(acc)

# In kết quả
print("🎯 Kết quả từng lần và độ chính xác trung bình:\n")
for name, scores in results.items():
    avg = np.mean(scores)
    print(f"{name} (avg: {avg:.4f})")
    for i, acc in enumerate(scores, 1):
        print(f"  Lần {i:2d}: {acc:.4f}")
    avg_results[name] = avg
    print()

# Chọn mô hình tốt nhất
best_model_name = max(avg_results, key=avg_results.get)
print(f"🏆 Mô hình tốt nhất: {best_model_name} với accuracy trung bình: {avg_results[best_model_name]:.4f}")
final_model = models[best_model_name]

# Huấn luyện lại trên toàn bộ tập
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
final_model.fit(X_scaled, y)

# Lưu feature names để Flask predict khớp
feature_names = X_raw.columns.tolist()

# Lưu mô hình + scaler + tên cột
joblib.dump({
    'model': final_model,
    'scaler': scaler,
    'feature_names': feature_names
}, 'best_model.pkl')

print("💾 Đã lưu mô hình vào 'best_model.pkl'")

# Vẽ biểu đồ
plt.figure(figsize=(8,5))
plt.bar(avg_results.keys(), avg_results.values(), color='teal')
plt.title('So sánh độ chính xác trung bình của các mô hình (10 lần chạy)')
plt.xlabel('Mô hình')
plt.ylabel('Accuracy trung bình')
plt.ylim(0, 1)
for i, v in enumerate(avg_results.values()):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
plt.tight_layout()
plt.show()
