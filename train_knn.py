import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier  

# Đọc và xử lý dữ liệu
df = pd.read_csv('crx.csv', header=None, na_values='?')
if df.iloc[0].astype(str).str.contains('A2').any():
    df = df.iloc[1:].reset_index(drop=True)
df.columns = ['A'+str(i) for i in range(1, 17)]

for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = df[col].astype('float64')
        except:
            df[col] = df[col].fillna(df[col].mode()[0])
    if df[col].dtype == 'float64':
        df[col] = df[col].fillna(df[col].mean())

df['label'] = LabelEncoder().fit_transform(df['A16'])
le = LabelEncoder()
for col in ['A1', 'A9', 'A10', 'A12']:
    df[col] = le.fit_transform(df[col])
df = pd.get_dummies(df, columns=['A4','A5','A6','A7','A13'], drop_first=False)

X_raw = df.drop(columns=['A16', 'label'])
y = df['label'].values

model = KNeighborsClassifier(n_neighbors=5)
results = []

# Huấn luyện 10 lần
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state= i + 42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results.append(acc)

# Kết quả
print("Naive Bayes:")
for i, acc in enumerate(results, 1):
    print(f"  Lần {i:2d}: {acc:.4f}")
avg_acc = np.mean(results)
print(f"✅ Accuracy trung bình: {avg_acc:.4f}")

# Vẽ biểu đồ độ chính xác qua 10 lần huấn luyện
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), results, marker='o', linestyle='-', color='blue', label='Accuracy từng lần')
plt.axhline(avg_acc, color='red', linestyle='--', label=f'Accuracy trung bình: {avg_acc:.4f}')
plt.title('Biểu đồ độ chính xác của KNN qua 10 lần chạy')
plt.xlabel('Lần chạy')
plt.ylabel('Accuracy')
plt.xticks(range(1, 11))
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Lưu model

joblib.dump({
    'model': model,
    'scaler': scaler,
    'feature_names': X_raw.columns.tolist()
}, 'knn_model.pkl')
print("💾 Đã lưu model vào 'knn_model.pkl'")
