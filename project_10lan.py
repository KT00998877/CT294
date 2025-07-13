import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Đọc dữ liệu
original_df = pd.read_csv('crx.csv')  
df = original_df.copy()    

# Xử lý dữ liệu bị thiếu
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

# Encode nhãn A16
le_y = LabelEncoder()
le_y.fit(['-', '+'])  # ép '+' là 0, '-' là 1
df['A16'] = le_y.transform(df['A16'])
label_mapping = dict(zip(le_y.classes_, le_y.transform(le_y.classes_)))
print("📌 Ánh xạ nhãn A16:", label_mapping)

# One-Hot Encoding các cột object (trừ A16)
df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns, drop_first=True)

# Tách X và y
X = df.drop(columns=['A16'])
y = df['A16']

# Khởi tạo mô hình
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),
    'Naive Bayes': GaussianNB()
}

results = {name: [] for name in models}  # Lưu kết quả của từng lần chạy

# Lặp lại 10 lần
for run in range(1, 11):
    # Chia dữ liệu train/test ngẫu nhiên mỗi lần
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state= run + 42
    )

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Đánh giá từng mô hình
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results[name].append(acc)

# Hiển thị kết quả từng lần và trung bình
print("Kết quả từng lần và trung bình:\n")
for name, acc_list in results.items():
    print(f"{name}:")
    for i, acc in enumerate(acc_list, 1):
        print(f"   Lần {i:2d}: {acc:.4f}")
    print(f"   Trung bình: {np.mean(acc_list):.4f}\n")

# Tạo DataFrame kết quả trung bình
avg_results = {name: np.mean(acc_list) for name, acc_list in results.items()}
df_results = pd.DataFrame(
    [(k, v) for k, v in avg_results.items()],
    columns=['Model', 'Average Accuracy']
)

# Vẽ biểu đồ
plt.figure(figsize=(8, 5))
plt.bar(df_results['Model'], df_results['Average Accuracy'], color=['blue'])
plt.xlabel('Model')
plt.ylabel('Average Accuracy')
plt.title(' Độ chính xác trung bình của các mô hình ')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

best_model_name = max(avg_results, key=avg_results.get)
best_acc = avg_results[best_model_name]

print(f"Mô hình tốt nhất: {best_model_name} với độ chính xác trung bình {best_acc:.4f}")

# Train mô hình tốt nhất lại trên toàn bộ dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
best_model = models[best_model_name]
best_model.fit(X_scaled, y)

# Lưu mô hình tốt nhất
joblib.dump({
    'model': best_model,
    'scaler': scaler,
    'feature_names': X.columns.tolist(), 
    'label_encoder_y': le_y
}, 'best_model.pkl')

import seaborn as sns
import matplotlib.pyplot as plt

# Giả sử bạn đã LabelEncode A16 như sau: '+' → 1 (Được duyệt), '-' → 0 (Không duyệt)
sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
sns.countplot(x='A16', data=df, palette='pastel')

# Đặt nhãn trục x cho dễ hiểu
plt.xticks([1, 0], ['❌ Không duyệt', '✅ Được duyệt'])
plt.title('Số lượng mẫu được duyệt và không được duyệt')
plt.xlabel('Kết quả phê duyệt')
plt.ylabel('Số lượng')
plt.tight_layout()
plt.show()

print("Đã lưu mô hình tốt nhất vào file best_model.pkl")

# Đánh giá bằng ma trận nhầm lẫn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Lặp qua từng mô hình
for name, model in models.items():
    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Huấn luyện và dự đoán
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_y.classes_)

    # Vẽ riêng từng ma trận
    plt.figure(figsize=(5, 4))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Ma trận nhầm lẫn - {name}')
    plt.grid(False)
    plt.tight_layout()
    plt.show()


# Đánh giá bằng các chỉ số Precision / Recall / F1-score:

from sklearn.metrics import classification_report
import seaborn as sns

def plot_classification_report(model, model_name):
    # Dự đoán trên toàn bộ tập dữ liệu
    y_pred = model.predict(X_scaled)
    
    # Tạo classification report dạng dict
    report_dict = classification_report(y, y_pred, target_names=['❌ Không duyệt', '✅ Được duyệt'], output_dict=True)
    
    # Bỏ hàng 'accuracy', 'macro avg', 'weighted avg' nếu không muốn
    df_report = pd.DataFrame(report_dict).transpose().round(2)
    
    # Vẽ bảng đẹp bằng seaborn
    plt.figure(figsize=(8, 3))
    sns.heatmap(df_report.iloc[:2, :3], annot=True, cmap='YlGnBu', cbar=False, fmt='.2f')
    plt.title(f'Classification Report - {model_name}')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Gọi hàm cho mô hình tốt nhất
plot_classification_report(best_model, best_model_name)

