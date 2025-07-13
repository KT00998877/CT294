import numpy as np
import pandas as pd

df = pd.read_csv('crx.csv')
print("Kiem tra du lieu bị thieu:")
# Thay '?' bằng NaN
df.replace('?', np.nan, inplace=True)

# Kiểm tra lại số lượng thiếu
print(df.isnull().sum())
