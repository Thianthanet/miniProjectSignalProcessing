import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# อ่านข้อมูลจากไฟล์ CSV
file_path = 'AQIMyanmar.csv'
df = pd.read_csv(file_path)

# เพิ่มคอลัมน์ลำดับ
df['ลำดับ'] = range(1, len(df) + 1)

# ตรวจสอบข้อมูลใน DataFrame
# print(df.head())

# เลือกเฉพาะคอลัมน์ที่ต้องการใช้
X = df[['AQI (CN)', 'Temperature', 'Pressure']]
y = df['AQI (US)']

# แบ่งข้อมูลเป็นชุดการฝึกอบรมและทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดลเชิงเส้น
model = LinearRegression()

# ฝึกโมเดล
model.fit(X_train, y_train)

# ทดสอบโมเดล
y_pred = model.predict(X_test)

print(df)
print(X_test)
print(y_train)

# แสดงกราฟเส้นการคาดการณ์ vs จริง
# plt.plot(y_test.index, y_test, label='Actual AQI (US)', marker='o')
# plt.plot(X_test.index, y_pred, label='Predicted AQI (US)', marker='x')
# plt.xlabel('ลำดับ')
# plt.ylabel('AQI (US)')
# plt.title('Predicted AQI')
# plt.legend()
# plt.show()
