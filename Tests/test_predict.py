import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Assuming your dataset is in a CSV file named 'AQIMyanmar.csv'
# Replace 'YourDataset.csv' with the actual file name if it's different
file_path = 'AQIMyanmar.csv'

# Read the dataset into a DataFrame
df = pd.read_csv(file_path)

# Extract the relevant column 'AQI (US)'
aqi_column = df['AQI (US)']

# Filter the data within the specified range (1-486)
filtered_aqi = aqi_column[(aqi_column >= 1) & (aqi_column <= 486)]

# Prepare the data for training
X = pd.DataFrame({'Data Points': range(1, len(filtered_aqi) + 1)})
y = filtered_aqi.values.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Plotting the original AQI data
plt.plot(filtered_aqi, label='Original AQI (US)', color='blue')

# Plotting the predicted AQI data
plt.scatter(X_test, y_test, color='red', label='Actual AQI (US)')
plt.plot(X_test, y_pred, color='black', linewidth=3, label='Predicted AQI (US)')

# Customize the plot
plt.title('AQI (US) and Predicted AQI (US) in AQIMyanmar')
plt.xlabel('Data Points')
plt.ylabel('AQI (US)')
plt.legend()
plt.show()
