import pandas as pd
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

# Plotting the data
plt.plot(filtered_aqi, label='AQI (US)')
plt.title('AQI (US) in AQIMyanmar')
plt.xlabel('Data Points')
plt.ylabel('AQI (US)')
plt.legend()
plt.show()
