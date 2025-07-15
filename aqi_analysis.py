import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'AQI and Lat Long of Countries complete.csv'
aqi_data = pd.read_csv(file_path)

# Basic Information
print("\nDataset Info:\n")
print(aqi_data.info())

print("\nMissing Values:\n")
print(aqi_data.isnull().sum())

print("\nSummary Statistics:\n")
print(aqi_data.describe())

# 1. AQI Distribution Plot
plt.figure(figsize=(10, 6))
sns.histplot(aqi_data['AQI Value'], kde=True, bins=30)
plt.title('Distribution of AQI Values')
plt.xlabel('AQI Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 2. Top 10 Countries by AQI
top_countries = aqi_data.groupby('Country')['AQI Value'].mean().sort_values(ascending=False).head(10).reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(data=top_countries, x='Country', y='AQI Value', hue='Country', palette='Reds_r', legend=False)
plt.title('Top 10 Countries by Average AQI')
plt.xlabel('Country')
plt.ylabel('Average AQI')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Categorize AQI
def categorize_aqi(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

aqi_data['AQI_Category_Calc'] = aqi_data['AQI Value'].apply(categorize_aqi)

# 4. AQI Category Count Plot
plt.figure(figsize=(10, 6))
sns.countplot(data=aqi_data, x='AQI_Category_Calc',
              order=['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous'],
              hue='AQI_Category_Calc', palette='viridis', legend=False)
plt.title('Distribution of AQI Categories')
plt.xlabel('AQI Category')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 5. Geographical Distribution
plt.figure(figsize=(10, 6))
sns.scatterplot(data=aqi_data, x='lng', y='lat', hue='AQI Value', size='AQI Value',
                palette='coolwarm', sizes=(10, 200), alpha=0.7)
plt.title('Geographical Distribution of AQI')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='AQI Value', loc='upper left')
plt.tight_layout()
plt.show()

# 6. Random Forest Regression
features = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value', 'SO2', 'lat', 'lng']
target = 'AQI Value'

X = aqi_data[features]
y = aqi_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nRandom Forest Regression Results:")
print(f"   RMSE: {rmse:.2f}")
print(f"   R² Score: {r2:.4f}")

# 7. Country AQI Report Function
def country_aqi_report(country_name):
    country_data = aqi_data[aqi_data['Country'].str.lower() == country_name.lower()]
    if country_data.empty:
        print(f"\nCountry '{country_name}' not found in the dataset.")
        return

    avg_aqi = country_data['AQI Value'].mean()
    country_avg_aqi = aqi_data.groupby('Country')['AQI Value'].mean().sort_values(ascending=False).reset_index()
    rank = country_avg_aqi[country_avg_aqi['Country'].str.lower() == country_name.lower()].index.item() + 1
    aqi_category = categorize_aqi(avg_aqi)

    pollutants = ['PM2.5 AQI Value', 'NO2 AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'SO2']
    avg_pollutants = country_data[pollutants].mean()

    print(f"\nAQI Report for {country_name.title()}")
    print(f"----------------------------------------")
    print(f"Average AQI: {avg_aqi:.2f}")
    print(f"AQI Category: {aqi_category}")
    print(f"AQI Rank: #{rank} (out of {len(country_avg_aqi)})\n")

    print("Average Pollutant Levels:")
    for pollutant, value in avg_pollutants.items():
        print(f"   • {pollutant}: {value:.2f}")

# Input a Country Name for Details
user_input = input("\nEnter a country name to get its AQI report: ").strip()
country_aqi_report(user_input)
