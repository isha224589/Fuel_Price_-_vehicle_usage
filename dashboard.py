import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Set wide layout
st.set_page_config(layout="wide")
st.title("🚗 Fuel Prices vs Vehicle Usage Dashboard")

# Load Data
fuel_df = pd.read_csv("fuel_prices_indian_basket.csv")
vehicle_df = pd.read_csv("VAHAN Vehicle Registrations by Fuel Type_Sample_Data.csv")

# Preprocessing
fuel_df['Month'] = pd.to_datetime(fuel_df['Month'], format='%b-%y')
fuel_df['Year'] = fuel_df['Month'].dt.year

vehicle_df['Date (date)'] = pd.to_datetime(vehicle_df['Date (date)'])
vehicle_df['Year'] = vehicle_df['Date (date)'].dt.year

# Aggregate vehicle registrations per year
vehicle_yearly = vehicle_df.groupby('Year')['Registrations (registrations)'].sum().reset_index()
vehicle_yearly.rename(columns={'Registrations (registrations)': 'Total Registrations'}, inplace=True)

# Merge both datasets on Year
merged = pd.merge(fuel_df, vehicle_yearly, on='Year', how='inner')

# Sidebar Filter
year_range = st.sidebar.slider("Select Year Range", int(merged['Year'].min()), int(merged['Year'].max()), (2018, 2022))
filtered = merged[(merged['Year'] >= year_range[0]) & (merged['Year'] <= year_range[1])]

# Fuel Price Trend Plot
st.subheader("📈 Fuel Price Trends")
fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=filtered, x='Year', y='Petrol (Rs./Ltr)', label='Petrol', ax=ax1)
sns.lineplot(data=filtered, x='Year', y='Diesel (Rs./Ltr)', label='Diesel', ax=ax1)
plt.ylabel("₹ per Litre")
plt.title("Petrol & Diesel Price Trend")
st.pyplot(fig1)

# Vehicle Registration Plot
st.subheader("🚙 Vehicle Registrations by Year")
fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.barplot(data=filtered, x='Year', y='Total Registrations', palette='viridis', ax=ax2)
plt.title("Year-wise Vehicle Registrations")
st.pyplot(fig2)

# Forecast using Linear Regression
X = sm.add_constant(merged['Year'])
model = sm.OLS(merged['Total Registrations'], X).fit()
future_years = pd.DataFrame({'Year': [2023, 2024, 2025]})
future_X = sm.add_constant(future_years)
predictions = model.predict(future_X)

# Create forecast DataFrame
forecast_df = pd.concat([
    merged.set_index('Year')['Total Registrations'],
    pd.Series(predictions.values, index=[2023, 2024, 2025], name='Total Registrations')
])

# Forecast Plot
st.subheader("📊 Forecasted Vehicle Registrations (2023–2025)")
fig3, ax3 = plt.subplots(figsize=(10, 4))
sns.lineplot(x=forecast_df.index, y=forecast_df.values, marker='o', ax=ax3)
plt.axvline(x=2022, color='gray', linestyle='--', label='Forecast Start')
plt.ylabel("Registrations")
plt.title("Forecast of Vehicle Registrations")
plt.legend()
st.pyplot(fig3)
