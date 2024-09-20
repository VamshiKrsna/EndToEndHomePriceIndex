import streamlit as st
import pickle
import pandas as pd

with open("saved_models/RandomForestRegressor.pkl", "rb") as file:
    model = pickle.load(file)

def main():
    st.title("Home Price Index Prediction")
    st.sidebar.header("Input Parameters")

    cpi = st.sidebar.number_input("Consumer Price Index (CPI)", min_value=0.0, value=260.0)
    federal_funds_rate = st.sidebar.number_input("Federal Funds Rate", min_value=0.0, max_value=10.0, value=0.05)
    gdp = st.sidebar.number_input("Gross Domestic Product (GDP) in Billion USD", min_value=0.0, value=21000.0)
    housing_starts = st.sidebar.number_input("Housing Starts (in Thousands)", min_value=0.0, value=1200.0)
    crude_oil_prices = st.sidebar.number_input("Crude Oil Prices (in USD per Barrel)", min_value=0.0, value=70.0)
    median_household_income = st.sidebar.number_input("Median Household Income (in USD)", min_value=0.0, value=68000.0)
    population = st.sidebar.number_input("Population (in Thousands)", min_value=0.0, value=330000.0)
    unemployment_rate = st.sidebar.number_input("Unemployment Rate", min_value=0.0, max_value=20.0, value=0.05)
    building_permits = st.sidebar.number_input("Building Permits (in Thousands)", min_value=0.0, value=1300.0)

    input_data = pd.DataFrame({
        'CPI': [cpi],
        'Federal Funds Rate': [federal_funds_rate],
        'GDP': [gdp],
        'Housing Starts': [housing_starts],
        'Crude Oil Prices': [crude_oil_prices],
        'Median Household Income': [median_household_income],
        'Population': [population],
        'Unemployment Rate': [unemployment_rate],
        'Building Permits': [building_permits]
    })

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.write(f"For the given input values, the Home Price Index (HPI) is **{prediction:.5f}**, indicating a "
             f"**{prediction - 100:.2f}%** increase since the base period (January 2000, HPI = 100). This means home "
             f"prices are **{prediction / 100:.3f}** times higher than the base value.")
    
    base_price = 100000  # Example base price in USD
    current_price = base_price * (prediction / 100)
    st.write(f"Example: If a house cost **100,000 USD** in January 2000, it would cost **{current_price:.2f}** USD for these given input values.")

if __name__ == "__main__":
    main()