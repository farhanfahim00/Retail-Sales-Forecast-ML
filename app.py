import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("Retail Sales Demand Forecasting Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Visualizations", "Prediction of Demand", "Model Performance"])

# Loading data
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/processed.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Loading model
@st.cache_resource
def load_model():
    with open('models/demand_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names

# Loading everything
df = load_data()
model, feature_names = load_model()


# Overview Page

if page == "Overview":
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Products", df['Product'].nunique())
    with col3:
        st.metric("Countries", df['Country'].nunique())
    with col4:
        st.metric("Date Range", f"{df['Date'].min().year} - {df['Date'].max().year}")
    
    st.markdown("---")
    
    # Showing sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    
    # Basic statistics
    st.subheader("Summary Statistics")
    st.dataframe(df[['Units Sold', 'Sale Price', 'Discounts', 'Sales', 'Profit']].describe())

# Visualizations Page
elif page == "Visualizations":
    st.header("Data Visualizations")

    # Sales Distribution

    st.subheader("Sales Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df['Sales'], bins=30, edgecolor='black', color='steelblue', alpha=0.7)
    ax.set_xlabel('Sales Amount ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Sales', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Sales Trend Over Time

    st.subheader("Sales Trend Over Time")
    daily_sales = df.groupby('Date')['Sales'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_sales['Date'], daily_sales['Sales'], color='green', linewidth=2)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Total Sales ($)', fontsize=12)
    ax.set_title('Sales Over Time', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Top Products by Sales

    st.subheader("Top Products by Total Sales")
    product_sales = df.groupby('Product')['Sales'].sum().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    product_sales.plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel('Total Sales ($)', fontsize=12)
    ax.set_ylabel('Product', fontsize=12)
    ax.set_title('Product Performance', fontsize=14, fontweight='bold')
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Demand by Discount Band

    st.subheader("Average Demand by Discount Band")
    discount_demand = df.groupby('Discount Band')['Units Sold'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    discount_demand.plot(kind='bar', ax=ax, color='orange')
    ax.set_xlabel('Discount Band', fontsize=12)
    ax.set_ylabel('Average Units Sold', fontsize=12)
    ax.set_title('Impact of Discounts on Demand', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Prediction of Demand Page

elif page == "Prediction of Demand":
    st.header("Predict Product Demand")
    st.write("Enter product details to predict demand (units sold)")
    
    # Creating input form

    col1, col2 = st.columns(2)
    
    with col1:
        product = st.selectbox("Product", df['Product'].unique())
        segment = st.selectbox("Customer Segment", df['Segment'].unique())
        country = st.selectbox("Country", df['Country'].unique())
        discount_band = st.selectbox("Discount Band", df['Discount Band'].unique())
    
    with col2:
        sale_price = st.number_input("Sale Price ($)", min_value=0.0, value=20.0, step=1.0)
        manufacturing_price = st.number_input("Manufacturing Price ($)", min_value=0.0, value=5.0, step=1.0)
        discounts = st.number_input("Discount Amount ($)", min_value=0.0, value=0.0, step=10.0)
        month = st.slider("Month", min_value=1, max_value=12, value=6)
    
    # Prediction button
    
    if st.button("Predict Demand", type="primary"):

        # Preparing input data
        input_data = pd.DataFrame({
            'Sale Price': [sale_price],
            'Manufacturing Price': [manufacturing_price],
            'Discounts': [discounts],
            'Month': [month]
        })
        
        # One-hot encode categorical features
        categorical_input = pd.DataFrame({
            'Product': [product],
            'Segment': [segment],
            'Country': [country],
            'Discount Band': [discount_band]
        })
        
        encoded_input = pd.get_dummies(categorical_input, drop_first=True)
        
        # Aligning with training features
        for col in feature_names:
            if col not in input_data.columns and col not in encoded_input.columns:
                encoded_input[col] = 0
        
        # Combining numeric and categorical
        final_input = pd.concat([input_data, encoded_input], axis=1)
        
        # Ensuring correct order
        final_input = final_input[feature_names]
        
        # Making prediction
        prediction = model.predict(final_input)[0]
        
        # Displaying result
        st.success(f"### Predicted Demand: **{prediction:.0f} units**")
        
        # Additional insights
        col1, col2, col3 = st.columns(3)
        with col1:
            estimated_sales = prediction * sale_price
            st.metric("Estimated Sales", f"${estimated_sales:,.0f}")
        with col2:
            estimated_revenue = estimated_sales - discounts
            st.metric("Net Revenue", f"${estimated_revenue:,.0f}")
        with col3:
            estimated_profit = estimated_revenue - (prediction * manufacturing_price)
            st.metric("Estimated Profit", f"${estimated_profit:,.0f}")

# Model Performance Page

elif page == "Model Performance":
    st.header("Model Performance Metrics")
    
    # Loading test predictions
    st.info("Model trained on 80% of data, tested on 20% unseen data")
    
    # Displaying metrics from training
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", "0.185", help="Model explains 18.5% of variance")
    with col2:
        st.metric("MAE", "610 units", help="Average prediction error")
    with col3:
        st.metric("RMSE", "764 units", help="Root Mean Squared Error")
    
    st.markdown("---")
    
    # Model interpretation
    st.subheader("What Does This Mean?")
    
    st.write("""
    **Model Performance Analysis:**
    
    - **R² Score (0.185):** The model explains 18.5% of the variance in demand. While this might seem low, 
      it's realistic for demand prediction where many external factors (marketing, competition, economy) 
      aren't captured in the data.
    
    - **MAE (610 units):** On average, predictions are off by about 610 units. For context, typical orders 
      range from 500-1,500 units.
    
    - **Key Insight:** Demand is inherently difficult to predict with limited features. This model provides 
      a baseline that's better than random guessing and can be improved with more data.
    """)
    
    st.markdown("---")
    
    # Top features
    st.subheader("Top 5 Most Important Features")
    
    feature_importance = pd.DataFrame({
        'Feature': ['Discount Band (Low)', 'Discount Band (None)', 'Segment (Midmarket)', 
                    'Country (Mexico)', 'Country (Germany)'],
        'Impact': [427, 410, 250, -225, -222]
    })
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['green' if x > 0 else 'red' for x in feature_importance['Impact']]
    ax.barh(feature_importance['Feature'], feature_importance['Impact'], color=colors, alpha=0.7)
    ax.set_xlabel('Coefficient Value', fontsize=12)
    ax.set_title('Feature Impact on Demand Prediction', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle'--', linewidth=0.8)
    st.pyplot(fig)
    
    st.write("""
    - **Positive coefficients (green):** Increase predicted demand
    - **Negative coefficients (red):** Decrease predicted demand
    """)