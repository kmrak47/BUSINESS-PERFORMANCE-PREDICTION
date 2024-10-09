import streamlit as st
import numpy as np
import pandas as pd
from prediction_helper import predict_business_performance
import joblib

# Load the trained model
try:
    model = joblib.load('model.pkl')
    if isinstance(model, dict):
        st.error("Loaded object is a dictionary, not a model. Please check the saved model.")
    else:
        st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Set up the Streamlit app title and layout
st.title("BUSINESS PERFORMANCE PREDICTION")

# Column layout for input fields
with st.form(key='prediction_form'):
    col1, col2 = st.columns(2)

    with col1:
        employee_count = st.number_input('Employee Count', min_value=1)
        annual_revenue = st.number_input('Annual Revenue (in millions)', min_value=0.0)
        profit_margin = st.slider('Profit Margin (%)', min_value=0.0, max_value=100.0, step=0.1)
        customer_satisfaction_score = st.slider('Customer Satisfaction Score (out of 10)', min_value=1.0, max_value=10.0)

    with col2:
        market_share = st.slider('Market Share (%)', min_value=0.0, max_value=100.0, step=0.1)
        debt_to_equity_ratio = st.number_input('Debt to Equity Ratio', min_value=0.0)
        ceo_age = st.number_input('CEO Age', min_value=25, max_value=100)
        employee_turnover_rate = st.slider('Employee Turnover Rate (%)', min_value=0.0, max_value=100.0, step=0.1)

    col3, col4 = st.columns(2)

    with col3:
        tech_adoption_score = st.slider('Tech Adoption Score (out of 10)', min_value=1.0, max_value=10.0)
        average_product_rating = st.slider('Average Product Rating (out of 5)', min_value=1.0, max_value=5.0)
        number_of_products = st.number_input('Number of Products', min_value=1)
        number_of_locations = st.number_input('Number of Locations', min_value=1)

    with col4:
        social_media_followers = st.number_input('Social Media Followers', min_value=0)
        website_traffic = st.number_input('Website Traffic (monthly)', min_value=0)
        employee_satisfaction_score = st.slider('Employee Satisfaction Score (out of 10)', min_value=1.0, max_value=10.0)

    # Button for Prediction
    submit_button = st.form_submit_button(label='Predict Business Performance')

    if submit_button:
        if isinstance(model, dict):
            st.error("Model is not loaded correctly.")
        else:
            # Gather inputs
            input_data = np.array([[employee_count, annual_revenue, profit_margin, customer_satisfaction_score,
                                    market_share, debt_to_equity_ratio, ceo_age, employee_turnover_rate,
                                    tech_adoption_score, average_product_rating, number_of_products, number_of_locations,
                                    social_media_followers, website_traffic, employee_satisfaction_score]])

            # Prediction
            result = predict_business_performance(model, input_data)

            # Display the prediction result
            st.markdown(f"<h3 style='text-align: center; color: green;'>Predicted Business Performance: {result}</h3>", unsafe_allow_html=True)
