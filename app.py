import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sales Prediction AI", page_icon="📈")
st.title("📈 Sales Prediction AI")
st.write("put Advertisement and sales prediction AI:")


model = joblib.load("sales_model.pkl")


tv = st.number_input("TV  (TV Ads):", min_value=0.0, value=200.0)
radio = st.number_input("Radio (Radio Ads):", min_value=0.0, value=40.0)
newspaper = st.number_input("Newspaper (Newspaper):", min_value=0.0, value=20.0)


if st.button("Predict"):
    input_data = pd.DataFrame(
        [[tv, radio, newspaper]], columns=["TV", "Radio", "Newspaper"]
    )
    prediction = model.predict(input_data)[0]
    st.success(f"predict sales: **{prediction:.2f}** (1000 units)")