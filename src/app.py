import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Paths to the model, label encoder, and evaluation metrics files
model_path = os.path.join(current_dir, 'rf_model.pkl')
encoder_path = os.path.join(current_dir, 'label_encoder.pkl')
metrics_path = os.path.join(current_dir, 'evaluation_metrics.pkl')

# Load the model, label encoder, and evaluation metrics
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

with open(metrics_path, 'rb') as f:
    metrics = pickle.load(f)

# Add custom CSS for styling
st.markdown("""
    <style>
    .message-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    .high-growth {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .low-growth {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .stButton>button {
        background-color: #00bfae;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and layout
st.title("GDP Growth Projector")
st.image(r"C:\\Users\\brady\\OneDrive\\Pictures\\Screenshots\\stock_image.png", width=600)  # Path to logo file
st.sidebar.header("Input Features")

# Sidebar for input features
col1, col2 = st.columns(2)  # Two columns for input layout

with col1:
    gdp = st.sidebar.number_input("GDP Value (In Billions)", value=0.0, step=0.1)
    unemployment = st.sidebar.number_input("Unemployment Rate", value=0.0, step=0.1)

with col2:
    inflation = st.sidebar.number_input("Inflation Rate", value=0.0, step=0.1)
    unemployment_change = st.sidebar.number_input("Unemployment Change (%)", value=0.0, step=0.1)
    inflation_rate = st.sidebar.number_input("Inflation Rate Change (%)", value=0.0, step=0.1)

# DataFrame for model input
input_data = pd.DataFrame({
    'GDP': [gdp],
    'Unemployment': [unemployment],
    'Inflation': [inflation],
    'Unemployment_Change': [unemployment_change],
    'Inflation_Rate': [inflation_rate]
})

# Project button
if st.button("Project GDP Growth"):
    # Make projections
    prediction = model.predict(input_data)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    # Display result with custom messages
    if predicted_label == "High":
        st.markdown("""
        <div class="message-box high-growth">
            The GDP Growth Rate is projected to be above 2%. Now may be a good time to invest!
        </div>
        """, unsafe_allow_html=True)
    elif predicted_label == "Low":
        st.markdown("""
        <div class="message-box low-growth">
            The GDP Growth Rate is projected to be below 2%. It may be wise to pull back on investments or wait to enter new markets.
        </div>
        """, unsafe_allow_html=True)

# Expander for model details
with st.expander("View Model Details"):
    st.write("""
This model infers the growth trajectory of the U.S. GDP based on economic indicators such as GDP, unemployment, and inflation. 
The model projects whether the GDP growth rate will be either 2% or more (high) or less than 2% (low). Each output is accompanied by relevant investment advice. Note that the model's GDP growth rate projections are not definite. The projections are made using a Random Forest model trained on historical data from Federal Reserve Economic Data.
    """)

# Expander for evaluation metrics
with st.expander("View Model Evaluation Metrics"):
    st.write("### Model Performance Metrics")
    st.write(f"- **Accuracy:** {metrics['Accuracy'] * 100:.2f}%")
    st.write(f"- **Precision:** {metrics['Precision']:.2f}")
    st.write(f"- **Recall:** {metrics['Recall']:.2f}")
    st.write(f"- **F1 Score:** {metrics['F1 Score']:.2f}")

# Expander for visualizations
with st.expander("View Model Visualizations"):
    # Load and display confusion matrix heatmap
    cm_path = os.path.join(current_dir, 'confusion_matrix.png')
    pr_path = os.path.join(current_dir, 'precision_recall_curve.png')

    if os.path.exists(cm_path):
        st.write("### Confusion Matrix")
        st.image(cm_path)

    if os.path.exists(pr_path):
        st.write("### Precision-Recall Curve")
        st.image(pr_path)