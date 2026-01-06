import streamlit as st
import numpy as np
import joblib
from sklearn.cluster import AgglomerativeClustering

st.set_page_config(page_title="Bank Customer Clustering", page_icon="🏦")
st.title("🏦 Bank Customer Segmentation (Agglomerative)")

# Load saved objects
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.subheader("Enter Customer Details")

user_input = []

for feature in features:
    value = st.number_input(feature)
    user_input.append(value)

if st.button("Find Cluster"):
    user_data = np.array(user_input).reshape(1, -1)

    # Scale input
    user_scaled = scaler.transform(user_data)

    # Load original scaled data
    X_scaled = scaler.transform(
        np.zeros((1, len(features)))
    )

    # Append new point
    X_new = np.vstack([X_scaled, user_scaled])

    model = AgglomerativeClustering(
        n_clusters=3,
        metric='euclidean',
        linkage='ward'
    )

    labels = model.fit_predict(X_new)
    user_cluster = labels[-1]

    st.success(f"Customer belongs to Cluster: {user_cluster}")
