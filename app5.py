import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------- App Title & Description --------------------
st.title("Customer Segmentation Dashboard")
st.write(
    "This system uses K-Means Clustering to group customers based on their "
    "purchasing behavior and similarities."
)

# -------------------- Upload Dataset --------------------
file = st.file_uploader("Upload Customer Dataset (CSV)", type="csv")

if file:
    df = pd.read_csv(file)

    # Select only numerical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # -------------------- Sidebar Inputs --------------------
    st.sidebar.header("Clustering Controls")

    feature1 = st.sidebar.selectbox("Select Feature 1", numeric_cols)
    feature2 = st.sidebar.selectbox("Select Feature 2", numeric_cols)

    k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)

    random_state = st.sidebar.number_input(
        "Random State (optional)", value=42, step=1
    )

    run = st.sidebar.button("Run Clustering")

    # -------------------- Run Clustering --------------------
    if run and feature1 != feature2:

        # Prepare data
        X = df[[feature1, feature2]]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # KMeans
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        # -------------------- Visualization --------------------
        st.subheader("📊 Cluster Visualization")

        plt.figure()
        plt.scatter(
            df[feature1],
            df[feature2],
            c=df['Cluster']
        )

        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        plt.scatter(
            centers[:, 0],
            centers[:, 1],
            s=250,
            marker='X'
        )

        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title("Customer Clusters")
        st.pyplot(plt)

        # -------------------- Cluster Summary --------------------
        st.subheader("📋 Cluster Summary")

        summary = (
            df.groupby('Cluster')
              .agg(
                  Count=('Cluster', 'size'),
                  Avg_Feature1=(feature1, 'mean'),
                  Avg_Feature2=(feature2, 'mean')
              )
        )

        st.dataframe(summary)

        # -------------------- Business Interpretation --------------------
        st.subheader("💡 Business Interpretation")

        for cluster in summary.index:
            st.write(
                f"🔹 Cluster {cluster}: Customers in this group show similar "
                f"purchasing behaviour for {feature1} and {feature2}."
            )

        # -------------------- User Guidance --------------------
        st.info(
            "Customers in the same cluster exhibit similar purchasing behaviour "
            "and can be targeted with similar business strategies."
        )

    elif run:
        st.warning("Please select two different numerical features.")
