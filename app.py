import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage

from model.utils import load_data
from model.train import train_model

st.set_page_config(
    page_title="Hierarchical Clustering",
    layout="wide"
)

st.title("🌳 Customer Segmentation using Hierarchical Clustering")

# Sidebar
n_clusters = st.sidebar.slider(
    "Number of Clusters",
    2,
    10,
    5
)

# Load Data
df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Features
X = df[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]

# Train Model
model, clusters = train_model(
    X,
    n_clusters
)

df["Cluster"] = clusters

# Cluster Distribution
st.subheader("Cluster Distribution")

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

st.bar_chart(cluster_counts)

# Scatter Plot
st.subheader("Customer Segments")

fig, ax = plt.subplots(figsize=(8, 6))

for cluster in sorted(df["Cluster"].unique()):

    cluster_data = df[
        df["Cluster"] == cluster
    ]

    ax.scatter(
        cluster_data["Annual Income (k$)"],
        cluster_data["Spending Score (1-100)"],
        label=f"Cluster {cluster}"
    )

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("Hierarchical Clustering")

ax.legend()

st.pyplot(fig)

# Dendrogram
st.subheader("Dendrogram")

linked = linkage(
    X,
    method="ward"
)

fig2, ax2 = plt.subplots(
    figsize=(12, 6)
)

dendrogram(
    linked,
    ax=ax2
)

ax2.set_title("Dendrogram")
ax2.set_xlabel("Customers")
ax2.set_ylabel("Distance")

st.pyplot(fig2)

# Cluster Summary
st.subheader("Cluster Summary")

summary = df.groupby(
    "Cluster"
)[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
].mean()

st.dataframe(summary)

# Business Insights
st.subheader("Business Insights")

for cluster in sorted(df["Cluster"].unique()):

    temp = df[
        df["Cluster"] == cluster
    ]

    avg_income = temp[
        "Annual Income (k$)"
    ].mean()

    avg_spending = temp[
        "Spending Score (1-100)"
    ].mean()

    st.write(
        f"Cluster {cluster}: "
        f"Average Income = {avg_income:.2f}, "
        f"Average Spending = {avg_spending:.2f}"
    )