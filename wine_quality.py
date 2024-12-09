import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')
    return data

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", [
    "1. Overview",
    "2. Data Exploration and Preparation",
    "3. Analysis and Insights",
    "4. Conclusions and Recommendations"
])

# Load data
data = load_data()

if section == "1. Overview":
    # App title and description
    st.title("Wine Quality Analysis")
    st.markdown("""
    This Streamlit app explores the Wine Quality dataset, performs data analysis using clustering and regression techniques, 
    and provides interactive visualizations for insights.
    """)

    # Display dataset structure
    st.subheader("Dataset Overview")
    st.dataframe(data.head())
    st.markdown("**Dataset Structure:**")
    st.write(data.info())
    st.markdown("**Descriptive Statistics:**")
    st.write(data.describe())

if section == "2. Data Exploration and Preparation":
    # Data Cleaning and Preparation
    st.title("2. Data Exploration and Preparation")
    st.subheader("Data Cleaning")
    if st.checkbox("Show missing values count"):
        st.write(data.isnull().sum())

    st.subheader("Correlation Heatmap")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

if section == "3. Analysis and Insights":
    # Clustering Analysis
    st.title("3. Analysis and Insights")
    st.subheader("Clustering Analysis")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.drop("quality", axis=1))

    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data_scaled)
    st.write("Cluster Centers:")
    st.write(kmeans.cluster_centers_)

    # Cluster Visualization
    st.subheader("Cluster Visualization")
    x_col = st.selectbox("Select X-axis feature", data.columns[:-1])
    y_col = st.selectbox("Select Y-axis feature", data.columns[:-1])

    fig, ax = plt.subplots()
    sns.scatterplot(x=data[x_col], y=data[y_col], hue=data['Cluster'], palette='viridis', ax=ax)
    st.pyplot(fig)

if section == "4. Conclusions and Recommendations":
    # Conclusions and Recommendations
    st.title("4. Conclusions and Recommendations")
    st.markdown("""
    - **Key Insight 1:** Highlight relationships between features and wine quality.
    - **Key Insight 2:** Provide actionable insights for winemakers.

    Explore the app's interactive visualizations to uncover more insights.
    """)
