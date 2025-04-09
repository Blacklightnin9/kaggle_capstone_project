import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import plotly.express as px
import streamlit as st

# Step 1: Load and clean both datasets
@st.cache_data
def load_and_clean_data():
    data_en = pd.read_csv("./dataset/cleaned_Produksi_Rambutan_Prov_Sul_Sel_2019_2020_en.csv", delimiter=",")
    data_id = pd.read_csv("./dataset/cleaned_Produksi_Rambutan_Prov_Sul_Sel_2019_2020_id.csv", delimiter=",")

    # Fix combined columns (split if necessary)
    if len(data_en.columns) == 1:
        data_en[['Regency/City', 'Production 2019', 'Production 2020']] = data_en.iloc[:, 0].str.split(',', expand=True)
    if len(data_id.columns) == 1:
        data_id[['Kabupaten/Kota', 'Produksi 2019', 'Produksi 2020']] = data_id.iloc[:, 0].str.split(',', expand=True)

    # Convert "Production" columns to numeric values
    data_en["Production 2019"] = pd.to_numeric(data_en["Production 2019"], errors='coerce').fillna(0)
    data_en["Production 2020"] = pd.to_numeric(data_en["Production 2020"], errors='coerce').fillna(0)
    data_id["Produksi 2019"] = pd.to_numeric(data_id["Produksi 2019"], errors='coerce').fillna(0)
    data_id["Produksi 2020"] = pd.to_numeric(data_id["Produksi 2020"], errors='coerce').fillna(0)

    # Concatenate both datasets into a unified DataFrame
    data = pd.concat([
        data_en.rename(columns={"Regency/City": "Region", "Production 2019": "Production_2019", "Production 2020": "Production_2020"}),
        data_id.rename(columns={"Kabupaten/Kota": "Region", "Produksi 2019": "Production_2019", "Produksi 2020": "Production_2020"})
    ], ignore_index=True)

    # Clean dataset: Remove duplicates and handle missing values
    data = data.drop_duplicates(subset=["Region", "Production_2019", "Production_2020"])
    data = data.dropna(subset=["Region"])
    data["Region"] = data["Region"].str.strip()  # Strip whitespace from region names

    return data

data = load_and_clean_data()

# Step 2: Generate embeddings
@st.cache_resource
def generate_embeddings(data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(data["Region"].tolist())
    return embeddings, model

embeddings_np, model = generate_embeddings(data)

# Step 3: Build FAISS index
@st.cache_resource
def build_faiss_index(embeddings_np):
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Use L2 distance for similarity
    index.add(embeddings_np)  # Add embeddings to FAISS index
    return index

index = build_faiss_index(embeddings_np)

# Function to query FAISS index
def search_region(query, top_k=5):
    query_embedding = model.encode([query.strip()])[0]  # Encode query and strip whitespace
    distances, indices = index.search(np.array([query_embedding]), top_k)  # Search FAISS index
    results = data.iloc[indices[0]]  # Retrieve matching rows
    results = results[["Region", "Production_2019", "Production_2020"]]  # Keep only relevant columns
    return results

# Streamlit App
st.title("Rambutan Production Query")

# User Input
query = st.text_input("Enter Region Name:")
top_k = st.slider("Number of Results to Display:", 1, 5, 3)

if query:
    results = search_region(query, top_k)
    if not results.empty:
        st.subheader("Search Results")
        st.dataframe(results)

        # Generate bar chart
        fig_query = px.bar(
            results,
            x="Region",
            y="Production_2020",
            title=f"Top {top_k} Results for Query: {query}",
            labels={"Region": "Region", "Production_2020": "Production (tons)"},
            color="Production_2020",
            height=400,
            width=1000
        )
        st.plotly_chart(fig_query)
    else:
        st.write("Region not found!")

# Overall Bar Chart
fig = px.bar(
    data,
    x="Region",
    y="Production_2020",
    title="Rambutan Production Across Regions (English & Bahasa Indonesia)",
    labels={"Region": "Region", "Production_2020": "Production (tons)"},
    color="Production_2020",
    height=600,
    width=1000
)
#rotate x-axis labels for better readability
fig.update_layout(
    xaxis=dict(
        tickangle=35,  # Turns the text vertically
        automargin=True
        tickmode="array",
        tickvals=[i for i in range(len(results["Region"]))],  # Space tick values
        ticktext=results["Region"]  # Ensure labels are spaced properly
    )
)

st.plotly_chart(fig)