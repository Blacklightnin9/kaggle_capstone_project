import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import plotly.express as px
import streamlit as st

# Load and clean datasets
@st.cache_data
def load_data():
    data_en = pd.read_csv("cleaned_Konsumsi_Rambutan_Perkapita_2024_en.csv", delimiter=",")
    data_id = pd.read_csv("cleaned_Konsumsi_Rambutan_Perkapita_2024_id.csv", delimiter=",")

    # Fix combined columns
    if len(data_en.columns) == 1:
        data_en[['Region', 'Consumption']] = data_en.iloc[:, 0].str.split(',', expand=True)
    if len(data_id.columns) == 1:
        data_id[['Region', 'Consumption']] = data_id.iloc[:, 0].str.split(',', expand=True)

    # Convert "Consumption" to numeric values
    data_en["Consumption"] = pd.to_numeric(data_en["Consumption"], errors='coerce').fillna(0)
    data_id["Consumption"] = pd.to_numeric(data_id["Consumption"], errors='coerce').fillna(0)

    # Concatenate datasets
    data = pd.concat([data_en, data_id], ignore_index=True)

    # Clean dataset
    data = data.drop_duplicates(subset=["Region", "Consumption"])
    data = data.dropna(subset=["Region", "Consumption"])
    data["Region"] = data["Region"].str.strip()

    return data

data = load_data()

# Generate embeddings
@st.cache_resource
def generate_embeddings(data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    batch_size = 32
    embeddings = []

    for i in tqdm(range(0, len(data), batch_size), desc="Generating Embeddings"):
        batch_data = data["Region"].iloc[i:i+batch_size].tolist()
        batch_embeddings = model.encode(batch_data)
        embeddings.extend(batch_embeddings)

    embeddings_np = np.array(embeddings)
    return embeddings_np, model

embeddings_np, model = generate_embeddings(data)

# Build FAISS index
@st.cache_resource
def create_faiss_index(embeddings_np):
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    return index

index = create_faiss_index(embeddings_np)

# Function to query FAISS index
def search_region(query, top_k=5):
    query_embedding = model.encode([query.strip()])[0]
    distances, indices = index.search(np.array([query_embedding]), top_k)
    results = data.iloc[indices[0]]
    results = results[["Region", "Consumption"]]
    return results

# Streamlit App Layout
st.title("Rambutan Consumption Analysis")
st.write("Explore rambutan consumption data across regions. Use the search box below to find specific regions.")

# Input for search query and top-k results
region_query = st.text_input("Enter Region Name:")
top_k = st.slider("Number of Results", 1, 5, 3)

# Search and display results
if region_query:
    results = search_region(region_query, top_k)
    if not results.empty:
        st.write(results)
        
        # Visualize results
        fig_query = px.bar(
            results,
            x="Region",
            y="Consumption",
            title=f"Top {top_k} Results for Query: {region_query}",
            labels={"Region": "Region", "Consumption": "Consumption (tons)"},
            color="Consumption"
        )
        st.plotly_chart(fig_query)
    else:
        st.write("Region not found!")

# Display full dataset chart
st.header("Full Dataset Visualization")
fig = px.bar(
    data,
    x="Region",
    y="Consumption",
    title="Rambutan Consumption Across Regions",
    labels={"Region": "Region", "Consumption": "Consumption (tons)"},
    color="Consumption",
    height=600,
    width=1500
)
fig.update_layout(
    xaxis=dict(
        tickangle=-45,
        automargin=True,
        title=dict(standoff=10),
        showgrid=False,
        type="category",
        rangeslider=dict(visible=True)
    )
)
st.plotly_chart(fig)