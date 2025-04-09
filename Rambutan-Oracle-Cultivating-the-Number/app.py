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

# Streamlit App with Title
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='font-size:50px;'>Rambutan Oracle: Cultivating the Numbers</h1>
        <p style='font-size:25px; color:#6A5ACD;'>
            Discover Insights, Unlock Patterns, and Harvest Data on Rambutan Production
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar for Interaction
# Sidebar title
st.sidebar.markdown(
    """
    <h2 style='text-align: center;'>Where Insights Bloom Like Rambutan!</h2>
    """,
    unsafe_allow_html=True
)

st.sidebar.image("./ab_logo.png", use_column_width=False, width=125)  # Add logo to sidebar
st.sidebar.markdown(
    """
    <h3 style='text-align: center; margin-top: -20px;'>Andi Bima</h3>
    """,
    unsafe_allow_html=True
)  # Add centered text below logo

st.sidebar.markdown(
    """
    <p style='font-size:13px; text-align: center; color: #E6E6FA;'>
        Use the input box above to search for a specific region by name. You can enter the name of a regency/city and get production insights.
        Adjust the slider below to control how many results you want to display. For example, slide to 3 to view the top 3 most relevant matches.
    </p>
    """,
    unsafe_allow_html=True
)  # Add an explanatory paragraph

st.sidebar.header("Search Options")
query = st.sidebar.text_input("Enter Region Name:")
top_k = st.sidebar.slider("Number of Results to Display:", 1, 10, 5)

if query:
    results = search_region(query, top_k)
    if not results.empty:
        st.subheader("Search Results")
        st.dataframe(results)

        fig_query = px.bar(
            results,
            x="Region",
            y="Production_2020",
            title=f"Top {top_k} Results for Query: {query}",
            labels={"Region": "Region", "Production_2020": "Production (tons)"},
            color="Production_2020",
            height=400,
            width=800  # Adjust width if necessary
        )
        fig_query.update_layout(
            xaxis=dict(
                tickangle=35,  # Rotate text for better readability
                automargin=True,
                tickmode="array",
                tickvals=[i for i in range(len(results["Region"]))],  # Space tick values
                ticktext=results["Region"]  # Ensure labels match
            )
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
    width=1200  # Adjust width for better spacing
)
fig.update_layout(
    xaxis=dict(
        tickangle=35,  # Rotate text for better readability
        automargin=True,
        tickmode="linear"  # Ensure consistent spacing
    )
)
st.plotly_chart(fig)