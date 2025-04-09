import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import plotly.express as px
import streamlit as st
from st_aggrid import AgGrid

# Load and clean datasets
@st.cache_data
def load_data():
    # Load datasets
    data_en = pd.read_csv("cleaned_Konsumsi_Rambutan_Perkapita_2024_en.csv", delimiter=",")
    data_id = pd.read_csv("cleaned_Konsumsi_Rambutan_Perkapita_2024_id.csv", delimiter=",")

    # Split combined columns if necessary
    if len(data_en.columns) == 1:
        data_en[['Region', 'Consumption']] = data_en.iloc[:, 0].str.split(',', expand=True)
    if len(data_id.columns) == 1:
        data_id[['Region', 'Consumption']] = data_id.iloc[:, 0].str.split(',', expand=True)

    # Convert "Consumption" to numeric
    data_en["Consumption"] = pd.to_numeric(data_en["Consumption"], errors="coerce").fillna(0)
    data_id["Consumption"] = pd.to_numeric(data_id["Consumption"], errors="coerce").fillna(0)

    # Combine datasets
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
    # Load the SentenceTransformer model from a local path
    model_path = "models/all-MiniLM-L6-v2"  # Path to the local model folder
    try:
        model = SentenceTransformer(model_path)
        st.write("Model loaded successfully!")  # Debugging message
    except OSError:
        st.error("Model files not found! Please ensure they are downloaded and available locally.")
        return None, None

    batch_size = 32
    embeddings = []

    # Check if data contains the 'Region' column
    if "Region" not in data.columns or data.empty:
        st.error("Dataset is empty or missing the 'Region' column!")
        return None, None

    for i in tqdm(range(0, len(data), batch_size), desc="Generating Embeddings"):
        batch_data = data["Region"].iloc[i : i + batch_size].tolist()
        batch_embeddings = model.encode(batch_data)
        embeddings.extend(batch_embeddings)

    embeddings_np = np.array(embeddings)
    return embeddings_np, model

embeddings_np, model = generate_embeddings(data)

# Build FAISS index
@st.cache_resource
def create_faiss_index(embeddings_np):
    if embeddings_np is None:
        st.error("Embeddings could not be generated! FAISS index cannot be created.")
        return None  # Avoids AttributeError

    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    return index

index = create_faiss_index(embeddings_np)

# Function to query FAISS index
def search_region(query, top_k=5):
    if model is None or index is None:
        st.error("Model or FAISS index is not initialized!")
        return pd.DataFrame()  # Return an empty DataFrame

    query_embedding = model.encode([query.strip()])[0]
    distances, indices = index.search(np.array([query_embedding]), top_k)
    results = data.iloc[indices[0]]
    results = results[["Region", "Consumption"]]
    return results

# Inject custom CSS styling
st.markdown("""
    <style>
    .stTitle {
        color: #FF6F61;
        font-size: 30px;
        font-weight: bold;
    }
    .stHeader {
        color: #333333;
        background-color: #FFD700;
    }
    .stButton > button {
        background-color: #FF6F61;
        color: white;
        border-radius: 8px;
        height: 40px;
        width: 120px;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit layout
st.title("Rambutan Consumption Analysis")
st.image("ab_logo.png", width=150, caption="Andi Bima")
st.write("Explore rambutan consumption data across regions and discover patterns.")

col1, col2 = st.columns(2)

with col1:
    st.header("Interactive Region Search")
    region_query = st.text_input("Enter Region Name:")
    top_k = st.slider("Number of Results", min_value=1, max_value=5, value=3)

    if region_query:
        results = search_region(region_query, top_k)
        if not results.empty:
            st.write(results)
            # Visualization
            fig_query = px.bar(
                results,
                x="Region",
                y="Consumption",
                title=f"Top {top_k} Results for Query: {region_query}",
                labels={"Region": "Region", "Consumption": "Consumption (tons)"},
                color="Consumption"
            )
            fig_query.update_layout(
                title_font=dict(size=20, color="#FF6F61"),
                paper_bgcolor="#F5F5F5",
                plot_bgcolor="#FFFFFF"
            )
            st.plotly_chart(fig_query)
        else:
            st.write("Region not found!")

with col2:
    st.header("Dataset Overview")
    AgGrid(data, theme="streamlit", fit_columns_on_grid_load=True)

# Full dataset visualization
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
    title_font=dict(size=22, color="#FF6F61"),
    xaxis=dict(tickangle=-45, rangeslider=dict(visible=True)),
    paper_bgcolor="#F5F5F5",
    plot_bgcolor="#FFFFFF"
)
st.plotly_chart(fig)