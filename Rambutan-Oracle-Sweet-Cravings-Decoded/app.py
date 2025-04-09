import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import plotly.express as px
import streamlit as st

#img URL
#image_url = 'ab_logo.png'
# Custom styling for purple highlight color
st.markdown(
    """
    <style>
    .main-title {
        color: #6A5ACD; /* Updated purple */ 
        font-size: 40px; 
        font-weight: bold; 
        text-align: center;
    }
    .description {
        color: #6A5ACD; /* Updated purple */ 
        font-size: 20px; 
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load and clean datasets
@st.cache_data
def load_data():
    # Load datasets
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

# Sidebar content
with st.sidebar:
    st.markdown(
    '<p style="text-align: center; font-size: 20px">Explore the Sweet Cravings!"</p>', unsafe_allow_html=True
    )
    st.image("Rambutan-Oracle-Sweet-Cravings-Decoded/ab_logo.png", width=200, caption="Andi Bima")
    #st.markdown(
    #f"<div style='text-align: center;'><img src='{image_url}' caption='Andi Bima' style='width:200px;'/></div>",
    #unsafe_allow_html=True
#)
    st.markdown(
    '<p class="description">Use the sidebar to explore and interact with the rambutan consumption data. Visualize trends, insights, and more!</p>',
    unsafe_allow_html=True
)  # Description
    region_query = st.text_input("Enter Region Name:", help="Type a region to analyze its consumption trends.")
    top_k = st.slider("Number of Results", min_value=1, max_value=5, step=1, value=3)

    # Add accordion sections (expanders) below slider
    with st.expander("About the App"):
        st.write(
            """
            Welcome to the Rambutan Consumption Analysis Dashboard! 
            This app provides insights into rambutan consumption trends across various regions. 
            Explore patterns in the dataset and visualize data interactively. Perfect for market analysis and understanding regional rambutan preferences.
            """
        )
    with st.expander("About the Data"):
        st.write(
            """
            The dataset includes cleaned information about rambutan consumption, collected from different regions.
            - **Fields**:
              - `Region`: Name of the region
              - `Consumption`: Per capita consumption (in tons)
            - **Sources**: Derived from a comprehensive study conducted in 2024.
            """
        )

# Main Page Content
st.markdown('<h2 class="main-title">Rambutan Oracle: Sweet Cravings Decoded</h2>', unsafe_allow_html=True)  # Title
#st.markdown('<h2 class="subtitle">Explore the Sweet Cravings!</h2>', unsafe_allow_html=True)  # Subtitle
#st.markdown('<h2 class="subtitle">Discover Rambutan Consumption Trends</h2>', unsafe_allow_html=True)  # Subtitle
st.markdown(
    '<p style="color: #6A5ACD; font-size: 30px; text-align: center; margin-bottom: 40px">Discover Rambutan Consumption Trends!</p>',
    unsafe_allow_html=True
)  # Description

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
    xaxis=dict(
        tickangle=-45,
        automargin=True,
        title=dict(standoff=10),
        showgrid=False,
        type="category",
        rangeslider=dict(visible=True)
    ),
    title_font=dict(color="#6A5ACD", size=24),
    title_x=0.1
)
st.plotly_chart(fig)