import streamlit as st
import pandas as pd
import base64

# ---------------------------
# LOAD THE DATASETS
# ---------------------------
english_df = pd.read_csv('./dataset/cleaned_rambutan_problems_symptoms_solution_en.csv')
indonesian_df = pd.read_csv('./dataset/cleaned_rambutan_problems_symptoms_solution_id.csv')
makassar_df = pd.read_csv('./dataset/cleaned_rambutan_problems_symptoms_solution_mak.csv')

# ---------------------------
# TRANSLATION MAP FOR LABELS
# ---------------------------
translations = {
    "English": {
        "language_label": "Choose Your Language",
        "filter_type_label": "The Codex of Healing",
        "filter_value_label": "Choose",
        "results_title": "The Codex of Healing:",
        "no_results": "No results match the selection.",
        "filter_options": {
            "Problem Name": "Problem Name",
            "Category": "Category",
            "Cause": "Cause",
            "Symptoms": "Symptoms",
            "Impact": "Impact",
            "Solution": "Solution",
        },
    },
    "Indonesian": {
        "language_label": "Pilih Bahasa",
        "filter_type_label": "Rambutan Neural Codex",
        "filter_value_label": "Pilih",
        "results_title": "Kitab Neural Rambutan:",
        "no_results": "Tidak ada hasil yang sesuai dengan pilihan Anda.",
        "filter_options": {
            "Problem Name": "Nama Masalah",
            "Category": "Kategori",
            "Cause": "Penyebab",
            "Symptoms": "Gejala",
            "Impact": "Dampak",
            "Solution": "Solusi",
        },
    },
    "Makassar": {
        "language_label": "Pilei Bahasa",
        "filter_type_label": "Pilei Jenis Filter",
        "filter_value_label": "Pilei",
        "results_title": "Hasil Penyaringan:",
        "no_results": "Tena nia hasil yang cocoka siagang pappilenu.",
        "filter_options": {
            "Problem Name": "Areng Masala",
            "Category": "Kategori",
            "Cause": "Masalana",
            "Symptoms": "Gejala",
            "Impact": "Akiba na",
            "Solution": "Solusina",
        },
    },
}

# ---------------------------
# FUNCTION TO SET BACKGROUND IMAGE
# ---------------------------
def set_background(image_path):
    """
    Ensures the background image always shows entirely, regardless of screen size.
    """
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_image});
            background-size: contain; /* Ensure the entire image is visible */
            background-repeat: no-repeat;
            background-position: center; /* Keeps the image centered */
            background-attachment: fixed; /* Keeps the image fixed while scrolling */
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Apply the background
set_background("./res/cristobol_v2.jpeg")

# ---------------------------
# STREAMLIT UI WITH CUSTOM CSS (UPDATED TEXT COLOR)
# ---------------------------

def add_custom_css():
    st.markdown(
        """
        <style>
        .sidebar-centered-text {
            text-align: center;
            font-size: 1.2em;
            font-weight: bold;
        }
        .main-title {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            margin-top: 20px;
            color: #ffffff !important; /* Bright white for readability */
        }
        .results-title {
            font-size: 1.8em;
            font-weight: bold;
            color: #ffffff !important; /* Bright white for readability */
        }
        .sidebar .stSelectbox label {
            font-size: 1.8em;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_custom_css()

# Main title
st.markdown('<h2 class="main-title">Mantra Neural Rambutan</h2>', unsafe_allow_html=True)

# Sidebar content
st.sidebar.title("Mantra Neural Rambutan")  # Sidebar title
st.sidebar.image("./ab_logo.png", use_container_width=True)  # Logo
st.sidebar.markdown('<p class="sidebar-centered-text">Andi Bima</p>', unsafe_allow_html=True)  # Centered text

# Language selection dropdown
language = st.sidebar.selectbox(
    "Choose Your Language",
    ["English", "Indonesian", "Makassar"],
)

localized = translations[language]

# Filter type dropdown
filter_type = st.sidebar.selectbox(
    localized["filter_type_label"],
    options=list(localized["filter_options"].values())
)

# Conditional rendering for filter value dropdown
if filter_type:
    # Determine corresponding column in the dataset
    filter_column = list(localized["filter_options"].keys())[\
        list(localized["filter_options"].values()).index(filter_type)\
    ]

    # Use the selected language's dataset
    df = english_df if language == "English" else indonesian_df

    # Populate filter values dynamically
    filter_values = df[filter_column].dropna().unique()
    filter_value = st.sidebar.selectbox(
        f"{localized['filter_value_label']} {filter_type}",
        options=filter_values
    )

    # Display filtered results
    if filter_value:
        filtered_df = df[df[filter_column].str.lower() == filter_value.lower()]
        st.markdown('<h3 class="results-title">The Codex of Healing:</h3>', unsafe_allow_html=True)
        if not filtered_df.empty:
            st.dataframe(filtered_df.style.set_properties(
                subset=["Symptoms", "Solution"],
                **{'white-space': 'pre-wrap', 'width': '500px'}
            ))
        else:
            st.write(localized["no_results"])
