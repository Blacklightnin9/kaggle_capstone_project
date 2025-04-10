import streamlit as st
import pandas as pd
import base64

# ---------------------------
# LOAD THE DATASETS
# ---------------------------
english_df = pd.read_csv('./dataset/cleaned_rambutan_problems_symptoms_solution_en.csv')
indonesian_df = pd.read_csv('./dataset/cleaned_rambutan_problems_symptoms_solution_id.csv')

# ---------------------------
# TRANSLATION MAP FOR LABELS
# ---------------------------
translations = {
    "English": {
        "language_label": "Choose Your Language",
        "filter_type_label": "The Codex of Healing",
        "filter_value_label": "Choose",
        "results_title": "Mantra Neural Rambutan:",
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
}

# ---------------------------
# FUNCTION TO SET BACKGROUND IMAGE DYNAMICALLY
# ---------------------------
def set_background(image_path):
    """
    Dynamically adjusts background image with sidebar state.
    """
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        body {{
            background-image: url(data:image/png;base64,{encoded_image});
            background-size: cover; /* Ensures full coverage */
            background-repeat: no-repeat;
            background-position: center top;
            transition: background-position 0.5s ease-in-out; /* Smooth position changes */
        }}
        .sidebar-open {{
            background-position: left top; /* Adjusted when sidebar is open */
        }}
        .sidebar-closed {{
            background-position: center top; /* Original position */
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
except Exception as e:
        st.error("Failed to load background image. Please check the file path and format.")
        st.write(f"Error details: {e}")


# Apply the dynamic background
set_background("./res/cristobol.png")

# ---------------------------
# STREAMLIT UI WITH DYNAMIC SIDEBAR STATE
# ---------------------------

# Sidebar content
st.sidebar.title("Mantra Neural Rambutan")  # Sidebar title
toggle_sidebar = st.sidebar.checkbox("Toggle Sidebar")  # Checkbox for sidebar state
st.sidebar.image("./ab_logo.png", use_container_width=True)  # Logo
st.sidebar.markdown('<p class="sidebar-centered-text">Andi Bima</p>', unsafe_allow_html=True)

if toggle_sidebar:
    st.markdown('<div class="sidebar-open"></div>', unsafe_allow_html=True)  # Sidebar open CSS class
    st.write("Sidebar is open! You can adjust the background position dynamically.")
else:
    st.markdown('<div class="sidebar-closed"></div>', unsafe_allow_html=True)  # Sidebar closed CSS class
    st.write("Sidebar is closed! Background returns to its original position.")

# Main title
st.markdown('<h2 class="main-title">Mantra Neural Rambutan</h2>', unsafe_allow_html=True)

# Language selection dropdown
language = st.sidebar.selectbox(
    "Choose Your Language",
    ["English", "Indonesian"],
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
        st.write(f"### {localized['results_title']}")
        if not filtered_df.empty:
            st.dataframe(filtered_df.style.set_properties(
                subset=["Symptoms", "Solution"],
                **{'white-space': 'pre-wrap', 'width': '500px'}
            ))
        else:
            st.write(localized["no_results"])