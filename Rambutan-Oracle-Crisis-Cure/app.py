import streamlit as st
import pandas as pd

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
        "results_title": "Rambutan Neural Codex:",
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
# STREAMLIT UI WITH CUSTOM CSS
# ---------------------------

# Add custom CSS for larger dropdown labels
def add_custom_css():
    st.markdown(
        """
        <style>
        .sidebar-centered-text {
            text-align: center;
            font-size: 1.2em;
            font-weight: bold;
        }

        .sidebar .stSelectbox label {
            font-size: 1.8em; /* Adjust size as needed */
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_custom_css()

# Add a title to the main app
st.title("Rambutan Oracle Cure for Crisis")  # Title added to the main app

# Sidebar for logo, text, and dropdowns
st.sidebar.title("Crisis Cure Assistant")  # Title added above the logo
st.sidebar.image("./ab_logo.png", use_container_width=True)  # Updated to use `use_container_width`
#st.sidebar.write("### Andi Bima")  # Add text below the logo


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
    filter_column = list(localized["filter_options"].keys())[
        list(localized["filter_options"].values()).index(filter_type)
    ]

    # Use the selected language's dataset
    df = english_df if language == "English" else indonesian_df

    # Populate filter values dynamically
    filter_values = df[filter_column].dropna().unique()
    placeholder = "Select Your Divination"  # Simulating placeholder behavior
    filter_value = st.sidebar.selectbox(
        f"{localized['filter_value_label']} {filter_type}",
        options=filter_values
    )

    # Display filtered results only after a real option is selected (not placeholder)
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
