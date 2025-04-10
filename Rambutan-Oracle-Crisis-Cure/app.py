import streamlit as st
import pandas as pd
import base64

def set_background(image_path):
    """
    Converts the image to base64 and sets it as the background.
    """
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_image});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image file
set_background("./res/cristobol.png")

# Initialize session state variables
if "filter_selected" not in st.session_state:
    st.session_state.filter_selected = False

# ---------------------------
# TRANSLATION MAP FOR LABELS
# ---------------------------
translations = {
    "English": {
        "language_label": "Choose Your Language",
        "filter_type_label": "Select a Filter Type",
        "filter_value_label": "Choose",
        "results_title": "Filtered Results:",
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
        "filter_type_label": "Pilih Jenis Filter",
        "filter_value_label": "Pilih",
        "results_title": "Hasil Penyaringan:",
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
# SAMPLE DATA
# ---------------------------
data = {
    "Problem Name": ["Issue 1", "Issue 2"],
    "Category": ["Category A", "Category B"],
    "Cause": ["Cause A", "Cause B"],
    "Symptoms": ["Symptom A", "Symptom B"],
    "Impact": ["Impact A", "Impact B"],
    "Solution": ["Solution A", "Solution B"],
}
english_df = pd.DataFrame(data)
indonesian_df = pd.DataFrame(data)  # Simulating identical structure for simplicity

# ---------------------------
# STREAMLIT SIDEBAR UI WITH CONDITIONAL DROPDOWNS
# ---------------------------

# Sidebar for title, logo, and dropdowns
st.sidebar.title("Rambutan Oracle Crisis Cure")
st.sidebar.image("./ab_logo.png", use_container_width=True)  # Ensure this file exists
st.sidebar.write("### Andi Bima")

# Language selection dropdown (always visible)
language = st.sidebar.selectbox(
    "Choose Your Language",
    options=["English", "Indonesian"],
)

localized = translations[language]  # Get localized translations

# Filter type dropdown (always visible)
filter_type = st.sidebar.selectbox(
    localized["filter_type_label"],
    options=list(localized["filter_options"].values()),
)

# **Update session state when user selects a filter type**
if filter_type:
    st.session_state.filter_selected = True

# Show filter value dropdown **ONLY IF a filter type is selected**
if st.session_state.filter_selected:
    filter_column = list(localized["filter_options"].keys())[
        list(localized["filter_options"].values()).index(filter_type)
    ]
    df = english_df if language == "English" else indonesian_df
    filter_values = df[filter_column].dropna().unique()

    filter_value = st.sidebar.selectbox(
        f"{localized['filter_value_label']} {filter_type}",
        options=filter_values,
        index=None,  # Keeps dropdown empty initially
    )

    # Display filtered results only after filter value is selected
    if filter_value:
        filtered_df = df[df[filter_column].str.lower() == filter_value.lower()]
        st.write(f"### {localized['results_title']}")
        if not filtered_df.empty:
            st.dataframe(filtered_df.style.set_properties(
                subset=['Symptoms', 'Solution'], **{'white-space': 'pre-wrap', 'width': '500px'}
            ))
        else:
            st.write(localized["no_results"])
