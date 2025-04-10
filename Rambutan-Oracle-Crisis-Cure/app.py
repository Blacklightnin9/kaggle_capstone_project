import streamlit as st
import pandas as pd

# Function to set background image
def set_background(image_path):
    """
    This function sets a PNG image as the background for the Streamlit app.
    """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_path});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image file
set_background("./cristobol.png")

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
# STREAMLIT UI
# ---------------------------
# Language selection
language = st.selectbox(
    "Choose Your Language",
    options=["English", "Indonesian"],
)

localized = translations[language]  # Get localized translations
st.write(f"### {localized['language_label']}")

# Filter type selection
filter_type = st.selectbox(
    localized["filter_type_label"],
    options=list(localized["filter_options"].values()),
)

# Populate filter values based on filter type
if filter_type:
    filter_column = list(localized["filter_options"].keys())[
        list(localized["filter_options"].values()).index(filter_type)
    ]
    df = english_df if language == "English" else indonesian_df
    filter_values = df[filter_column].dropna().unique()
    filter_value = st.selectbox(
        f"{localized['filter_value_label']} {filter_type}",
        options=filter_values,
    )

    # Display filtered results
    if filter_value:
        filtered_df = df[df[filter_column].str.lower() == filter_value.lower()]
        st.write(f"### {localized['results_title']}")
        if not filtered_df.empty:
            st.dataframe(filtered_df.style.set_properties(
                subset=['Symptoms', 'Solution'], **{'white-space': 'pre-wrap', 'width': '500px'}
            ))
        else:
            st.write(localized["no_results"])
