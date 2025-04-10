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
# FUNCTION TO SET BACKGROUND IMAGE
# ---------------------------
def set_background(image_path):
    """
    Sets a PNG image as the background for the Streamlit app.
    """
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_image});
            background-size: 750px 500px; /* Slightly larger image */
            background-repeat: no-repeat;
            background-position: bottom center; /* Move lower for balance */
        }}
        .result-table {{
            border-radius: 50%; /* Circular effect */
            opacity: 0.9; /* Transparent glassy effect */
            box-shadow: 0px 0px 20px rgba(255, 255, 255, 0.6); /* Glow around the table */
            padding: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True