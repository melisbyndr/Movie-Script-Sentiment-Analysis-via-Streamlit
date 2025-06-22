import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import hashlib
import os
import json
from utils.extract_data import extract_script_info
from utils.analyze import (
    plot_character_line_counts, 
    plot_character_heatmap, 
    plot_scene_intensity, 
    plot_character_interaction_network,
    plot_average_dialogue_length
)

# --- Configuration ---
CACHE_DIR = "analysis_cache"
METADATA_FILE = os.path.join(CACHE_DIR, "metadata.json")

# --- Helper Functions ---
def ensure_cache_dir():
    """Creates the cache directory and metadata file if they don't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'w') as f:
            json.dump({}, f)

def get_file_hash(file_content):
    """Computes SHA256 hash of the file content."""
    return hashlib.sha256(file_content).hexdigest()

def load_metadata():
    """Loads the metadata from the JSON file."""
    with open(METADATA_FILE, 'r') as f:
        return json.load(f)

def get_cached_analyses():
    """Returns a dictionary of cached analyses from metadata."""
    return load_metadata()

# --- Main App ---
ensure_cache_dir()

st.set_page_config(
    page_title="Movie Script Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Movie Script Analyzer")
st.markdown("""
Welcome! This tool analyzes movie scripts for character interactions, scene structures, and emotional arcs.

**How to use:**
1.  **Upload a New Script**: Use the sidebar to upload a new PDF script.
2.  **Load a Previous Analysis**: Select a previously analyzed script from the sidebar to instantly view its results.
3.  **Navigate Pages**: Explore the analysis pages that appear after a script is loaded.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("Manage Scripts")
    
    # --- File Uploader ---
    uploaded_file = st.file_uploader("Upload a new movie script (PDF only)", type=["pdf"])

    if uploaded_file:
        file_content = uploaded_file.getvalue()
        file_hash = get_file_hash(file_content)
        
        # Check if analysis is cached
        if os.path.exists(os.path.join(CACHE_DIR, f"{file_hash}.parquet")):
            st.info("This script has been analyzed before. Loading results...")
            df_analyzed = pd.read_parquet(os.path.join(CACHE_DIR, f"{file_hash}.parquet"))
            st.session_state.df_analyzed = df_analyzed
            st.session_state.file_hash = file_hash
            st.success("Results loaded successfully!")
        else:
            st.info("New script detected. Processing...")
            with st.spinner("Extracting text and preparing for analysis..."):
                doc = fitz.open(stream=file_content, filetype="pdf")
                script_text = "".join(page.get_text() for page in doc)
                df_raw = extract_script_info(script_text)
                
                # Save raw data to cache for future re-analysis
                raw_data_file = os.path.join(CACHE_DIR, f"raw_{file_hash}.parquet")
                df_raw.to_parquet(raw_data_file)
                
                st.session_state.df_raw = df_raw # Store raw df for analysis page
                st.session_state.file_hash = file_hash
                st.session_state.original_filename = uploaded_file.name
                # Clear any previously loaded analyzed data
                if 'df_analyzed' in st.session_state:
                    del st.session_state.df_analyzed
                st.success("Script ready for analysis.")
                st.warning("Please go to the 'Sentiment Analysis' page to perform the full analysis.")

    # --- Cached Analyses Loader ---
    st.markdown("---")
    st.header("Load Previous Analysis")
    cached_analyses = get_cached_analyses()

    if not cached_analyses:
        st.caption("No scripts analyzed yet.")
    else:
        # Create a selectbox for loading cached scripts
        selected_script = st.selectbox(
            "Choose a script to load:",
            options=list(cached_analyses.values()),
            index=None,
            placeholder="Select a script..."
        )
        if selected_script:
            # Find the hash corresponding to the selected filename
            selected_hash = next((h for h, name in cached_analyses.items() if name == selected_script), None)
            if selected_hash:
                df_analyzed = pd.read_parquet(os.path.join(CACHE_DIR, f"{selected_hash}.parquet"))
                st.session_state.df_analyzed = df_analyzed
                st.session_state.file_hash = selected_hash
                
                # Check if raw data exists, if not, try to extract it from the analyzed data
                raw_data_file = os.path.join(CACHE_DIR, f"raw_{selected_hash}.parquet")
                if not os.path.exists(raw_data_file):
                    # Create raw data from analyzed data (without sentiment/emotion columns)
                    raw_columns = ['scene_id', 'character', 'clean_dialogue']
                    if all(col in df_analyzed.columns for col in raw_columns):
                        df_raw = df_analyzed[raw_columns].copy()
                        df_raw.to_parquet(raw_data_file)
                        st.session_state.df_raw = df_raw
                
                st.success(f"Loaded '{selected_script}' successfully!")

# Display a message on the main page depending on the state
if 'df_analyzed' in st.session_state or 'df_raw' in st.session_state:
     st.info("Script loaded. Please navigate to the analysis pages using the sidebar.")
else:
     st.info("Get started by uploading a new script or loading a previous analysis from the sidebar.")

# Check if the dataframe is in the session state
if 'df_script' not in st.session_state:
    st.info("Please upload a PDF script using the sidebar to begin analysis.")
else:
    # Get the dataframe from session state
    df_script = st.session_state.df_script
    
    st.header("General Script Analysis")
    
    # Create columns for side-by-side plots
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Characters by Line Count")
        fig_line_counts = plot_character_line_counts(df_script, top_n=10)
        st.pyplot(fig_line_counts)

    with col2:
        st.subheader("Character Heatmap by Scene")
        fig_heatmap = plot_character_heatmap(df_script, top_n=10)
        st.pyplot(fig_heatmap)
    
    # Show the raw data in an expander for verification
    with st.expander("View Processed Data"):
        st.write("Extracted script data:", df_script)
    
    # --- Deeper Analysis Section ---
    st.header("Deeper Script Analysis")

    # 1. Scene Intensity Plot
    st.subheader("Scene Intensity Across the Script")
    fig_scene_intensity = plot_scene_intensity(df_script)
    st.pyplot(fig_scene_intensity)

    # 2. Columns for Interaction and Dialogue Length
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Character Interaction Network")
        fig_interaction = plot_character_interaction_network(df_script, top_n=7) 
        st.pyplot(fig_interaction)
    
    with col4:
        st.subheader("Average Dialogue Length")
        fig_avg_length = plot_average_dialogue_length(df_script, top_n=10)
        st.pyplot(fig_avg_length)
