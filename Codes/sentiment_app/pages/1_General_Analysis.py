import streamlit as st
from utils.analyze import (
    plot_character_line_counts, 
    plot_character_heatmap, 
    plot_scene_intensity, 
    plot_character_interaction_network,
    plot_average_dialogue_length
)

st.set_page_config(page_title="General Analysis", layout="wide")

st.title("General Script Analysis")

st.markdown("""
This page provides a general analysis of the uploaded movie script. 
Explore character line counts, scene-by-scene heatmaps, overall scene intensity, and more.
""")

# Determine which dataframe to use (analyzed or raw)
df = None
if 'df_analyzed' in st.session_state:
    df = st.session_state.df_analyzed
elif 'df_raw' in st.session_state:
    df = st.session_state.df_raw

# Main content
if df is not None:
    # Get the dataframe from session state
    df_script = df
    
    st.header("Character & Scene Overview")
    
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
    
    st.header("Deeper Script Analysis")

    st.subheader("Scene Intensity Across the Script")
    fig_scene_intensity = plot_scene_intensity(df_script)
    st.pyplot(fig_scene_intensity)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Character Interaction Network")
        fig_interaction = plot_character_interaction_network(df_script, top_n=7) 
        st.pyplot(fig_interaction)
    
    with col4:
        st.subheader("Average Dialogue Length")
        fig_avg_length = plot_average_dialogue_length(df_script, top_n=10)
        st.pyplot(fig_avg_length)

    with st.expander("View Processed Data"):
        st.write("Extracted script data:", df_script)
else:
    st.warning("Please upload a script on the main page to begin the analysis.") 