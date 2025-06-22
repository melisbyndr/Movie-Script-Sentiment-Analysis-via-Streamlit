import streamlit as st
import pandas as pd
import os
import json
from utils.sentiment_analysis import (
    SENTIMENT_MODELS, 
    EMOTION_MODELS, 
    analyze_text,
    merge_results_with_df,
    save_model_analysis,
    load_model_analyses,
    load_analysis_by_id,
    get_analysis_metadata,
    compare_model_results,
    plot_model_comparison,
    delete_analysis
)

st.set_page_config(page_title="Model Management", layout="wide")

st.title("Model Management & Comparison")
st.markdown("""
This page allows you to manage multiple model analyses, compare different model configurations, 
and track the performance of various sentiment and emotion analysis models.
""")

# --- Load existing analyses ---
all_analyses = load_model_analyses()

if all_analyses:
    st.header("Saved Model Analyses")
    
    # Create a comprehensive table of saved analyses
    analyses_data = []
    for analysis_id, metadata in all_analyses.items():
        analyses_data.append({
            'Analysis Name': metadata['analysis_name'],
            'Sentiment Model': metadata['sentiment_model'].split('/')[-1],
            'Emotion Model': metadata['emotion_model'].split('/')[-1],
            'Dialogues': metadata['total_dialogues'],
            'Characters': metadata['total_characters'],
            'Scenes': metadata['total_scenes'],
            'Created': metadata['created_at'][:19],
            'Analysis ID': analysis_id
        })
    
    analyses_df = pd.DataFrame(analyses_data)
    
    # Display analyses table
    st.dataframe(analyses_df.drop('Analysis ID', axis=1), use_container_width=True)
    
    # Analysis selection for comparison
    st.subheader("Model Comparison")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_analyses = st.multiselect(
            "Select analyses to compare (choose 2 or more):",
            options=analyses_df['Analysis Name'].tolist(),
            help="Select multiple analyses to compare their results"
        )
    
    with col2:
        if st.button("Compare Selected Models", key="compare_btn"):
            if len(selected_analyses) >= 2:
                # Get analysis IDs for selected analyses
                name_to_id = dict(zip(analyses_df['Analysis Name'], analyses_df['Analysis ID']))
                selected_ids = [name_to_id[name] for name in selected_analyses if name in name_to_id]
                
                if len(selected_ids) >= 2:
                    # Compare results
                    comparison_df = compare_model_results(selected_ids)
                    if comparison_df is not None:
                        st.session_state.comparison_results = comparison_df
                        st.session_state.selected_analysis_ids = selected_ids
                        st.success(f"Comparing {len(selected_analyses)} analyses!")
                        st.rerun()
                    else:
                        st.error("Failed to compare analyses. Please try again.")
                else:
                    st.warning("Could not find analysis IDs for selected analyses.")
            else:
                st.warning("Please select at least 2 analyses to compare.")
    
    # Show comparison results
    if 'comparison_results' in st.session_state and 'selected_analysis_ids' in st.session_state:
        st.subheader("Model Comparison Results")
        
        comparison_df = st.session_state.comparison_results
        
        # Display detailed comparison table
        st.dataframe(comparison_df, use_container_width=True)
        
        # Show comparison plots
        fig_comparison = plot_model_comparison(comparison_df)
        if fig_comparison:
            st.pyplot(fig_comparison)
        
        # Detailed metrics comparison
        st.subheader("Detailed Metrics Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Sentiment Analysis Metrics**")
            sentiment_metrics = comparison_df[['analysis_name', 'avg_sentiment_score', 
                                             'sentiment_positive_pct', 'sentiment_negative_pct']]
            st.dataframe(sentiment_metrics, use_container_width=True)
        
        with col2:
            st.write("**Emotion Analysis Metrics**")
            emotion_metrics = comparison_df[['analysis_name', 'avg_emotion_score', 
                                           'top_emotion', 'top_emotion_pct']]
            st.dataframe(emotion_metrics, use_container_width=True)
        
        # Load specific analysis for detailed view
        st.subheader("Load Analysis for Detailed View")
        selected_analysis = st.selectbox(
            "Choose an analysis to view in detail:",
            options=comparison_df['analysis_name'].tolist()
        )
        
        if st.button("Load Analysis", key="load_analysis_btn"):
            analysis_id = comparison_df[comparison_df['analysis_name'] == selected_analysis]['analysis_id'].iloc[0]
            df_loaded = load_analysis_by_id(analysis_id)
            if df_loaded is not None:
                st.session_state.df_analyzed = df_loaded
                st.session_state.current_analysis_id = analysis_id
                st.success(f"Loaded analysis: {selected_analysis}")
                st.info("Go to the Sentiment Analysis page to view detailed results.")
            else:
                st.error("Failed to load analysis.")
    
    # Analysis management
    st.header("Analysis Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Delete All Analyses", key="delete_all_btn"):
            for analysis_id in all_analyses.keys():
                delete_analysis(analysis_id)
            st.success("All analyses deleted!")
            st.rerun()
    
    with col2:
        if st.button("Refresh Analysis List", key="refresh_btn"):
            st.rerun()
    
    with col3:
        # Individual analysis deletion
        analysis_to_delete = st.selectbox(
            "Select analysis to delete:",
            options=analyses_df['Analysis Name'].tolist(),
            key="delete_select"
        )
        
        if st.button("Delete Selected", key="delete_selected_btn"):
            name_to_id = dict(zip(analyses_df['Analysis Name'], analyses_df['Analysis ID']))
            if analysis_to_delete in name_to_id:
                delete_analysis(name_to_id[analysis_to_delete])
                st.success(f"Deleted analysis: {analysis_to_delete}")
                st.rerun()
    
    # Export functionality
    st.header("Export Results")
    
    if 'comparison_results' in st.session_state:
        comparison_df = st.session_state.comparison_results
        
        # Download comparison results
        st.download_button(
            label="Download Comparison Results as CSV",
            data=comparison_df.to_csv(index=False),
            file_name="model_comparison_results.csv",
            mime="text/csv"
        )
    
    # Download all analyses metadata
    st.download_button(
        label="Download All Analyses Metadata as JSON",
        data=json.dumps(all_analyses, indent=4),
        file_name="all_analyses_metadata.json",
        mime="application/json"
    )

else:
    st.info("No saved analyses found. Run some analyses on the Sentiment Analysis page to see them here.")

# --- Quick Analysis Section ---
st.header("Quick Analysis")
st.markdown("""
If you have a script loaded, you can quickly run a new analysis here.
""")

if 'df_raw' in st.session_state:
    df_raw = st.session_state.df_raw
    
    st.subheader("Run New Analysis")
    
    # Analysis name
    analysis_name = st.text_input(
        "Analysis Name:",
        placeholder="e.g., BERT Large + RoBERTa GoEmotions",
        help="Give your analysis a descriptive name"
    )
    
    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        sentiment_model_choice = st.selectbox(
            "Sentiment Model:", 
            options=list(SENTIMENT_MODELS.keys())
        )
    with col2:
        emotion_model_choice = st.selectbox(
            "Emotion Model:", 
            options=list(EMOTION_MODELS.keys())
        )
    
    # Show model details
    st.info(f"""
    **Selected Configuration:**
    - **Sentiment:** {sentiment_model_choice} ({SENTIMENT_MODELS[sentiment_model_choice]})
    - **Emotion:** {emotion_model_choice} ({EMOTION_MODELS[emotion_model_choice]})
    - **Script:** {st.session_state.original_filename}
    - **Dialogues:** {len(df_raw)} lines
    """)
    
    # Run analysis
    if st.button("Run Analysis", key="quick_analysis_btn"):
        if not analysis_name:
            st.error("Please provide an analysis name.")
        else:
            try:
                with st.spinner("Running analysis..."):
                    dialogues = df_raw['clean_dialogue'].dropna().tolist()
                    
                    if not dialogues:
                        st.error("The script contains no analyzable dialogue.")
                    else:
                        # Run analyses
                        sentiment_results = analyze_text(dialogues, "sentiment-analysis", sentiment_model_choice)
                        emotion_results = analyze_text(dialogues, "text-classification", emotion_model_choice)
                        
                        # Merge results
                        df_analyzed = merge_results_with_df(df_raw, sentiment_results, emotion_results)
                        
                        # Save with new system
                        analysis_id, metadata = save_model_analysis(
                            df_analyzed, 
                            st.session_state.file_hash, 
                            st.session_state.original_filename,
                            SENTIMENT_MODELS[sentiment_model_choice],
                            EMOTION_MODELS[emotion_model_choice],
                            analysis_name
                        )
                        
                        st.success(f"Analysis complete! Analysis ID: {analysis_id}")
                        st.info(f"Analysis saved as: {metadata['analysis_name']}")
                        st.rerun()

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.error("Please check your internet connection and try again.")

elif 'df_analyzed' in st.session_state:
    st.info("You have an analysis loaded. Go to the Sentiment Analysis page to view results or run new analyses.")
else:
    st.warning("Please upload a script on the main page to run analyses.")

# --- Statistics Section ---
if all_analyses:
    st.header("Analysis Statistics")
    
    # Calculate statistics
    total_analyses = len(all_analyses)
    unique_sentiment_models = set()
    unique_emotion_models = set()
    total_dialogues = 0
    
    for metadata in all_analyses.values():
        unique_sentiment_models.add(metadata['sentiment_model'].split('/')[-1])
        unique_emotion_models.add(metadata['emotion_model'].split('/')[-1])
        total_dialogues += metadata['total_dialogues']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", total_analyses)
    
    with col2:
        st.metric("Unique Sentiment Models", len(unique_sentiment_models))
    
    with col3:
        st.metric("Unique Emotion Models", len(unique_emotion_models))
    
    with col4:
        st.metric("Total Dialogues Analyzed", total_dialogues)
    
    # Model usage statistics
    st.subheader("Model Usage Statistics")
    
    sentiment_usage = {}
    emotion_usage = {}
    
    for metadata in all_analyses.values():
        sentiment_model = metadata['sentiment_model'].split('/')[-1]
        emotion_model = metadata['emotion_model'].split('/')[-1]
        
        sentiment_usage[sentiment_model] = sentiment_usage.get(sentiment_model, 0) + 1
        emotion_usage[emotion_model] = emotion_usage.get(emotion_model, 0) + 1
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sentiment Model Usage**")
        for model, count in sorted(sentiment_usage.items(), key=lambda x: x[1], reverse=True):
            st.write(f"- {model}: {count} analyses")
    
    with col2:
        st.write("**Emotion Model Usage**")
        for model, count in sorted(emotion_usage.items(), key=lambda x: x[1], reverse=True):
            st.write(f"- {model}: {count} analyses") 