import streamlit as st
import pandas as pd
import os
import json
from utils.sentiment_analysis import (
    SENTIMENT_MODELS, 
    EMOTION_MODELS, 
    analyze_text,
    merge_results_with_df,
    plot_overall_sentiment_distribution,
    plot_overall_emotion_distribution,
    plot_sentiment_timeline_by_scene,
    plot_sentiment_timeline_for_character,
    plot_context_aware_sentiment,
    plot_sentiment_trend_analysis,
    plot_emotion_intensity_analysis,
    plot_character_emotion_intensity,
    analyze_context_aware_sentiment,
    analyze_sentiment_trends,
    calculate_emotion_intensity,
    save_model_analysis,
    load_model_analyses,
    load_analysis_by_id,
    get_analysis_metadata,
    compare_model_results,
    plot_model_comparison,
    delete_analysis,
    load_raw_data_from_cache
)

# --- Configuration ---
CACHE_DIR = "analysis_cache"
METADATA_FILE = os.path.join(CACHE_DIR, "metadata.json")

# --- Helper Functions ---
def save_analysis(df, file_hash, original_filename):
    """Saves the analyzed dataframe and updates metadata."""
    # Save the dataframe
    df.to_parquet(os.path.join(CACHE_DIR, f"{file_hash}.parquet"))
    
    # Update metadata
    with open(METADATA_FILE, 'r+') as f:
        metadata = json.load(f)
        metadata[file_hash] = original_filename
        f.seek(0)
        json.dump(metadata, f, indent=4)
        f.truncate()

@st.cache_data
def convert_df_to_csv(df):
    """Caches the conversion of the dataframe to CSV."""
    return df.to_csv(index=False).encode('utf-8')

# --- Page Setup ---
st.set_page_config(page_title="Sentiment Analysis", layout="wide")
st.title("Sentiment & Emotion Analysis")

st.markdown("""
Analyze the script's dialogue for emotional tone and sentiment. 
You can now run multiple models and compare their results!
""")

# --- Model Management Section ---
st.header("Model Management")

# Load existing analyses
all_analyses = load_model_analyses()

if all_analyses:
    st.subheader("Saved Model Analyses")
    
    # Create a table of saved analyses
    analyses_data = []
    for analysis_id, metadata in all_analyses.items():
        analyses_data.append({
            'Analysis Name': metadata['analysis_name'],
            'Sentiment Model': metadata['sentiment_model'].split('/')[-1],
            'Emotion Model': metadata['emotion_model'].split('/')[-1],
            'Dialogues': metadata['total_dialogues'],
            'Characters': metadata['total_characters'],
            'Created': metadata['created_at'][:19],  # Show date and time
            'Analysis ID': analysis_id
        })
    
    analyses_df = pd.DataFrame(analyses_data)
    
    # Display analyses with selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_analyses = st.multiselect(
            "Select analyses to compare:",
            options=analyses_df['Analysis Name'].tolist(),
            help="Select multiple analyses to compare their results"
        )
    
    with col2:
        if st.button("Compare Selected", key="compare_btn"):
            if len(selected_analyses) >= 2:
                # Get analysis IDs for selected analyses
                selected_ids = []
                # Create a mapping from analysis name to ID
                name_to_id = dict(zip(analyses_df['Analysis Name'], analyses_df['Analysis ID']))
                for name in selected_analyses:
                    if name in name_to_id:
                        selected_ids.append(name_to_id[name])
                
                # Compare results
                comparison_df = compare_model_results(selected_ids)
                if comparison_df is not None:
                    st.session_state.comparison_results = comparison_df
                    st.session_state.selected_analysis_ids = selected_ids
                    st.success(f"Comparing {len(selected_analyses)} analyses!")
                    st.rerun()
            else:
                st.warning("Please select at least 2 analyses to compare.")
    
    # Show comparison results if available
    if 'comparison_results' in st.session_state and 'selected_analysis_ids' in st.session_state:
        st.subheader("Model Comparison Results")
        
        comparison_df = st.session_state.comparison_results
        
        # Display comparison table
        st.dataframe(comparison_df[['analysis_name', 'sentiment_model', 'emotion_model', 
                                  'avg_sentiment_score', 'avg_emotion_score', 
                                  'sentiment_positive_pct', 'sentiment_negative_pct']], 
                    use_container_width=True)
        
        # Show comparison plots
        fig_comparison = plot_model_comparison(comparison_df)
        if fig_comparison:
            st.pyplot(fig_comparison)
        
        # Allow loading a specific analysis
        st.subheader("Load Specific Analysis")
        selected_analysis = st.selectbox(
            "Choose an analysis to view in detail:",
            options=comparison_df['analysis_name'].tolist()
        )
        
        if st.button("Load Analysis", key="load_analysis_btn"):
            analysis_id = comparison_df[comparison_df['analysis_name'] == selected_analysis]['analysis_id'].iloc[0]
            df_loaded = load_analysis_by_id(analysis_id)
            if df_loaded is not None:
                # Get metadata to extract file_hash and original_filename
                metadata = get_analysis_metadata(analysis_id)
                if metadata:
                    st.session_state.file_hash = metadata['file_hash']
                    st.session_state.original_filename = metadata['original_filename']
                
                st.session_state.df_analyzed = df_loaded
                st.session_state.current_analysis_id = analysis_id
                st.success(f"Loaded analysis: {selected_analysis}")
                st.rerun()
    
    # Analysis management
    st.subheader("Manage Analyses")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Delete All Analyses", key="delete_all_btn"):
            for analysis_id in all_analyses.keys():
                delete_analysis(analysis_id)
            st.success("All analyses deleted!")
            st.rerun()
    
    with col2:
        if st.button("Refresh Analysis List", key="refresh_btn"):
            st.rerun()

# --- New Analysis Section ---
st.header("Run New Analysis")

# Check if we have a raw script to analyze
if 'df_raw' in st.session_state:
    df_raw = st.session_state.df_raw
    
    st.subheader("Model Configuration")
    
    # Analysis name
    analysis_name = st.text_input(
        "Analysis Name (optional):",
        placeholder="e.g., DistilBERT + DistilRoBERTa Analysis",
        help="Give your analysis a descriptive name for easy identification"
    )
    
    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        sentiment_model_choice = st.selectbox(
            "Choose a Sentiment Model", 
            options=list(SENTIMENT_MODELS.keys()),
            help="Select the model for sentiment analysis"
        )
    with col2:
        emotion_model_choice = st.selectbox(
            "Choose an Emotion Model", 
            options=list(EMOTION_MODELS.keys()),
            help="Select the model for emotion classification"
        )
    
    # Show model details
    st.info(f"""
    **Selected Models:**
    - **Sentiment:** {sentiment_model_choice} ({SENTIMENT_MODELS[sentiment_model_choice]})
    - **Emotion:** {emotion_model_choice} ({EMOTION_MODELS[emotion_model_choice]})
    """)
    
    # Analysis execution
    if st.button("Run Analysis", key="run_analysis_btn"):
        dialogues = df_raw['clean_dialogue'].dropna().tolist()
        if not dialogues:
            st.error("The script contains no analyzable dialogue.")
        else:
            try:
                with st.spinner("Running sentiment and emotion analysis..."):
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
                    
                    # Update session state
                    st.session_state.df_analyzed = df_analyzed
                    st.session_state.current_analysis_id = analysis_id
                    
                    if 'df_raw' in st.session_state:
                        del st.session_state.df_raw

                st.success(f"Analysis complete! Analysis ID: {analysis_id}")
                st.info(f"Analysis saved as: {metadata['analysis_name']}")
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.error("Please check your internet connection and try again.")

elif 'df_analyzed' in st.session_state:
    st.success("Analysis loaded. Showing results for the current analysis.")
    df_analyzed = st.session_state.df_analyzed
    
    # Show current analysis info
    if 'current_analysis_id' in st.session_state:
        metadata = get_analysis_metadata(st.session_state.current_analysis_id)
        if metadata:
            st.info(f"""
            **Current Analysis:** {metadata['analysis_name']}
            - Sentiment Model: {metadata['sentiment_model'].split('/')[-1]}
            - Emotion Model: {metadata['emotion_model'].split('/')[-1]}
            - Created: {metadata['created_at'][:19]}
            """)
    
    # Add re-run with different models section
    st.subheader("Re-run with Different Models")
    st.markdown("""
    Want to try different models on the same script? You can re-analyze the current script 
    with different sentiment and emotion models to compare results.
    """)
    
    # Model selection for re-run
    col1, col2 = st.columns(2)
    with col1:
        new_sentiment_model = st.selectbox(
            "Choose a new Sentiment Model", 
            options=list(SENTIMENT_MODELS.keys()),
            help="Select a different model for sentiment analysis",
            key="rerun_sentiment"
        )
    with col2:
        new_emotion_model = st.selectbox(
            "Choose a new Emotion Model", 
            options=list(EMOTION_MODELS.keys()),
            help="Select a different model for emotion classification",
            key="rerun_emotion"
        )
    
    # Show new model details
    st.info(f"""
    **New Models Selected:**
    - **Sentiment:** {new_sentiment_model} ({SENTIMENT_MODELS[new_sentiment_model]})
    - **Emotion:** {new_emotion_model} ({EMOTION_MODELS[new_emotion_model]})
    """)
    
    # Re-run analysis button
    new_analysis_name = st.text_input(
        "New Analysis Name:",
        placeholder="e.g., BERT Large + RoBERTa Analysis",
        help="Give your new analysis a descriptive name",
        key="rerun_analysis_name"
    )
    
    if st.button("Re-run Analysis with New Models", key="rerun_analysis_btn"):
        # Get the original raw data
        if 'df_raw' not in st.session_state:
            # Try to get raw data from cache
            df_raw = load_raw_data_from_cache(st.session_state.file_hash)
            if df_raw is None:
                # If raw data doesn't exist, try to create it from the analyzed data
                if 'df_analyzed' in st.session_state:
                    df_analyzed = st.session_state.df_analyzed
                    # Extract raw columns from analyzed data
                    raw_columns = ['scene_id', 'character', 'clean_dialogue']
                    if all(col in df_analyzed.columns for col in raw_columns):
                        df_raw = df_analyzed[raw_columns].copy()
                        st.session_state.df_raw = df_raw
                        st.info("Created raw data from analyzed data for re-analysis.")
                    else:
                        st.error("Original script data not found. Please upload the script again.")
                        st.stop()
                else:
                    st.error("Original script data not found. Please upload the script again.")
                    st.stop()
            else:
                st.session_state.df_raw = df_raw
        
        if 'df_raw' in st.session_state:
            df_raw = st.session_state.df_raw
            dialogues = df_raw['clean_dialogue'].dropna().tolist()
            
            if not dialogues:
                st.error("The script contains no analyzable dialogue.")
            else:
                try:
                    with st.spinner("Running new sentiment and emotion analysis..."):
                        # Run analyses with new models
                        sentiment_results = analyze_text(dialogues, "sentiment-analysis", new_sentiment_model)
                        emotion_results = analyze_text(dialogues, "text-classification", new_emotion_model)
                        
                        # Merge results
                        df_new_analyzed = merge_results_with_df(df_raw, sentiment_results, emotion_results)
                        
                        # Save with new system
                        analysis_id, metadata = save_model_analysis(
                            df_new_analyzed, 
                            st.session_state.file_hash, 
                            st.session_state.original_filename,
                            SENTIMENT_MODELS[new_sentiment_model],
                            EMOTION_MODELS[new_emotion_model],
                            new_analysis_name
                        )
                        
                        # Update session state
                        st.session_state.df_analyzed = df_new_analyzed
                        st.session_state.current_analysis_id = analysis_id

                    st.success(f"New analysis complete! Analysis ID: {analysis_id}")
                    st.info(f"New analysis saved as: {metadata['analysis_name']}")
                    st.rerun()

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    st.error("Please check your internet connection and try again.")
    
    # Create tabs for different analysis types
    tab1, tab2, tab3, tab4 = st.tabs([
        "Basic Analysis", 
        "Context-Aware Analysis", 
        "Trend Analysis",
        "Emotion Intensity"
    ])
    
    with tab1:
        # Display Basic Visualizations
        st.header("Overall Script Emotion")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment Distribution")
            fig_sentiment = plot_overall_sentiment_distribution(df_analyzed)
            st.pyplot(fig_sentiment)
        with col2:
            st.subheader("Emotion Distribution")
            fig_emotion = plot_overall_emotion_distribution(df_analyzed)
            st.pyplot(fig_emotion)
            
        st.header("Sentiment Timeline")
        st.plotly_chart(plot_sentiment_timeline_by_scene(df_analyzed), use_container_width=True)

        # --- Character-Specific Analysis ---
        st.header("Character-Specific Sentiment Analysis")
        
        # Create a list of characters with their line counts for the selectbox
        character_counts = df_analyzed['character'].value_counts().reset_index()
        character_counts.columns = ['character', 'count']
        character_counts['display_name'] = character_counts['character'] + " (" + character_counts['count'].astype(str) + " lines)"
        
        selected_char = st.selectbox(
            "Select a character to see their sentiment timeline:",
            options=character_counts['display_name']
        )

        if selected_char:
            # Find the original character name from the display name
            original_char_name = character_counts[character_counts['display_name'] == selected_char]['character'].iloc[0]
            st.plotly_chart(plot_sentiment_timeline_for_character(df_analyzed, original_char_name), use_container_width=True)
    
    with tab2:
        st.header("Context-Aware Sentiment Analysis")
        st.markdown("""
        This analysis considers the context of surrounding scenes and dialogues to provide 
        more nuanced sentiment understanding, including context shifts and sentiment deviations.
        """)
        
        # Context window size selection
        window_size = st.slider("Context Window Size", min_value=1, max_value=10, value=3, 
                               help="Number of surrounding dialogues to consider for context")
        
        fig_context = plot_context_aware_sentiment(df_analyzed, window_size)
        st.pyplot(fig_context)
        
        # Show context metrics
        context_result = analyze_context_aware_sentiment(df_analyzed, window_size)
        if context_result:
            df_context, scene_context = context_result
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Context Shifts", scene_context['context_shifts'].sum())
            with col2:
                st.metric("Avg Sentiment Deviation", f"{scene_context['avg_deviation'].mean():.3f}")
            with col3:
                st.metric("Sentiment Volatility", f"{scene_context['sentiment_volatility'].mean():.3f}")
            with col4:
                st.metric("Context Shifts per Scene", f"{scene_context['context_shifts'].mean():.1f}")
    
    with tab3:
        st.header("Sentiment Trend Analysis")
        st.markdown("""
        Trend analysis examines how sentiment evolves over time, identifying improvement, 
        decline, and stable phases in character development.
        """)
        
        # Overall trend analysis
        st.subheader("Overall Script Trend")
        fig_overall_trend = plot_sentiment_trend_analysis(df_analyzed)
        st.pyplot(fig_overall_trend)
        
        # Character-specific trend analysis
        st.subheader("Character-Specific Trend Analysis")
        character_counts = df_analyzed['character'].value_counts().reset_index()
        character_counts.columns = ['character', 'count']
        character_counts['display_name'] = character_counts['character'] + " (" + character_counts['count'].astype(str) + " lines)"
        
        selected_char_trend = st.selectbox(
            "Select a character for trend analysis:",
            options=character_counts['display_name'],
            key="trend_character"
        )

        if selected_char_trend:
            original_char_name = character_counts[character_counts['display_name'] == selected_char_trend]['character'].iloc[0]
            fig_char_trend = plot_sentiment_trend_analysis(df_analyzed, original_char_name)
            st.pyplot(fig_char_trend)
            
            # Show trend metrics
            trend_result = analyze_sentiment_trends(df_analyzed, original_char_name)
            if trend_result:
                df_trend, development_metrics = trend_result
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Dialogues", development_metrics['total_dialogues'])
                with col2:
                    st.metric("Overall Trend", f"{development_metrics['overall_trend']:.3f}")
                with col3:
                    st.metric("Trend Consistency", f"{development_metrics['trend_consistency']:.3f}")
                with col4:
                    if 'improvement_phases' in development_metrics:
                        st.metric("Improvement Phases", development_metrics['improvement_phases'])
    
    with tab4:
        st.header("Emotion Intensity Analysis")
        st.markdown("""
        Emotion intensity analysis measures the strength and impact of emotional expressions, 
        considering dialogue length, sentiment context, and emotion confidence scores.
        """)
        
        # Overall emotion intensity
        st.subheader("Overall Emotion Intensity")
        fig_intensity = plot_emotion_intensity_analysis(df_analyzed)
        st.pyplot(fig_intensity)
        
        # Character-specific emotion intensity
        st.subheader("Character-Specific Emotion Intensity")
        character_counts = df_analyzed['character'].value_counts().reset_index()
        character_counts.columns = ['character', 'count']
        character_counts['display_name'] = character_counts['character'] + " (" + character_counts['count'].astype(str) + " lines)"
        
        selected_char_intensity = st.selectbox(
            "Select a character for intensity analysis:",
            options=character_counts['display_name'],
            key="intensity_character"
        )

        if selected_char_intensity:
            original_char_name = character_counts[character_counts['display_name'] == selected_char_intensity]['character'].iloc[0]
            fig_char_intensity = plot_character_emotion_intensity(df_analyzed, original_char_name)
            st.pyplot(fig_char_intensity)
            
            # Show intensity metrics
            df_char = df_analyzed[df_analyzed['character'] == original_char_name]
            df_intensity = calculate_emotion_intensity(df_char)
            if df_intensity is not None:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Intensity", f"{df_intensity['normalized_intensity'].mean():.3f}")
                with col2:
                    st.metric("Max Intensity", f"{df_intensity['normalized_intensity'].max():.3f}")
                with col3:
                    st.metric("Intensity Variability", f"{df_intensity['normalized_intensity'].std():.3f}")
                with col4:
                    high_intensity_count = len(df_intensity[df_intensity['intensity_level'] == 'High'])
                    st.metric("High Intensity Moments", high_intensity_count)

    # --- Download Button ---
    st.header("Download Analysis")
    st.download_button(
        label="Download Current Analysis as CSV",
        data=convert_df_to_csv(df_analyzed),
        file_name=f"analysis_{st.session_state.current_analysis_id[:8]}.csv",
        mime="text/csv",
    )

else:
    st.warning("Please upload a script on the main page to begin the analysis.") 