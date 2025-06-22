import streamlit as st
import pandas as pd
from utils.analyze import (
    plot_character_arc,
    plot_character_personality_profile,
    plot_character_interaction_quality,
    plot_protagonist_antagonist_analysis,
    create_character_personality_profile,
    analyze_character_interaction_quality,
    identify_protagonist_antagonist
)

st.set_page_config(page_title="Character Analysis", layout="wide")

st.title("Character Analysis")
st.markdown("""
This page provides comprehensive character analysis including character arcs, personality profiles, 
interaction quality, and protagonist vs antagonist identification.
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
    
    # Get list of characters
    characters = df_script['character'].unique()
    
    st.header("Character Selection")
    selected_character = st.selectbox(
        "Choose a character to analyze:",
        options=characters,
        index=0 if len(characters) > 0 else None
    )
    
    if selected_character:
        st.subheader(f"Analysis for: {selected_character}")
        
        # Create tabs for different analysis types
        tab1, tab2, tab3, tab4 = st.tabs([
            "Character Arc", 
            "Personality Profile", 
            "Interaction Quality",
            "Protagonist vs Antagonist"
        ])
        
        with tab1:
            st.subheader("Character Arc Analysis")
            st.markdown("""
            Character arc analysis tracks how a character evolves throughout the story, 
            including dialogue patterns, sentiment trends, and development across different acts.
            """)
            
            fig_arc = plot_character_arc(df_script, selected_character)
            st.pyplot(fig_arc)
            
            # Show character arc metrics
            from utils.analyze import analyze_character_arc
            df_char, arc_analysis, sentiment_trend = analyze_character_arc(df_script, selected_character)
            
            if arc_analysis:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Dialogues", arc_analysis['total_dialogues'])
                with col2:
                    st.metric("Total Scenes", arc_analysis['total_scenes'])
                with col3:
                    st.metric("Avg Dialogue Length", f"{arc_analysis['avg_dialogue_length']:.1f} words")
                with col4:
                    if 'act1_dialogues' in arc_analysis:
                        st.metric("Act Distribution", f"{arc_analysis['act1_dialogues']}/{arc_analysis['act2_dialogues']}/{arc_analysis['act3_dialogues']}")
        
        with tab2:
            st.subheader("Personality Profile")
            st.markdown("""
            Personality profile analysis examines character traits based on dialogue patterns, 
            sentiment distribution, emotional states, and communication style.
            """)
            
            fig_profile = plot_character_personality_profile(df_script, selected_character)
            st.pyplot(fig_profile)
            
            # Show personality metrics
            profile = create_character_personality_profile(df_script, selected_character)
            if profile:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Dominance Score", f"{profile['dominance_score']:.1f}%")
                with col2:
                    st.metric("Scene Presence", f"{profile['scene_presence']:.2f}")
                with col3:
                    st.metric("Dialogue Variability", f"{profile['dialogue_variability']:.1f}")
                with col4:
                    if profile.get('emotional_stability') is not None:
                        st.metric("Emotional Stability", f"{profile['emotional_stability']:.2f}")
                
                # Show primary emotion if available
                if profile.get('primary_emotion'):
                    st.info(f"Primary Emotion: {profile['primary_emotion']}")
        
        with tab3:
            st.subheader("Interaction Quality Analysis")
            st.markdown("""
            Interaction quality analysis examines how the character interacts with others, 
            including dialogue balance, interaction frequency, and relationship dynamics.
            """)
            
            fig_interaction = plot_character_interaction_quality(df_script, selected_character)
            st.pyplot(fig_interaction)
            
            # Show interaction metrics
            result = analyze_character_interaction_quality(df_script, selected_character)
            if result:
                df_interactions, quality_metrics = result
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Interactions", quality_metrics['total_interactions'])
                with col2:
                    st.metric("Unique Characters", quality_metrics['unique_characters'])
                with col3:
                    st.metric("Avg Dialogue Balance", f"{quality_metrics['avg_dialogue_balance']:.2f}")
                with col4:
                    st.metric("Interaction Diversity", f"{quality_metrics['interaction_diversity']:.1%}")
                
                # Show most interactive character
                if quality_metrics.get('most_interactive_character'):
                    st.info(f"Most Interactive Character: {quality_metrics['most_interactive_character']}")
        
        with tab4:
            st.subheader("Protagonist vs Antagonist Analysis")
            st.markdown("""
            This analysis identifies the protagonist and antagonist based on dialogue patterns, 
            sentiment analysis, screen time, and character consistency.
            """)
            
            fig_prot_ant = plot_protagonist_antagonist_analysis(df_script)
            st.pyplot(fig_prot_ant)
            
            # Show protagonist and antagonist identification
            protagonist, antagonist = identify_protagonist_antagonist(df_script)
            if protagonist is not None and antagonist is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Protagonist:** {protagonist['character']}")
                    st.metric("Protagonist Score", f"{protagonist['protagonist_score']:.2f}")
                with col2:
                    st.error(f"**Antagonist:** {antagonist['character']}")
                    st.metric("Antagonist Score", f"{antagonist['antagonist_score']:.2f}")
                
                # Show comparison metrics
                st.subheader("Character Role Comparison")
                comparison_data = {
                    'Metric': ['Total Dialogues', 'Scene Presence', 'Avg Sentiment', 'Character Consistency'],
                    'Protagonist': [
                        protagonist['total_dialogues'],
                        protagonist['scene_presence'],
                        protagonist['avg_sentiment'],
                        protagonist['sentiment_consistency']
                    ],
                    'Antagonist': [
                        antagonist['total_dialogues'],
                        antagonist['scene_presence'],
                        antagonist['avg_sentiment'],
                        antagonist['sentiment_consistency']
                    ]
                }
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
    
    else:
        st.warning("No characters found in the script.")
        
else:
    st.warning("Please upload a script on the main page to begin the analysis.") 