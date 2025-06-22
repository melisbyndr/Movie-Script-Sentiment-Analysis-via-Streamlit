import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import json
import hashlib
from datetime import datetime

# --- Configuration ---
CACHE_DIR = "analysis_cache"

def ensure_cache_dir():
    """Creates the cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)

# A dictionary to hold our model choices
SENTIMENT_MODELS = {
    "DistilBERT (Default)": "distilbert-base-uncased-finetuned-sst-2-english",
    "BERT Large": "nlptown/bert-base-multilingual-uncased-sentiment",
    "RoBERTa Large": "siebert/sentiment-roberta-large-english",
}

EMOTION_MODELS = {
    "DistilRoBERTa (Default)": "j-hartmann/emotion-english-distilroberta-base",
    "RoBERTa GoEmotions": "SamLowe/roberta-base-go_emotions",
    # Add more models here if needed
}

@st.cache_resource
def get_pipeline(task, model):
    """
    Loads and caches a Hugging Face pipeline.
    The _resource decorator is used because pipelines are complex objects.
    """
    st.info(f"Loading model: {model}. This may take a moment...")
    return pipeline(task, model=model)

def analyze_text(text_list, task, model_name):
    """
    Analyzes a list of texts for sentiment or emotion using a specified model.
    
    Args:
        text_list (list): A list of strings to analyze.
        task (str): The pipeline task (e.g., "sentiment-analysis", "text-classification").
        model_name (str): The name of the model from the Hugging Face hub.
        
    Returns:
        list: A list of analysis results from the pipeline.
    """
    if not text_list:
        return []
    
    # Select the correct model from the full name
    if task == "sentiment-analysis":
        model_path = SENTIMENT_MODELS[model_name]
    elif task == "text-classification":
         model_path = EMOTION_MODELS[model_name]
    else:
        raise ValueError("Unsupported task. Choose 'sentiment-analysis' or 'text-classification'.")

    # Get the pipeline (it will be cached after the first run)
    analyzer = get_pipeline(task, model=model_path)
    
    # Run the analysis
    # Ensure all items in the list are strings and handle potential None values
    clean_text_list = [str(text) for text in text_list if text]
    
    if not clean_text_list:
        return []

    results = analyzer(clean_text_list)
    return results

def merge_results_with_df(df, sentiment_results, emotion_results):
    """
    Merges the sentiment and emotion analysis results back into the original dataframe.
    """
    # Create dataframes from the analysis results
    df_sentiment = pd.DataFrame(sentiment_results)
    df_emotion = pd.DataFrame(emotion_results)

    # Rename columns for clarity
    df_sentiment = df_sentiment.rename(columns={'label': 'sentiment_label', 'score': 'sentiment_score'})
    df_emotion = df_emotion.rename(columns={'label': 'emotion_label', 'score': 'emotion_score'})

    # Combine them with the original dataframe that contains dialogues
    # This assumes the results are in the same order as the dialogues passed to the analyzer
    df_dialogues = df[df['clean_dialogue'].notna()].copy()
    df_dialogues = df_dialogues.reset_index()

    df_analyzed = pd.concat([df_dialogues, df_sentiment, df_emotion], axis=1)
    
    # Merge back into the original full dataframe, keeping all rows
    df_final = pd.merge(df.reset_index(), df_analyzed[['index', 'sentiment_label', 'sentiment_score', 'emotion_label', 'emotion_score']], on='index', how='left')
    df_final = df_final.set_index('index')

    return df_final

def plot_overall_sentiment_distribution(df):
    """
    Plots a pie chart of the overall sentiment distribution.
    """
    sentiment_counts = df['sentiment_label'].value_counts()
    
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90,
           colors=sns.color_palette("coolwarm", len(sentiment_counts)))
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title("Overall Sentiment Distribution")
    return fig

def plot_overall_emotion_distribution(df, top_n=10):
    """
    Plots a bar chart of the overall emotion distribution for the top N emotions.
    """
    emotion_counts = df['emotion_label'].value_counts().nlargest(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=emotion_counts.values, y=emotion_counts.index, ax=ax, palette="viridis")
    ax.set_title(f"Top {top_n} Emotions Across the Script")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Emotion")
    plt.tight_layout()
    return fig

def map_sentiment_to_score(df):
    """Converts sentiment labels to numerical scores."""
    # Simple mapping: POSITIVE -> 1, NEGATIVE -> -1, NEUTRAL or other -> 0
    # The exact labels might vary by model, so this needs to be robust.
    df['sentiment_numeric'] = df['sentiment_label'].apply(
        lambda x: 1 if 'pos' in str(x).lower() else (-1 if 'neg' in str(x).lower() else 0)
    )
    return df

def plot_sentiment_timeline_by_scene(df):
    """
    Plots an interactive timeline of average sentiment score per 10 scenes.
    """
    df_with_score = map_sentiment_to_score(df)
    
    # Create bins of 10 scenes
    df_with_score['scene_bin'] = (df_with_score['scene_id'] // 10) * 10
    
    # Calculate average sentiment per bin
    scene_sentiment = df_with_score.groupby('scene_bin')['sentiment_numeric'].mean().reset_index()
    
    fig = px.line(
        scene_sentiment,
        x='scene_bin',
        y='sentiment_numeric',
        title="Average Sentiment Trend Across Scenes (in 10-scene Bins)",
        labels={'scene_bin': 'Scene (Start of Bin)', 'sentiment_numeric': 'Average Sentiment Score'},
        markers=True
    )
    fig.update_layout(
        xaxis_title="Scene Bins",
        yaxis_title="Average Sentiment (1=Positive, -1=Negative)",
        hovermode="x unified"
    )
    return fig

def plot_sentiment_timeline_for_character(df, character_name):
    """
    Plots an interactive timeline of a specific character's sentiment evolution with a range slider.
    """
    df_char = df[df['character'] == character_name].copy()
    if df_char.empty:
        return go.Figure().update_layout(title_text=f"No data for character: {character_name}")
        
    df_char = map_sentiment_to_score(df_char)

    # Use a rolling average to smooth the timeline
    df_char['sentiment_smoothed'] = df_char['sentiment_numeric'].rolling(window=5, min_periods=1, center=True).mean()

    fig = px.line(
        df_char,
        x='scene_id',
        y='sentiment_smoothed',
        title=f"Sentiment Timeline for {character_name}",
        labels={'scene_id': 'Scene ID', 'sentiment_smoothed': 'Sentiment Score (Smoothed)'},
        hover_data=['clean_dialogue', 'sentiment_label'],
        markers=True
    )
    fig.update_layout(
        xaxis_title="Scene ID",
        yaxis_title="Sentiment (1=Positive, -1=Negative)",
        hovermode="x unified",
        xaxis_rangeslider_visible=True
    )
    return fig

def analyze_context_aware_sentiment(df_script, window_size=3):
    """
    Analyzes sentiment considering the context of surrounding scenes and dialogues.
    """
    if 'sentiment_score' not in df_script.columns or df_script.empty:
        return None
    
    df_context = df_script.copy()
    
    # Sort by scene_id to maintain chronological order
    df_context = df_context.sort_values('scene_id')
    
    # Calculate context-aware sentiment using rolling window
    df_context['context_sentiment'] = df_context['sentiment_score'].rolling(
        window=window_size, min_periods=1, center=True
    ).mean()
    
    # Calculate sentiment deviation from context
    df_context['sentiment_deviation'] = df_context['sentiment_score'] - df_context['context_sentiment']
    
    # Identify context shifts (significant changes in context sentiment)
    df_context['context_shift'] = df_context['context_sentiment'].diff().abs() > 0.2
    
    # Calculate scene-level context metrics
    scene_context = df_context.groupby('scene_id').agg({
        'sentiment_score': ['mean', 'std'],
        'context_sentiment': 'mean',
        'sentiment_deviation': ['mean', 'std'],
        'context_shift': 'sum'
    }).reset_index()
    
    # Flatten column names
    scene_context.columns = [
        'scene_id', 'avg_sentiment', 'sentiment_volatility', 
        'context_sentiment', 'avg_deviation', 'deviation_volatility', 'context_shifts'
    ]
    
    return df_context, scene_context

def plot_context_aware_sentiment(df_script, window_size=3):
    """
    Visualizes context-aware sentiment analysis.
    """
    result = analyze_context_aware_sentiment(df_script, window_size)
    
    if result is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sentiment data not available for context analysis", ha='center', va='center')
        return fig
    
    df_context, scene_context = result
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Context-Aware Sentiment Analysis', fontsize=16)
    
    # Plot 1: Original vs Context sentiment
    axes[0, 0].scatter(df_context['scene_id'], df_context['sentiment_score'], 
                      alpha=0.6, label='Original Sentiment', s=20)
    axes[0, 0].plot(df_context['scene_id'], df_context['context_sentiment'], 
                   color='red', linewidth=2, label='Context Sentiment')
    axes[0, 0].set_title('Original vs Context-Aware Sentiment')
    axes[0, 0].set_xlabel('Scene ID')
    axes[0, 0].set_ylabel('Sentiment Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Sentiment deviation distribution
    axes[0, 1].hist(df_context['sentiment_deviation'], bins=20, alpha=0.7, color='skyblue')
    axes[0, 1].set_title('Sentiment Deviation from Context')
    axes[0, 1].set_xlabel('Sentiment Deviation')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Plot 3: Context shifts over time
    context_shifts = df_context[df_context['context_shift'] == True]
    if not context_shifts.empty:
        axes[1, 0].scatter(context_shifts['scene_id'], context_shifts['sentiment_score'], 
                          color='red', s=50, alpha=0.8, label='Context Shifts')
    axes[1, 0].plot(df_context['scene_id'], df_context['context_sentiment'], 
                   color='blue', linewidth=2, label='Context Sentiment')
    axes[1, 0].set_title('Context Shifts Detection')
    axes[1, 0].set_xlabel('Scene ID')
    axes[1, 0].set_ylabel('Sentiment Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Scene-level context metrics
    axes[1, 1].scatter(scene_context['avg_sentiment'], scene_context['sentiment_volatility'], 
                      s=scene_context['context_shifts']*20, alpha=0.7, c=scene_context['avg_deviation'], 
                      cmap='coolwarm')
    axes[1, 1].set_title('Scene Sentiment Characteristics')
    axes[1, 1].set_xlabel('Average Sentiment')
    axes[1, 1].set_ylabel('Sentiment Volatility')
    plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Avg Deviation')
    
    plt.tight_layout()
    return fig

def analyze_sentiment_trends(df_script, character_name=None):
    """
    Analyzes sentiment trends and character development over time.
    """
    if 'sentiment_score' not in df_script.columns or df_script.empty:
        return None
    
    if character_name:
        df_trend = df_script[df_script['character'] == character_name].copy()
    else:
        df_trend = df_script.copy()
    
    if df_trend.empty:
        return None
    
    # Sort by scene_id
    df_trend = df_trend.sort_values('scene_id')
    
    # Calculate trend metrics
    df_trend['sentiment_numeric'] = df_trend['sentiment_label'].apply(
        lambda x: 1 if 'pos' in str(x).lower() else (-1 if 'neg' in str(x).lower() else 0)
    )
    
    # Rolling averages for trend analysis
    df_trend['sentiment_trend_5'] = df_trend['sentiment_numeric'].rolling(window=5, min_periods=1).mean()
    df_trend['sentiment_trend_10'] = df_trend['sentiment_numeric'].rolling(window=10, min_periods=1).mean()
    
    # Calculate trend direction and strength
    df_trend['trend_direction'] = df_trend['sentiment_trend_5'].diff()
    df_trend['trend_strength'] = df_trend['trend_direction'].abs()
    
    # Identify trend phases
    df_trend['trend_phase'] = 'stable'
    df_trend.loc[df_trend['trend_direction'] > 0.1, 'trend_phase'] = 'improving'
    df_trend.loc[df_trend['trend_direction'] < -0.1, 'trend_phase'] = 'declining'
    
    # Calculate development metrics
    total_dialogues = len(df_trend)
    if total_dialogues >= 3:
        # Divide into development phases
        phase1_end = total_dialogues // 3
        phase2_end = 2 * total_dialogues // 3
        
        phase1 = df_trend.iloc[:phase1_end]
        phase2 = df_trend.iloc[phase1_end:phase2_end]
        phase3 = df_trend.iloc[phase2_end:]
        
        development_metrics = {
            'total_dialogues': total_dialogues,
            'phase1_avg_sentiment': phase1['sentiment_numeric'].mean(),
            'phase2_avg_sentiment': phase2['sentiment_numeric'].mean(),
            'phase3_avg_sentiment': phase3['sentiment_numeric'].mean(),
            'overall_trend': phase3['sentiment_numeric'].mean() - phase1['sentiment_numeric'].mean(),
            'trend_consistency': 1 - df_trend['trend_strength'].std(),
            'improvement_phases': len(df_trend[df_trend['trend_phase'] == 'improving']),
            'decline_phases': len(df_trend[df_trend['trend_phase'] == 'declining'])
        }
    else:
        development_metrics = {
            'total_dialogues': total_dialogues,
            'overall_trend': 0,
            'trend_consistency': 0
        }
    
    return df_trend, development_metrics

def plot_sentiment_trend_analysis(df_script, character_name=None):
    """
    Visualizes sentiment trend analysis and character development.
    """
    result = analyze_sentiment_trends(df_script, character_name)
    
    if result is None:
        fig, ax = plt.subplots()
        title = f"Sentiment trend data not available for {character_name}" if character_name else "Sentiment trend data not available"
        ax.text(0.5, 0.5, title, ha='center', va='center')
        return fig
    
    df_trend, development_metrics = result
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    title = f'Sentiment Trend Analysis: {character_name}' if character_name else 'Overall Sentiment Trend Analysis'
    fig.suptitle(title, fontsize=16)
    
    # Plot 1: Sentiment trend over time
    axes[0, 0].plot(df_trend['scene_id'], df_trend['sentiment_numeric'], 
                   alpha=0.6, label='Individual Sentiment', marker='o', markersize=3)
    axes[0, 0].plot(df_trend['scene_id'], df_trend['sentiment_trend_5'], 
                   color='red', linewidth=2, label='5-Dialogue Trend')
    axes[0, 0].plot(df_trend['scene_id'], df_trend['sentiment_trend_10'], 
                   color='blue', linewidth=2, label='10-Dialogue Trend')
    axes[0, 0].set_title('Sentiment Trend Over Time')
    axes[0, 0].set_xlabel('Scene ID')
    axes[0, 0].set_ylabel('Sentiment Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Trend phase distribution
    phase_counts = df_trend['trend_phase'].value_counts()
    axes[0, 1].pie(phase_counts.values, labels=phase_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Trend Phase Distribution')
    
    # Plot 3: Development phases comparison
    if 'phase1_avg_sentiment' in development_metrics:
        phases = ['Phase 1', 'Phase 2', 'Phase 3']
        phase_sentiments = [
            development_metrics['phase1_avg_sentiment'],
            development_metrics['phase2_avg_sentiment'],
            development_metrics['phase3_avg_sentiment']
        ]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        axes[1, 0].bar(phases, phase_sentiments, color=colors)
        axes[1, 0].set_title('Sentiment by Development Phase')
        axes[1, 0].set_ylabel('Average Sentiment')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    else:
        axes[1, 0].text(0.5, 0.5, 'Insufficient data for phase analysis', ha='center', va='center')
        axes[1, 0].set_title('Sentiment by Development Phase')
    
    # Plot 4: Trend strength distribution
    axes[1, 1].hist(df_trend['trend_strength'], bins=15, alpha=0.7, color='green')
    axes[1, 1].set_title('Trend Strength Distribution')
    axes[1, 1].set_xlabel('Trend Strength')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig

def calculate_emotion_intensity(df_script):
    """
    Calculates emotion intensity scores based on emotion confidence and context.
    """
    if 'emotion_score' not in df_script.columns or df_script.empty:
        return None
    
    df_intensity = df_script.copy()
    
    # Base intensity from emotion score
    df_intensity['base_intensity'] = df_intensity['emotion_score']
    
    # Context intensity (if sentiment data available)
    if 'sentiment_score' in df_intensity.columns:
        # High emotion + extreme sentiment = higher intensity
        sentiment_abs = df_intensity['sentiment_score'].abs()
        df_intensity['context_intensity'] = sentiment_abs * df_intensity['emotion_score']
    else:
        df_intensity['context_intensity'] = df_intensity['emotion_score']
    
    # Dialogue length intensity (longer dialogues might indicate stronger emotions)
    df_intensity['dialogue_length'] = df_intensity['clean_dialogue'].str.split().str.len()
    df_intensity['length_intensity'] = np.log1p(df_intensity['dialogue_length']) * 0.1
    
    # Calculate overall intensity score
    df_intensity['overall_intensity'] = (
        df_intensity['base_intensity'] * 0.5 +
        df_intensity['context_intensity'] * 0.3 +
        df_intensity['length_intensity'] * 0.2
    )
    
    # Normalize intensity to 0-1 scale
    df_intensity['normalized_intensity'] = (
        df_intensity['overall_intensity'] - df_intensity['overall_intensity'].min()
    ) / (df_intensity['overall_intensity'].max() - df_intensity['overall_intensity'].min())
    
    # Categorize intensity levels
    df_intensity['intensity_level'] = pd.cut(
        df_intensity['normalized_intensity'],
        bins=[0, 0.33, 0.66, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    return df_intensity

def plot_emotion_intensity_analysis(df_script):
    """
    Visualizes emotion intensity analysis.
    """
    df_intensity = calculate_emotion_intensity(df_script)
    
    if df_intensity is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Emotion data not available for intensity analysis", ha='center', va='center')
        return fig
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Emotion Intensity Analysis', fontsize=16)
    
    # Plot 1: Intensity distribution
    axes[0, 0].hist(df_intensity['normalized_intensity'], bins=20, alpha=0.7, color='purple')
    axes[0, 0].set_title('Emotion Intensity Distribution')
    axes[0, 0].set_xlabel('Normalized Intensity')
    axes[0, 0].set_ylabel('Frequency')
    
    # Plot 2: Intensity by emotion type
    emotion_intensity = df_intensity.groupby('emotion_label')['normalized_intensity'].mean().sort_values(ascending=True)
    axes[0, 1].barh(range(len(emotion_intensity)), emotion_intensity.values, color='orange')
    axes[0, 1].set_yticks(range(len(emotion_intensity)))
    axes[0, 1].set_yticklabels(emotion_intensity.index)
    axes[0, 1].set_title('Average Intensity by Emotion Type')
    axes[0, 1].set_xlabel('Average Intensity')
    
    # Plot 3: Intensity timeline
    scene_intensity = df_intensity.groupby('scene_id')['normalized_intensity'].mean()
    axes[1, 0].plot(scene_intensity.index, scene_intensity.values, marker='o', linewidth=2, color='red')
    axes[1, 0].set_title('Emotion Intensity Timeline')
    axes[1, 0].set_xlabel('Scene ID')
    axes[1, 0].set_ylabel('Average Intensity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Intensity level distribution
    intensity_counts = df_intensity['intensity_level'].value_counts()
    axes[1, 1].pie(intensity_counts.values, labels=intensity_counts.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Intensity Level Distribution')
    
    plt.tight_layout()
    return fig

def plot_character_emotion_intensity(df_script, character_name):
    """
    Visualizes emotion intensity for a specific character.
    """
    df_char = df_script[df_script['character'] == character_name].copy()
    
    if df_char.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No data available for character: {character_name}", ha='center', va='center')
        return fig
    
    df_intensity = calculate_emotion_intensity(df_char)
    
    if df_intensity is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Emotion data not available for {character_name}", ha='center', va='center')
        return fig
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Emotion Intensity Analysis: {character_name}', fontsize=16)
    
    # Plot 1: Character's emotion intensity over time
    df_intensity = df_intensity.sort_values('scene_id')
    axes[0, 0].scatter(df_intensity['scene_id'], df_intensity['normalized_intensity'], 
                      c=df_intensity['emotion_score'], cmap='viridis', s=50, alpha=0.7)
    axes[0, 0].set_title('Emotion Intensity Timeline')
    axes[0, 0].set_xlabel('Scene ID')
    axes[0, 0].set_ylabel('Normalized Intensity')
    plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0], label='Emotion Score')
    
    # Plot 2: Character's emotion distribution
    emotion_counts = df_intensity['emotion_label'].value_counts()
    axes[0, 1].pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Emotion Distribution')
    
    # Plot 3: Intensity vs dialogue length
    axes[1, 0].scatter(df_intensity['dialogue_length'], df_intensity['normalized_intensity'], 
                      alpha=0.6, s=30)
    axes[1, 0].set_title('Intensity vs Dialogue Length')
    axes[1, 0].set_xlabel('Dialogue Length (words)')
    axes[1, 0].set_ylabel('Normalized Intensity')
    
    # Plot 4: Top intense moments
    top_intense = df_intensity.nlargest(5, 'normalized_intensity')
    axes[1, 1].barh(range(len(top_intense)), top_intense['normalized_intensity'], 
                   color='red', alpha=0.7)
    axes[1, 1].set_yticks(range(len(top_intense)))
    axes[1, 1].set_yticklabels([f"Scene {s}" for s in top_intense['scene_id']])
    axes[1, 1].set_title('Top 5 Most Intense Moments')
    axes[1, 1].set_xlabel('Intensity Score')
    
    plt.tight_layout()
    return fig

def save_model_analysis(df, file_hash, original_filename, sentiment_model, emotion_model, analysis_name=None):
    """
    Saves the analyzed dataframe with model information and creates a unique identifier.
    """
    # Ensure cache directory exists
    ensure_cache_dir()
    
    # Create analysis metadata
    analysis_id = hashlib.md5(f"{file_hash}_{sentiment_model}_{emotion_model}_{datetime.now().isoformat()}".encode()).hexdigest()
    
    if analysis_name is None:
        analysis_name = f"Analysis_{sentiment_model.split('/')[-1]}_{emotion_model.split('/')[-1]}"
    
    analysis_metadata = {
        'analysis_id': analysis_id,
        'file_hash': file_hash,
        'original_filename': original_filename,
        'sentiment_model': sentiment_model,
        'emotion_model': emotion_model,
        'analysis_name': analysis_name,
        'created_at': datetime.now().isoformat(),
        'total_dialogues': len(df),
        'total_characters': df['character'].nunique(),
        'total_scenes': df['scene_id'].nunique()
    }
    
    # Save the dataframe
    df.to_parquet(os.path.join(CACHE_DIR, f"{analysis_id}.parquet"))
    
    # Update metadata file
    metadata_file = os.path.join(CACHE_DIR, "model_analyses.json")
    
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            all_analyses = json.load(f)
    else:
        all_analyses = {}
    
    all_analyses[analysis_id] = analysis_metadata
    
    with open(metadata_file, 'w') as f:
        json.dump(all_analyses, f, indent=4)
    
    return analysis_id, analysis_metadata

def load_model_analyses():
    """
    Loads all saved model analyses metadata.
    """
    # Ensure cache directory exists
    ensure_cache_dir()
    
    metadata_file = os.path.join(CACHE_DIR, "model_analyses.json")
    
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    else:
        return {}

def load_analysis_by_id(analysis_id):
    """
    Loads a specific analysis by its ID.
    """
    # Ensure cache directory exists
    ensure_cache_dir()
    
    analysis_file = os.path.join(CACHE_DIR, f"{analysis_id}.parquet")
    
    if os.path.exists(analysis_file):
        try:
            df = pd.read_parquet(analysis_file)
            return df
        except Exception as e:
            print(f"Error loading analysis {analysis_id}: {e}")
            return None
    else:
        print(f"Analysis file not found: {analysis_file}")
        return None

def get_analysis_metadata(analysis_id):
    """
    Gets metadata for a specific analysis.
    """
    all_analyses = load_model_analyses()
    return all_analyses.get(analysis_id)

def compare_model_results(analysis_ids):
    """
    Compares results from multiple model analyses.
    """
    if len(analysis_ids) < 2:
        return None
    
    comparison_data = []
    
    for analysis_id in analysis_ids:
        df = load_analysis_by_id(analysis_id)
        metadata = get_analysis_metadata(analysis_id)
        
        if df is not None and metadata is not None:
            # Calculate basic metrics
            sentiment_dist = df['sentiment_label'].value_counts(normalize=True)
            emotion_dist = df['emotion_label'].value_counts(normalize=True)
            
            comparison_data.append({
                'analysis_id': analysis_id,
                'analysis_name': metadata['analysis_name'],
                'sentiment_model': metadata['sentiment_model'],
                'emotion_model': metadata['emotion_model'],
                'total_dialogues': len(df),
                'avg_sentiment_score': df['sentiment_score'].mean() if 'sentiment_score' in df.columns else 0,
                'avg_emotion_score': df['emotion_score'].mean() if 'emotion_score' in df.columns else 0,
                'sentiment_positive_pct': sentiment_dist.get('POSITIVE', 0) if 'POSITIVE' in sentiment_dist.index else 0,
                'sentiment_negative_pct': sentiment_dist.get('NEGATIVE', 0) if 'NEGATIVE' in sentiment_dist.index else 0,
                'top_emotion': emotion_dist.index[0] if len(emotion_dist) > 0 else 'Unknown',
                'top_emotion_pct': emotion_dist.iloc[0] if len(emotion_dist) > 0 else 0,
                'created_at': metadata['created_at']
            })
    
    return pd.DataFrame(comparison_data)

def plot_model_comparison(comparison_df):
    """
    Creates comparison plots for multiple model results.
    """
    if comparison_df is None or len(comparison_df) < 2:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Comparison Results', fontsize=16)
    
    # Plot 1: Average sentiment scores
    axes[0, 0].bar(comparison_df['analysis_name'], comparison_df['avg_sentiment_score'], color='skyblue')
    axes[0, 0].set_title('Average Sentiment Scores')
    axes[0, 0].set_ylabel('Average Sentiment Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Average emotion scores
    axes[0, 1].bar(comparison_df['analysis_name'], comparison_df['avg_emotion_score'], color='lightcoral')
    axes[0, 1].set_title('Average Emotion Confidence Scores')
    axes[0, 1].set_ylabel('Average Emotion Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Sentiment distribution comparison
    x = np.arange(len(comparison_df))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, comparison_df['sentiment_positive_pct'], width, label='Positive', color='green')
    axes[1, 0].bar(x + width/2, comparison_df['sentiment_negative_pct'], width, label='Negative', color='red')
    axes[1, 0].set_title('Sentiment Distribution Comparison')
    axes[1, 0].set_ylabel('Percentage')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(comparison_df['analysis_name'], rotation=45)
    axes[1, 0].legend()
    
    # Plot 4: Top emotion comparison
    top_emotions = comparison_df['top_emotion'].value_counts()
    axes[1, 1].pie(top_emotions.values, labels=top_emotions.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Most Common Top Emotions Across Models')
    
    plt.tight_layout()
    return fig

def delete_analysis(analysis_id):
    """
    Deletes a specific analysis and its metadata.
    """
    # Delete the parquet file
    analysis_file = os.path.join(CACHE_DIR, f"{analysis_id}.parquet")
    if os.path.exists(analysis_file):
        os.remove(analysis_file)
    
    # Remove from metadata
    metadata_file = os.path.join(CACHE_DIR, "model_analyses.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            all_analyses = json.load(f)
        
        if analysis_id in all_analyses:
            del all_analyses[analysis_id]
        
        with open(metadata_file, 'w') as f:
            json.dump(all_analyses, f, indent=4)

def load_raw_data_from_cache(file_hash):
    """
    Loads raw script data from cache using file hash.
    """
    # Ensure cache directory exists
    ensure_cache_dir()
    
    # Try to find the raw data file
    raw_data_file = os.path.join(CACHE_DIR, f"raw_{file_hash}.parquet")
    
    if os.path.exists(raw_data_file):
        try:
            df = pd.read_parquet(raw_data_file)
            return df
        except Exception as e:
            print(f"Error loading raw data for hash {file_hash}: {e}")
            return None
    else:
        print(f"Raw data file not found: {raw_data_file}")
        return None
