import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from itertools import combinations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def prepare_scene_pivot(df_script):
    """
    Returns a pivot table: characters (rows) vs. scenes (columns) with line counts.
    """
    if "character" not in df_script.columns or "scene_id" not in df_script.columns:
        return pd.DataFrame() # Return empty df if columns are missing

    sorted_counts = (
        df_script
        .groupby(["character", "scene_id"])
        .size()
        .reset_index(name="line_count")
        .sort_values(by="line_count", ascending=False)
    )



    return sorted_counts


def plot_character_heatmap(df_script, top_n=10):
    """
    Heatmap of top N characters across scenes.
    """
    if "character" not in df_script.columns or "scene_id" not in df_script.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Data not available for heatmap.", ha='center', va='center')
        return fig

    sorted_counts = prepare_scene_pivot(df_script)
    
    if sorted_counts.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data available for heatmap.", ha='center', va='center')
        return fig

    scene_characters_pivot = (
    sorted_counts
    .pivot_table(index="character", columns="scene_id", aggfunc="size", fill_value=0)
)

    character_order = scene_characters_pivot.sum(axis=1).sort_values(ascending=False).index

    # Sort scene IDs to ensure they are in order (default is likely correct, but this guarantees it)
    scene_order = sorted(scene_characters_pivot.columns)

    scene_characters_pivot = scene_characters_pivot.loc[character_order, scene_order]

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(scene_characters_pivot.head(top_n), cmap="YlGnBu", cbar=True, ax=ax)
    ax.set_title(f"Top {top_n} Character Appearances per Scene")
    ax.set_xlabel("Scene ID")
    ax.set_ylabel("Character")
    plt.tight_layout()

    return fig


def plot_character_line_counts(df_script, top_n=10):
    """
    Bar plot of top N characters by total line count.
    """
    fig, ax = plt.subplots()
    if 'character' not in df_script.columns or df_script.empty:
        ax.text(0.5, 0.5, "No character data available.", ha='center', va='center')
        return fig

    line_counts = (
        df_script.groupby("character")
        .size()
        .reset_index(name="line_count")
        .sort_values(by="line_count", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=line_counts, x="line_count", y="character", ax=ax)
    ax.set_title(f"Top {top_n} Characters by Line Count")
    ax.set_xlabel("Line Count")
    ax.set_ylabel("Character")
    plt.tight_layout()

    return fig

def plot_scene_intensity(df_script):
    """
    Plots the total number of lines per scene as a line chart.
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    if 'scene_id' not in df_script.columns or df_script.empty:
        ax.text(0.5, 0.5, "No scene data available.", ha='center', va='center')
        return fig

    scene_intensity = df_script.groupby("scene_id").size().reset_index(name="total_lines")
    sns.lineplot(data=scene_intensity, x="scene_id", y="total_lines", marker='o', ax=ax)
    ax.set_title("Scene Intensity: Total Lines per Scene")
    ax.set_xlabel("Scene ID")
    ax.set_ylabel("Total Number of Lines")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

def plot_average_dialogue_length(df_script, top_n=10):
    """
    Plots the average dialogue length (in words) for the top N characters.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    if 'clean_dialogue' not in df_script.columns or 'character' not in df_script.columns or df_script.empty:
        ax.text(0.5, 0.5, "Data not available for dialogue analysis.", ha='center', va='center')
        return fig

    df_script['dialogue_words'] = df_script['clean_dialogue'].str.split().str.len()
    
    top_characters = df_script['character'].value_counts().nlargest(top_n).index
    df_top_characters = df_script[df_script['character'].isin(top_characters)]

    avg_length = df_top_characters.groupby('character')['dialogue_words'].mean().sort_values(ascending=False).reset_index()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=avg_length, x="dialogue_words", y="character", ax=ax)
    ax.set_title(f"Average Dialogue Length (Top {top_n} Characters)")
    ax.set_xlabel("Average Words per Dialogue")
    ax.set_ylabel("Character")
    plt.tight_layout()
    return fig

def plot_character_interaction_network(df_script, top_n=7):
    """
    Creates and plots a character interaction network based on co-occurrence in scenes.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    if 'character' not in df_script.columns or 'scene_id' not in df_script.columns or df_script.empty:
        ax.text(0.5, 0.5, "Data not available for interaction network.", ha='center', va='center')
        return fig

    top_characters = df_script['character'].value_counts().nlargest(top_n).index
    df_top = df_script[df_script['character'].isin(top_characters)]

    scene_characters = df_top.groupby('scene_id')['character'].unique().apply(list)

    G = nx.Graph()
    G.add_nodes_from(top_characters)

    for scene_id, characters in scene_characters.items():
        for char1, char2 in combinations(characters, 2):
            if G.has_edge(char1, char2):
                G[char1][char2]['weight'] += 1
            else:
                G.add_edge(char1, char2, weight=1)

    pos = nx.spring_layout(G, k=0.8, iterations=50)

    # Calculate node sizes based on degree, ensuring integer conversion for the linter
    node_sizes = []
    for n in G.nodes():
        degree = G.degree(n)
        if isinstance(degree, (int, float)):
            node_sizes.append(int(degree * 500))
        else:
            node_sizes.append(500)  # Default size
    
    edges = G.edges(data=True)
    edge_weights = [d.get('weight', 1) for (u, v, d) in edges]
    
    # Normalize weights for better visualization
    edge_widths = [1.0] * len(edges) # Default width
    if edge_weights and max(edge_weights) > 0:
        max_weight = max(edge_weights)
        edge_widths = [float((w / max_weight * 6.0) + 1.0) for w in edge_weights]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.9, ax=ax)
    for i, (u, v) in enumerate(G.edges()):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=edge_widths[i], alpha=0.7, edge_color='gray', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax)

    ax.set_title(f"Character Interaction Network (Top {top_n})")
    plt.tight_layout()
    
    return fig

def analyze_character_arc(df_script, character_name):
    """
    Analyzes the character arc by tracking sentiment and dialogue patterns over time.
    """
    if 'character' not in df_script.columns or df_script.empty:
        return None, None, None
    
    df_char = df_script[df_script['character'] == character_name].copy()
    if df_char.empty:
        return None, None, None
    
    # Calculate character arc metrics
    df_char = df_char.sort_values('scene_id')
    
    # Dialogue complexity (words per dialogue)
    df_char['dialogue_words'] = df_char['clean_dialogue'].str.split().str.len()
    
    # Sentiment progression (if available)
    if 'sentiment_score' in df_char.columns:
        df_char['sentiment_numeric'] = df_char['sentiment_label'].apply(
            lambda x: 1 if 'pos' in str(x).lower() else (-1 if 'neg' in str(x).lower() else 0)
        )
        sentiment_trend = df_char['sentiment_numeric'].rolling(window=3, min_periods=1).mean()
    else:
        sentiment_trend = None
    
    # Character development phases
    total_scenes = len(df_char)
    if total_scenes >= 3:
        # Divide into three acts
        act1_end = total_scenes // 3
        act2_end = 2 * total_scenes // 3
        
        act1 = df_char.iloc[:act1_end]
        act2 = df_char.iloc[act1_end:act2_end]
        act3 = df_char.iloc[act2_end:]
        
        arc_analysis = {
            'total_dialogues': len(df_char),
            'total_scenes': df_char['scene_id'].nunique(),
            'avg_dialogue_length': df_char['dialogue_words'].mean(),
            'act1_dialogues': len(act1),
            'act2_dialogues': len(act2),
            'act3_dialogues': len(act3),
            'dialogue_distribution': [len(act1), len(act2), len(act3)]
        }
    else:
        arc_analysis = {
            'total_dialogues': len(df_char),
            'total_scenes': df_char['scene_id'].nunique(),
            'avg_dialogue_length': df_char['dialogue_words'].mean(),
            'dialogue_distribution': [len(df_char)]
        }
    
    return df_char, arc_analysis, sentiment_trend

def plot_character_arc(df_script, character_name):
    """
    Creates a comprehensive character arc visualization.
    """
    df_char, arc_analysis, sentiment_trend = analyze_character_arc(df_script, character_name)
    
    if df_char is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No data available for character: {character_name}", ha='center', va='center')
        return fig
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Character Arc Analysis: {character_name}', fontsize=16)
    
    # Plot 1: Dialogue frequency over time
    dialogue_freq = df_char.groupby('scene_id').size().reset_index(name='dialogue_count')
    axes[0, 0].plot(dialogue_freq['scene_id'], dialogue_freq['dialogue_count'], marker='o', linewidth=2)
    axes[0, 0].set_title('Dialogue Frequency Over Time')
    axes[0, 0].set_xlabel('Scene ID')
    axes[0, 0].set_ylabel('Number of Dialogues')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Average dialogue length over time
    avg_length = df_char.groupby('scene_id')['dialogue_words'].mean().reset_index()
    axes[0, 1].plot(avg_length['scene_id'], avg_length['dialogue_words'], marker='s', color='orange', linewidth=2)
    axes[0, 1].set_title('Average Dialogue Length Over Time')
    axes[0, 1].set_xlabel('Scene ID')
    axes[0, 1].set_ylabel('Average Words per Dialogue')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Sentiment trend (if available)
    if sentiment_trend is not None:
        axes[1, 0].plot(df_char['scene_id'], sentiment_trend, marker='^', color='green', linewidth=2)
        axes[1, 0].set_title('Sentiment Trend Over Time')
        axes[1, 0].set_xlabel('Scene ID')
        axes[1, 0].set_ylabel('Sentiment Score')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Sentiment data not available', ha='center', va='center')
        axes[1, 0].set_title('Sentiment Trend Over Time')
    
    # Plot 4: Act distribution
    acts = ['Act 1', 'Act 2', 'Act 3']
    act_counts = arc_analysis['dialogue_distribution']
    if len(act_counts) == 3:
        axes[1, 1].bar(acts, act_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1, 1].set_title('Dialogue Distribution Across Acts')
        axes[1, 1].set_ylabel('Number of Dialogues')
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data for act analysis', ha='center', va='center')
        axes[1, 1].set_title('Dialogue Distribution Across Acts')
    
    plt.tight_layout()
    return fig

def create_character_personality_profile(df_script, character_name):
    """
    Creates a personality profile based on dialogue patterns and sentiment.
    """
    if 'character' not in df_script.columns or df_script.empty:
        return None
    
    df_char = df_script[df_script['character'] == character_name].copy()
    if df_char.empty:
        return None
    
    # Calculate personality metrics
    profile = {
        'character_name': character_name,
        'total_dialogues': len(df_char),
        'total_scenes': df_char['scene_id'].nunique(),
        'avg_dialogue_length': df_char['clean_dialogue'].str.split().str.len().mean(),
        'dialogue_variability': df_char['clean_dialogue'].str.split().str.len().std(),
        'scene_presence': len(df_char) / df_script['scene_id'].nunique(),
        'dominance_score': len(df_char) / len(df_script) * 100
    }
    
    # Sentiment analysis (if available)
    if 'sentiment_label' in df_char.columns:
        sentiment_dist = df_char['sentiment_label'].value_counts(normalize=True)
        profile['sentiment_profile'] = sentiment_dist.to_dict()
        
        # Calculate emotional stability
        if 'sentiment_score' in df_char.columns:
            profile['emotional_stability'] = 1 - df_char['sentiment_score'].std()
        else:
            profile['emotional_stability'] = None
    
    # Emotion analysis (if available)
    if 'emotion_label' in df_char.columns:
        emotion_dist = df_char['emotion_label'].value_counts(normalize=True)
        profile['emotion_profile'] = emotion_dist.to_dict()
        profile['primary_emotion'] = emotion_dist.index[0] if len(emotion_dist) > 0 else None
    
    # Dialogue style analysis
    df_char['dialogue_words'] = df_char['clean_dialogue'].str.split().str.len()
    profile['dialogue_style'] = {
        'avg_length': df_char['dialogue_words'].mean(),
        'length_variability': df_char['dialogue_words'].std(),
        'short_dialogues': len(df_char[df_char['dialogue_words'] <= 5]),
        'long_dialogues': len(df_char[df_char['dialogue_words'] >= 20])
    }
    
    return profile

def plot_character_personality_profile(df_script, character_name):
    """
    Visualizes the character personality profile.
    """
    profile = create_character_personality_profile(df_script, character_name)
    
    if profile is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No data available for character: {character_name}", ha='center', va='center')
        return fig
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Personality Profile: {character_name}', fontsize=16)
    
    # Plot 1: Basic metrics radar chart
    metrics = ['Dialogue Count', 'Scene Presence', 'Avg Length', 'Dominance']
    values = [
        profile['total_dialogues'] / 100,  # Normalize
        profile['scene_presence'] * 10,    # Normalize
        profile['avg_dialogue_length'] / 10,  # Normalize
        profile['dominance_score'] / 10    # Normalize
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]  # Complete the circle
    angles += angles[:1]
    
    axes[0, 0].plot(angles, values, 'o-', linewidth=2)
    axes[0, 0].fill(angles, values, alpha=0.25)
    axes[0, 0].set_xticks(angles[:-1])
    axes[0, 0].set_xticklabels(metrics)
    axes[0, 0].set_title('Character Metrics')
    axes[0, 0].grid(True)
    
    # Plot 2: Sentiment distribution (if available)
    sentiment_profile = profile.get('sentiment_profile')
    if sentiment_profile is not None and len(sentiment_profile) > 0:
        sentiment_data = sentiment_profile
        axes[0, 1].pie(sentiment_data.values(), labels=sentiment_data.keys(), autopct='%1.1f%%')
        axes[0, 1].set_title('Sentiment Distribution')
    else:
        axes[0, 1].text(0.5, 0.5, 'Sentiment data not available', ha='center', va='center')
        axes[0, 1].set_title('Sentiment Distribution')
    
    # Plot 3: Emotion distribution (if available)
    emotion_profile = profile.get('emotion_profile')
    if emotion_profile is not None and len(emotion_profile) > 0:
        emotion_data = emotion_profile
        top_emotions = dict(list(emotion_data.items())[:5])  # Top 5 emotions
        axes[1, 0].barh(list(top_emotions.keys()), list(top_emotions.values()))
        axes[1, 0].set_title('Top Emotions')
        axes[1, 0].set_xlabel('Frequency')
    else:
        axes[1, 0].text(0.5, 0.5, 'Emotion data not available', ha='center', va='center')
        axes[1, 0].set_title('Top Emotions')
    
    # Plot 4: Dialogue style
    style_data = profile['dialogue_style']
    style_metrics = ['Short', 'Medium', 'Long']
    df_char_temp = df_script[df_script['character'] == character_name]
    style_counts = [
        style_data['short_dialogues'],
        len(df_char_temp) - style_data['short_dialogues'] - style_data['long_dialogues'],
        style_data['long_dialogues']
    ]
    axes[1, 1].bar(style_metrics, style_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1, 1].set_title('Dialogue Length Distribution')
    axes[1, 1].set_ylabel('Number of Dialogues')
    
    plt.tight_layout()
    return fig

def analyze_character_interaction_quality(df_script, character_name):
    """
    Analyzes the quality and nature of character interactions.
    """
    if 'character' not in df_script.columns or 'scene_id' not in df_script.columns:
        return None
    
    # Get scenes where the character appears
    char_scenes = df_script[df_script['character'] == character_name]['scene_id'].unique()
    
    # Find other characters in the same scenes
    interaction_data = []
    for scene_id in char_scenes:
        scene_characters = df_script[df_script['scene_id'] == scene_id]['character'].unique()
        other_characters = [c for c in scene_characters if c != character_name]
        
        for other_char in other_characters:
            # Get dialogues for both characters in this scene
            char_dialogues = df_script[(df_script['scene_id'] == scene_id) & 
                                     (df_script['character'] == character_name)]
            other_dialogues = df_script[(df_script['scene_id'] == scene_id) & 
                                      (df_script['character'] == other_char)]
            
            # Calculate interaction metrics
            interaction = {
                'scene_id': scene_id,
                'other_character': other_char,
                'char_dialogue_count': len(char_dialogues),
                'other_dialogue_count': len(other_dialogues),
                'total_dialogues': len(char_dialogues) + len(other_dialogues),
                'dialogue_balance': abs(len(char_dialogues) - len(other_dialogues)) / (len(char_dialogues) + len(other_dialogues)) if (len(char_dialogues) + len(other_dialogues)) > 0 else 0
            }
            
            # Sentiment analysis for interaction (if available)
            if 'sentiment_score' in df_script.columns:
                char_sentiment = char_dialogues['sentiment_score'].mean() if len(char_dialogues) > 0 else 0
                other_sentiment = other_dialogues['sentiment_score'].mean() if len(other_dialogues) > 0 else 0
                interaction['sentiment_compatibility'] = 1 - abs(char_sentiment - other_sentiment)
            
            interaction_data.append(interaction)
    
    if not interaction_data:
        return None
    
    df_interactions = pd.DataFrame(interaction_data)
    
    # Calculate overall interaction quality metrics
    quality_metrics = {
        'total_interactions': len(df_interactions),
        'unique_characters': df_interactions['other_character'].nunique(),
        'avg_dialogue_balance': df_interactions['dialogue_balance'].mean(),
        'most_interactive_character': df_interactions.groupby('other_character')['total_dialogues'].sum().idxmax() if len(df_interactions) > 0 else None,
        'interaction_diversity': df_interactions['other_character'].nunique() / len(df_script['character'].unique()) if len(df_script['character'].unique()) > 1 else 0
    }
    
    return df_interactions, quality_metrics

def plot_character_interaction_quality(df_script, character_name):
    """
    Visualizes character interaction quality analysis.
    """
    result = analyze_character_interaction_quality(df_script, character_name)
    
    if result is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No interaction data available for character: {character_name}", ha='center', va='center')
        return fig
    
    df_interactions, quality_metrics = result
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Interaction Quality Analysis: {character_name}', fontsize=16)
    
    # Plot 1: Interaction frequency with other characters
    char_interaction_counts = df_interactions.groupby('other_character')['total_dialogues'].sum().sort_values(ascending=True)
    axes[0, 0].barh(range(len(char_interaction_counts)), char_interaction_counts.values)
    axes[0, 0].set_yticks(range(len(char_interaction_counts)))
    axes[0, 0].set_yticklabels(char_interaction_counts.index)
    axes[0, 0].set_title('Interaction Frequency with Other Characters')
    axes[0, 0].set_xlabel('Total Dialogues')
    
    # Plot 2: Dialogue balance distribution
    axes[0, 1].hist(df_interactions['dialogue_balance'], bins=10, alpha=0.7, color='skyblue')
    axes[0, 1].set_title('Dialogue Balance Distribution')
    axes[0, 1].set_xlabel('Dialogue Balance (0=Equal, 1=Unequal)')
    axes[0, 1].set_ylabel('Frequency')
    
    # Plot 3: Interaction quality metrics
    metrics = ['Total Interactions', 'Unique Characters', 'Interaction Diversity']
    values = [
        quality_metrics['total_interactions'],
        quality_metrics['unique_characters'],
        quality_metrics['interaction_diversity'] * 100  # Convert to percentage
    ]
    axes[1, 0].bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1, 0].set_title('Interaction Quality Metrics')
    axes[1, 0].set_ylabel('Count/Percentage')
    
    # Plot 4: Scene-wise interaction pattern
    scene_interactions = df_interactions.groupby('scene_id')['total_dialogues'].sum()
    axes[1, 1].plot(scene_interactions.index, scene_interactions.values, marker='o', linewidth=2)
    axes[1, 1].set_title('Interaction Intensity Across Scenes')
    axes[1, 1].set_xlabel('Scene ID')
    axes[1, 1].set_ylabel('Total Dialogues in Scene')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def identify_protagonist_antagonist(df_script):
    """
    Identifies protagonist and antagonist based on dialogue patterns, sentiment, and screen time.
    """
    if 'character' not in df_script.columns or df_script.empty:
        return None, None
    
    # Calculate character metrics
    char_metrics = []
    
    for character in df_script['character'].unique():
        df_char = df_script[df_script['character'] == character]
        
        metrics = {
            'character': character,
            'total_dialogues': len(df_char),
            'total_scenes': df_char['scene_id'].nunique(),
            'avg_dialogue_length': df_char['clean_dialogue'].str.split().str.len().mean(),
            'scene_presence': len(df_char) / df_script['scene_id'].nunique(),
            'dominance_score': len(df_char) / len(df_script) * 100
        }
        
        # Sentiment analysis (if available)
        if 'sentiment_score' in df_char.columns:
            metrics['avg_sentiment'] = df_char['sentiment_score'].mean()
            metrics['sentiment_consistency'] = 1 - df_char['sentiment_score'].std()
        else:
            metrics['avg_sentiment'] = 0
            metrics['sentiment_consistency'] = 0
        
        # Emotion analysis (if available)
        if 'emotion_label' in df_char.columns:
            emotion_dist = df_char['emotion_label'].value_counts(normalize=True)
            metrics['primary_emotion'] = emotion_dist.index[0] if len(emotion_dist) > 0 else 'unknown'
        else:
            metrics['primary_emotion'] = 'unknown'
        
        char_metrics.append(metrics)
    
    df_metrics = pd.DataFrame(char_metrics)
    
    # Normalize metrics for scoring
    scaler = StandardScaler()
    numeric_cols = ['total_dialogues', 'total_scenes', 'avg_dialogue_length', 'scene_presence', 'dominance_score']
    df_metrics[numeric_cols] = scaler.fit_transform(df_metrics[numeric_cols])
    
    # Calculate protagonist score (high screen time, positive sentiment, consistent character)
    df_metrics['protagonist_score'] = (
        df_metrics['total_dialogues'] * 0.3 +
        df_metrics['scene_presence'] * 0.3 +
        df_metrics['avg_sentiment'] * 0.2 +
        df_metrics['sentiment_consistency'] * 0.2
    )
    
    # Calculate antagonist score (high screen time, negative sentiment, or high conflict)
    df_metrics['antagonist_score'] = (
        df_metrics['total_dialogues'] * 0.3 +
        df_metrics['scene_presence'] * 0.3 +
        (-df_metrics['avg_sentiment']) * 0.2 +
        (1 - df_metrics['sentiment_consistency']) * 0.2  # Inconsistent emotions
    )
    
    # Identify protagonist and antagonist
    protagonist = df_metrics.loc[df_metrics['protagonist_score'].idxmax()]
    antagonist = df_metrics.loc[df_metrics['antagonist_score'].idxmax()]
    
    return protagonist, antagonist

def plot_protagonist_antagonist_analysis(df_script):
    """
    Visualizes protagonist vs antagonist analysis.
    """
    protagonist, antagonist = identify_protagonist_antagonist(df_script)
    
    if protagonist is None or antagonist is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No character data available for analysis", ha='center', va='center')
        return fig
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Protagonist vs Antagonist Analysis', fontsize=16)
    
    # Plot 1: Protagonist vs Antagonist comparison
    comparison_metrics = ['Total Dialogues', 'Scene Presence', 'Avg Sentiment', 'Character Consistency']
    prot_values = [
        protagonist['total_dialogues'],
        protagonist['scene_presence'],
        protagonist['avg_sentiment'],
        protagonist['sentiment_consistency']
    ]
    ant_values = [
        antagonist['total_dialogues'],
        antagonist['scene_presence'],
        antagonist['avg_sentiment'],
        antagonist['sentiment_consistency']
    ]
    
    x = np.arange(len(comparison_metrics))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, prot_values, width, label='Protagonist', color='#4ECDC4')
    axes[0, 0].bar(x + width/2, ant_values, width, label='Antagonist', color='#FF6B6B')
    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Normalized Score')
    axes[0, 0].set_title('Character Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(comparison_metrics)
    axes[0, 0].legend()
    
    # Plot 2: Character role scores
    all_characters = df_script['character'].value_counts().head(10)
    char_scores = []
    char_names = []
    
    for char in all_characters.index:
        char_data = df_script[df_script['character'] == char]
        if len(char_data) > 0:
            prot_score = (
                len(char_data) / len(df_script) * 0.4 +
                (char_data['sentiment_score'].mean() if 'sentiment_score' in char_data.columns else 0) * 0.3 +
                (1 - char_data['sentiment_score'].std() if 'sentiment_score' in char_data.columns else 0) * 0.3
            )
            char_scores.append(prot_score)
            char_names.append(char)
    
    axes[0, 1].barh(char_names, char_scores, color=['#4ECDC4' if i == 0 else '#FF6B6B' if i == 1 else '#45B7D1' for i in range(len(char_names))])
    axes[0, 1].set_title('Character Role Scores')
    axes[0, 1].set_xlabel('Protagonist Score')
    
    # Plot 3: Sentiment distribution comparison
    if 'sentiment_score' in df_script.columns:
        prot_dialogues = df_script[df_script['character'] == protagonist['character']]
        ant_dialogues = df_script[df_script['character'] == antagonist['character']]
        
        axes[1, 0].hist(prot_dialogues['sentiment_score'], alpha=0.7, label='Protagonist', color='#4ECDC4', bins=10)
        axes[1, 0].hist(ant_dialogues['sentiment_score'], alpha=0.7, label='Antagonist', color='#FF6B6B', bins=10)
        axes[1, 0].set_title('Sentiment Distribution Comparison')
        axes[1, 0].set_xlabel('Sentiment Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'Sentiment data not available', ha='center', va='center')
        axes[1, 0].set_title('Sentiment Distribution Comparison')
    
    # Plot 4: Character arc comparison
    prot_scenes = df_script[df_script['character'] == protagonist['character']].groupby('scene_id').size()
    ant_scenes = df_script[df_script['character'] == antagonist['character']].groupby('scene_id').size()
    
    axes[1, 1].plot(prot_scenes.index, prot_scenes.values, label='Protagonist', color='#4ECDC4', linewidth=2)
    axes[1, 1].plot(ant_scenes.index, ant_scenes.values, label='Antagonist', color='#FF6B6B', linewidth=2)
    axes[1, 1].set_title('Screen Time Evolution')
    axes[1, 1].set_xlabel('Scene ID')
    axes[1, 1].set_ylabel('Dialogues per Scene')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
