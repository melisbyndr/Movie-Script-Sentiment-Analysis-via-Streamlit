import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

st.set_page_config(page_title="Model Comparison", layout="wide")

st.title("Model Performance Comparison")
st.markdown("""
This page allows you to compare the performance of sentiment and emotion analysis models 
against your labeled dataset. Upload your labeled data and see how well the models perform.
""")

# --- Helper Functions ---
def load_labeled_data(uploaded_file):
    """Load and validate the labeled dataset."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Please upload a CSV or Excel file.")
            return None
        
        # Check required columns
        required_cols = ['scene_id', 'dialogue', 'true_sentiment', 'true_emotion']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.info("Required columns: scene_id, dialogue, true_sentiment, true_emotion")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def prepare_comparison_data(labeled_df, model_df):
    """Prepare data for comparison by matching scenes and dialogues."""
    # Merge labeled data with model predictions
    comparison_df = pd.merge(
        labeled_df, 
        model_df[['scene_id', 'clean_dialogue', 'sentiment_label', 'sentiment_score', 'emotion_label', 'emotion_score']], 
        left_on=['scene_id', 'dialogue'], 
        right_on=['scene_id', 'clean_dialogue'], 
        how='inner'
    )
    
    if comparison_df.empty:
        st.warning("No matching dialogues found between labeled data and model predictions.")
        return None
    
    return comparison_df

def calculate_sentiment_metrics(comparison_df):
    """Calculate sentiment analysis performance metrics."""
    # Encode labels for comparison
    le_sentiment = LabelEncoder()
    
    # Combine all unique labels
    all_sentiment_labels = pd.concat([
        comparison_df['true_sentiment'],
        comparison_df['sentiment_label']
    ]).unique()
    
    le_sentiment.fit(all_sentiment_labels)
    
    y_true = le_sentiment.transform(comparison_df['true_sentiment'])
    y_pred = le_sentiment.transform(comparison_df['sentiment_label'])
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_labels = le_sentiment.classes_
    
    return metrics, cm, cm_labels, le_sentiment

def calculate_emotion_metrics(comparison_df):
    """Calculate emotion analysis performance metrics."""
    # Encode labels for comparison
    le_emotion = LabelEncoder()
    
    # Combine all unique labels
    all_emotion_labels = pd.concat([
        comparison_df['true_emotion'],
        comparison_df['emotion_label']
    ]).unique()
    
    le_emotion.fit(all_emotion_labels)
    
    y_true = le_emotion.transform(comparison_df['true_emotion'])
    y_pred = le_emotion.transform(comparison_df['emotion_label'])
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_labels = le_emotion.classes_
    
    return metrics, cm, cm_labels, le_emotion

def plot_confusion_matrix(cm, labels, title):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    return fig

def plot_metrics_comparison(sentiment_metrics, emotion_metrics):
    """Plot metrics comparison between sentiment and emotion analysis."""
    metrics_names = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)']
    
    sentiment_values = [
        sentiment_metrics['accuracy'],
        sentiment_metrics['precision_macro'],
        sentiment_metrics['recall_macro'],
        sentiment_metrics['f1_macro']
    ]
    
    emotion_values = [
        emotion_metrics['accuracy'],
        emotion_metrics['precision_macro'],
        emotion_metrics['recall_macro'],
        emotion_metrics['f1_macro']
    ]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics_names))
    width = 0.35
    
    ax.bar(x - width/2, sentiment_values, width, label='Sentiment Analysis', color='#4ECDC4')
    ax.bar(x + width/2, emotion_values, width, label='Emotion Analysis', color='#FF6B6B')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(sentiment_values):
        ax.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    for i, v in enumerate(emotion_values):
        ax.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def plot_error_analysis(comparison_df):
    """Plot error analysis showing where models make mistakes."""
    # Sentiment errors
    sentiment_errors = comparison_df[comparison_df['true_sentiment'] != comparison_df['sentiment_label']]
    emotion_errors = comparison_df[comparison_df['true_emotion'] != comparison_df['emotion_label']]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Error Analysis', fontsize=16)
    
    # Plot 1: Sentiment error distribution
    if not sentiment_errors.empty:
        error_counts = sentiment_errors.groupby(['true_sentiment', 'sentiment_label']).size().reset_index(name='count')
        pivot_errors = error_counts.pivot(index='true_sentiment', columns='sentiment_label', values='count').fillna(0)
        sns.heatmap(pivot_errors, annot=True, fmt='g', cmap='Reds', ax=axes[0, 0])
        axes[0, 0].set_title('Sentiment Error Patterns\n(True vs Predicted)')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')
    else:
        axes[0, 0].text(0.5, 0.5, 'No sentiment errors found', ha='center', va='center')
        axes[0, 0].set_title('Sentiment Error Patterns')
    
    # Plot 2: Emotion error distribution
    if not emotion_errors.empty:
        error_counts = emotion_errors.groupby(['true_emotion', 'emotion_label']).size().reset_index(name='count')
        pivot_errors = error_counts.pivot(index='true_emotion', columns='emotion_label', values='count').fillna(0)
        sns.heatmap(pivot_errors, annot=True, fmt='g', cmap='Reds', ax=axes[0, 1])
        axes[0, 1].set_title('Emotion Error Patterns\n(True vs Predicted)')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('True')
    else:
        axes[0, 1].text(0.5, 0.5, 'No emotion errors found', ha='center', va='center')
        axes[0, 1].set_title('Emotion Error Patterns')
    
    # Plot 3: Error rate by scene
    scene_sentiment_errors = sentiment_errors.groupby('scene_id').size() / comparison_df.groupby('scene_id').size()
    scene_emotion_errors = emotion_errors.groupby('scene_id').size() / comparison_df.groupby('scene_id').size()
    
    axes[1, 0].plot(scene_sentiment_errors.index, scene_sentiment_errors.values, 
                   marker='o', label='Sentiment Errors', color='red')
    axes[1, 0].plot(scene_emotion_errors.index, scene_emotion_errors.values, 
                   marker='s', label='Emotion Errors', color='blue')
    axes[1, 0].set_title('Error Rate by Scene')
    axes[1, 0].set_xlabel('Scene ID')
    axes[1, 0].set_ylabel('Error Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Confidence vs accuracy
    if 'sentiment_score' in comparison_df.columns and 'emotion_score' in comparison_df.columns:
        sentiment_correct = comparison_df['true_sentiment'] == comparison_df['sentiment_label']
        emotion_correct = comparison_df['true_emotion'] == comparison_df['emotion_label']
        
        axes[1, 1].scatter(comparison_df['sentiment_score'], sentiment_correct.astype(int), 
                          alpha=0.6, label='Sentiment', s=20)
        axes[1, 1].scatter(comparison_df['emotion_score'], emotion_correct.astype(int), 
                          alpha=0.6, label='Emotion', s=20)
        axes[1, 1].set_title('Confidence vs Accuracy')
        axes[1, 1].set_xlabel('Model Confidence Score')
        axes[1, 1].set_ylabel('Correct Prediction (1=Yes, 0=No)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Confidence scores not available', ha='center', va='center')
        axes[1, 1].set_title('Confidence vs Accuracy')
    
    plt.tight_layout()
    return fig

# --- Main Interface ---
st.header("Upload Labeled Dataset")

# File upload
uploaded_file = st.file_uploader(
    "Upload your labeled dataset (CSV or Excel)",
    type=['csv', 'xlsx'],
    help="File should contain columns: scene_id, dialogue, true_sentiment, true_emotion"
)

if uploaded_file is not None:
    # Load labeled data
    labeled_df = load_labeled_data(uploaded_file)
    
    if labeled_df is not None:
        st.success(f"Successfully loaded {len(labeled_df)} labeled dialogues")
        
        # Show sample of labeled data
        with st.expander("View Labeled Dataset Sample"):
            st.dataframe(labeled_df.head(10))
        
        # Check if we have model predictions
        if 'df_analyzed' in st.session_state:
            model_df = st.session_state.df_analyzed
            
            st.header("Model Comparison")
            
            # Prepare comparison data
            comparison_df = prepare_comparison_data(labeled_df, model_df)
            
            if comparison_df is not None:
                st.success(f"Found {len(comparison_df)} matching dialogues for comparison")
                
                # Show comparison sample
                with st.expander("View Comparison Sample"):
                    comparison_sample = comparison_df[['scene_id', 'dialogue', 'true_sentiment', 'sentiment_label', 'true_emotion', 'emotion_label']].head(10)
                    st.dataframe(comparison_sample)
                
                # Calculate metrics
                sentiment_metrics, sentiment_cm, sentiment_labels, le_sentiment = calculate_sentiment_metrics(comparison_df)
                emotion_metrics, emotion_cm, emotion_labels, le_emotion = calculate_emotion_metrics(comparison_df)
                
                # Display metrics
                st.header("Performance Metrics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sentiment Analysis Performance")
                    st.metric("Accuracy", f"{sentiment_metrics['accuracy']:.3f}")
                    st.metric("Precision (Macro)", f"{sentiment_metrics['precision_macro']:.3f}")
                    st.metric("Recall (Macro)", f"{sentiment_metrics['recall_macro']:.3f}")
                    st.metric("F1 Score (Macro)", f"{sentiment_metrics['f1_macro']:.3f}")
                
                with col2:
                    st.subheader("Emotion Analysis Performance")
                    st.metric("Accuracy", f"{emotion_metrics['accuracy']:.3f}")
                    st.metric("Precision (Macro)", f"{emotion_metrics['precision_macro']:.3f}")
                    st.metric("Recall (Macro)", f"{emotion_metrics['recall_macro']:.3f}")
                    st.metric("F1 Score (Macro)", f"{emotion_metrics['f1_macro']:.3f}")
                
                # Create tabs for detailed analysis
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Metrics Comparison", 
                    "Confusion Matrices", 
                    "Error Analysis",
                    "Detailed Results"
                ])
                
                with tab1:
                    st.subheader("Model Performance Comparison")
                    fig_comparison = plot_metrics_comparison(sentiment_metrics, emotion_metrics)
                    st.pyplot(fig_comparison)
                    
                    # Additional metrics table
                    st.subheader("Detailed Metrics")
                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'Precision (Macro)', 'Precision (Weighted)', 
                                 'Recall (Macro)', 'Recall (Weighted)', 'F1 (Macro)', 'F1 (Weighted)'],
                        'Sentiment Analysis': [
                            sentiment_metrics['accuracy'],
                            sentiment_metrics['precision_macro'],
                            sentiment_metrics['precision_weighted'],
                            sentiment_metrics['recall_macro'],
                            sentiment_metrics['recall_weighted'],
                            sentiment_metrics['f1_macro'],
                            sentiment_metrics['f1_weighted']
                        ],
                        'Emotion Analysis': [
                            emotion_metrics['accuracy'],
                            emotion_metrics['precision_macro'],
                            emotion_metrics['precision_weighted'],
                            emotion_metrics['recall_macro'],
                            emotion_metrics['recall_weighted'],
                            emotion_metrics['f1_macro'],
                            emotion_metrics['f1_weighted']
                        ]
                    })
                    st.dataframe(metrics_df, use_container_width=True)
                
                with tab2:
                    st.subheader("Confusion Matrices")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Sentiment Analysis Confusion Matrix")
                        fig_sentiment_cm = plot_confusion_matrix(sentiment_cm, sentiment_labels, "Sentiment Analysis")
                        st.pyplot(fig_sentiment_cm)
                    
                    with col2:
                        st.write("Emotion Analysis Confusion Matrix")
                        fig_emotion_cm = plot_confusion_matrix(emotion_cm, emotion_labels, "Emotion Analysis")
                        st.pyplot(fig_emotion_cm)
                
                with tab3:
                    st.subheader("Error Analysis")
                    fig_error = plot_error_analysis(comparison_df)
                    st.pyplot(fig_error)
                    
                    # Show some example errors
                    st.subheader("Example Errors")
                    
                    sentiment_errors = comparison_df[comparison_df['true_sentiment'] != comparison_df['sentiment_label']]
                    emotion_errors = comparison_df[comparison_df['true_emotion'] != comparison_df['emotion_label']]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if not sentiment_errors.empty:
                            st.write("Sentiment Analysis Errors:")
                            error_sample = sentiment_errors[['dialogue', 'true_sentiment', 'sentiment_label']].head(5)
                            st.dataframe(error_sample)
                        else:
                            st.success("No sentiment analysis errors found!")
                    
                    with col2:
                        if not emotion_errors.empty:
                            st.write("Emotion Analysis Errors:")
                            error_sample = emotion_errors[['dialogue', 'true_emotion', 'emotion_label']].head(5)
                            st.dataframe(error_sample)
                        else:
                            st.success("No emotion analysis errors found!")
                
                with tab4:
                    st.subheader("Detailed Comparison Results")
                    
                    # Filter options
                    st.write("Filter Results:")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        show_correct = st.checkbox("Show Correct Predictions", value=True)
                    with col2:
                        show_sentiment_errors = st.checkbox("Show Sentiment Errors", value=True)
                    with col3:
                        show_emotion_errors = st.checkbox("Show Emotion Errors", value=True)
                    
                    # Filter data
                    filtered_df = comparison_df.copy()
                    
                    if not show_correct:
                        sentiment_errors = comparison_df['true_sentiment'] != comparison_df['sentiment_label']
                        emotion_errors = comparison_df['true_emotion'] != comparison_df['emotion_label']
                        filtered_df = filtered_df[sentiment_errors | emotion_errors]
                    
                    if not show_sentiment_errors:
                        sentiment_correct = comparison_df['true_sentiment'] == comparison_df['sentiment_label']
                        filtered_df = filtered_df[sentiment_correct]
                    
                    if not show_emotion_errors:
                        emotion_correct = comparison_df['true_emotion'] == comparison_df['emotion_label']
                        filtered_df = filtered_df[emotion_correct]
                    
                    # Display filtered results
                    display_cols = ['scene_id', 'dialogue', 'true_sentiment', 'sentiment_label', 
                                  'true_emotion', 'emotion_label']
                    if 'sentiment_score' in filtered_df.columns:
                        display_cols.extend(['sentiment_score', 'emotion_score'])
                    
                    st.dataframe(filtered_df[display_cols], use_container_width=True)
                    
                    # Download comparison results
                    st.download_button(
                        label="Download Comparison Results",
                        data=filtered_df.to_csv(index=False),
                        file_name="model_comparison_results.csv",
                        mime="text/csv"
                    )
        
        else:
            st.warning("No model predictions available. Please run sentiment analysis on a script first.")
    
else:
    st.info("Please upload a labeled dataset to begin comparison.")
    
    # Show expected format
    st.header("Expected Dataset Format")
    st.markdown("""
    Your labeled dataset should contain the following columns:
    
    - **scene_id**: Scene identifier (integer)
    - **dialogue**: The dialogue text (string)
    - **true_sentiment**: True sentiment label (e.g., 'positive', 'negative', 'neutral')
    - **true_emotion**: True emotion label (e.g., 'joy', 'sadness', 'anger', 'fear')
    
    Example:
    """)
    
    example_data = pd.DataFrame({
        'scene_id': [1, 1, 2, 2, 3],
        'dialogue': [
            "I'm so happy to see you!",
            "This is terrible news.",
            "I love this movie.",
            "I'm scared of the dark.",
            "This makes me angry."
        ],
        'true_sentiment': ['positive', 'negative', 'positive', 'negative', 'negative'],
        'true_emotion': ['joy', 'sadness', 'joy', 'fear', 'anger']
    })
    
    st.dataframe(example_data) 