#!/usr/bin/env python3
#
# youtube_transcript_emotion.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline, AutoTokenizer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import numpy as np
from scipy.stats import zscore
from scipy.spatial.distance import euclidean
import requests
from bs4 import BeautifulSoup

def get_text_from_url(url):
    """
    Fetches and returns the main text content from the given URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the text from the webpage
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        return text

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL content: {e}")
        sys.exit(1)

def get_text_from_file(file_path):
    """
    Reads and returns the text content from a given file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text

    except FileNotFoundError as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def get_input_text(input_source, source_type):
    """
    Determines the type of input source (url, file, youtube) and fetches the text.
    """
    if source_type == 'url':
        return get_text_from_url(input_source)
    elif source_type == 'file':
        return get_text_from_file(input_source)
    elif source_type == 'youtube':
        video_id = get_youtube_video_id(input_source)
        return download_transcript(video_id)
    else:
        raise ValueError("Invalid source type. Please choose 'url', 'file', or 'youtube'.")

def calculate_emotional_baseline(all_results):
    emotion_scores = defaultdict(list)
    for result in all_results:
        for emotion in result['emotions'][0]:
            emotion_scores[emotion['label']].append(emotion['score'])

    emotion_baseline = {emotion: np.mean(scores) for emotion, scores in emotion_scores.items()}
    return emotion_baseline

def calculate_ebd_score(all_results, emotion_baseline):
    for result in all_results:
        ebd_sum = 0
        for emotion in result['emotions'][0]:
            baseline = emotion_baseline[emotion['label']]
            ebd_sum += abs(emotion['score'] - baseline) / baseline
        result['ebd_score'] = ebd_sum / len(result['emotions'][0])

    return all_results

def calculate_emotional_inconsistency_score(all_results):
    for i in range(1, len(all_results)):
        previous_dominant = all_results[i - 1]['dominant_emotion']
        current_dominant = all_results[i]['dominant_emotion']
        if previous_dominant != current_dominant:
            all_results[i]['ei_score'] = 1
        else:
            all_results[i]['ei_score'] = 0

    return all_results

def calculate_cec_score(all_results):
    for result in all_results:
        cec_score = 0
        for emotion in result['emotions'][0]:
            if emotion['label'] == result['dominant_emotion'] and emotion['score'] < 0.5:
                cec_score += 1
        result['cec_score'] = cec_score

    return all_results

def calculate_deception_score(all_results):
    """
    Calculates the overall Deception Score for each chunk.
    This score combines the various anomaly and inconsistency measures.
    """
    for result in all_results:
        deception_score = (
            0.3 * result.get('is_anomaly_knn', 0) +
            0.3 * result.get('is_anomaly_isolation_forest', 0) +
            0.2 * result.get('ei_score', 0) +  # Emotional Inconsistency score
            0.2 * result.get('is_anomaly_zscore', 0)
        )
        result['deception_score'] = deception_score

    return all_results

def plot_deception_score_over_time(all_results):
    """
    Plots the deception score over time.
    """
    chunk_numbers = [result['chunk_number'] for result in all_results]
    deception_scores = [result['deception_score'] for result in all_results]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chunk_numbers, y=deception_scores,
        mode='lines+markers',
        line=dict(color='red'),
        name='Deception Score'
    ))

    fig.update_layout(
        title="Deception Score Over Time",
        xaxis_title="Chunk Number",
        yaxis_title="Deception Score",
        template='plotly_dark'
    )

    fig.show()

def calculate_overall_deception_score(all_results, threshold=0.5):
    """
    Calculate an overall deception score based on the frequency and proportion of high deception scores.
    """
    high_deception_chunks = [result for result in all_results if result['deception_score'] > threshold]
    frequency_of_high_deception = len(high_deception_chunks)
    proportion_of_high_deception = frequency_of_high_deception / len(all_results)
    
    # Combine frequency and proportion into a final overall deception score
    overall_deception_score = (frequency_of_high_deception * proportion_of_high_deception) / len(all_results)
    
    return overall_deception_score, frequency_of_high_deception, proportion_of_high_deception

def generate_deception_summary(all_results):
    deception_scores = [result['deception_score'] for result in all_results]
    
    max_deception_result = max(all_results, key=lambda x: x['deception_score'])
    min_deception_result = min(all_results, key=lambda x: x['deception_score'])
    median_deception_result = sorted(all_results, key=lambda x: x['deception_score'])[len(all_results) // 2]

    overall_deception_score, high_deception_count, high_deception_proportion = calculate_overall_deception_score(all_results)
    
    summary = {
        "total_chunks_processed": len(all_results),  # Total number of chunks processed
        "overall_deception_score": overall_deception_score,  # New overall deception score
        "frequency_of_high_deception": high_deception_count,  # Frequency of high deception chunks
        "proportion_of_high_deception": high_deception_proportion,  # Proportion of high deception chunks
        "average_deception_score": np.mean(deception_scores),
        "max_deception_score": max_deception_result['deception_score'],
        "min_deception_score": min_deception_result['deception_score'],
        "std_deception_score": np.std(deception_scores),
        "max_deception_chunk": max_deception_result['chunk_number'],
        "max_deception_text": max_deception_result['text'],
        "min_deception_chunk": min_deception_result['chunk_number'],
        "min_deception_text": min_deception_result['text'],
        "median_deception_chunk": median_deception_result['chunk_number'],
        "median_deception_text": median_deception_result['text']
    }
    return summary

def get_youtube_video_id(url):
    if "v=" in url:
        return url.split("v=")[-1]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1]
    else:
        raise ValueError("Invalid YouTube URL format.")

def download_transcript(video_id, language='en'):
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
    full_text = " ".join([item['text'] for item in transcript])
    return full_text

def chunk_text(text, chunk_size=500):
    tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    decoded_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
    return decoded_chunks

def perform_emotion_analysis(chunks):
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    all_results = []

    for i, chunk in enumerate(chunks):
        emotions = emotion_classifier(chunk)
        dominant_emotion = max(emotions[0], key=lambda x: x['score'])
        result = {
            "chunk_number": i + 1,
            "text": chunk,
            "emotions": emotions,
            "dominant_emotion": dominant_emotion['label'],
            "dominant_emotion_score": dominant_emotion['score']
        }
        all_results.append(result)

    return all_results

def detect_anomalies_knn(all_results, n_neighbors=5):
    emotion_scores = []
    for result in all_results:
        chunk_scores = [emotion['score'] for emotion in result['emotions'][0]]
        emotion_scores.append(chunk_scores)

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
    anomaly_labels = lof.fit_predict(emotion_scores)

    for i, result in enumerate(all_results):
        result['is_anomaly_knn'] = bool(anomaly_labels[i] == -1)

    return all_results

def detect_anomalies_isolation_forest(all_results):
    emotion_scores = []
    for result in all_results:
        chunk_scores = [emotion['score'] for emotion in result['emotions'][0]]
        emotion_scores.append(chunk_scores)

    iso_forest = IsolationForest(contamination=0.1)
    anomaly_labels = iso_forest.fit_predict(emotion_scores)

    for i, result in enumerate(all_results):
        result['is_anomaly_isolation_forest'] = bool(anomaly_labels[i] == -1)

    return all_results

def detect_anomalies_zscore(all_results, threshold=3):
    emotion_scores = []
    for result in all_results:
        chunk_scores = [emotion['score'] for emotion in result['emotions'][0]]
        emotion_scores.append(chunk_scores)

    z_scores = np.abs(zscore(emotion_scores))
    anomaly_labels = np.any(z_scores > threshold, axis=1)

    for i, result in enumerate(all_results):
        result['is_anomaly_zscore'] = bool(anomaly_labels[i])

    return all_results

def plot_dominant_emotions_bar(all_results):
    dominant_emotions = [result['dominant_emotion'] for result in all_results]
    emotion_counts = Counter(dominant_emotions)

    fig = px.bar(x=list(emotion_counts.keys()), y=list(emotion_counts.values()),
                 labels={'x': 'Emotion', 'y': 'Frequency'},
                 title="Distribution of Dominant Emotions Across Chunks",
                 template='plotly_dark')
    fig.show()

def plot_emotion_scores_bar_over_time(all_results):
    emotion_labels = list(set(emotion['label'] for result in all_results for emotion in result['emotions'][0]))
    chunk_numbers = [result['chunk_number'] for result in all_results]

    fig = go.Figure()

    for emotion in emotion_labels:
        scores = []
        for result in all_results:
            score = next((emotion_data['score'] for emotion_data in result['emotions'][0] if emotion_data['label'] == emotion), 0)
            scores.append(score)
        fig.add_trace(go.Bar(x=chunk_numbers, y=scores, name=emotion))

    fig.update_layout(barmode='group', xaxis_tickangle=-45, title="Emotion Scores Over Time (Grouped by Chunk)", template='plotly_dark')
    fig.show()

def plot_overall_emotion_distribution_pie(all_results):
    emotion_aggregates = Counter()

    for result in all_results:
        for emotion in result['emotions'][0]:
            emotion_aggregates[emotion['label']] += emotion['score']

    fig = px.pie(names=list(emotion_aggregates.keys()), values=list(emotion_aggregates.values()),
                 title="Overall Emotion Distribution in Transcript",
                 template='plotly_dark')
    fig.show()

def plot_anomalies_scatter(all_results, anomaly_type='knn'):
    chunk_numbers = [result['chunk_number'] for result in all_results]
    emotion_labels = list(set(emotion['label'] for result in all_results for emotion in result['emotions'][0]))

    fig = go.Figure()

    for emotion in emotion_labels:
        scores = []
        anomalies = []
        for result in all_results:
            score = next((emotion_data['score'] for emotion_data in result['emotions'][0] if emotion_data['label'] == emotion), 0)
            scores.append(score)
            anomalies.append(result[f'is_anomaly_{anomaly_type}'])

        fig.add_trace(go.Scatter(
            x=chunk_numbers, y=scores, mode='markers', name=f'{emotion} - Normal',
            marker=dict(color='blue'),
            hoverinfo='x+y'
        ))

        fig.add_trace(go.Scatter(
            x=[chunk_numbers[i] for i in range(len(anomalies)) if anomalies[i]],
            y=[scores[i] for i in range(len(anomalies)) if anomalies[i]],
            mode='markers', name=f'{emotion} - Anomaly',
            marker=dict(color='red', size=10, symbol='x'),
            hoverinfo='x+y'
        ))

    fig.update_layout(title=f"Anomalies in Emotion Scores ({anomaly_type.upper()})", xaxis_title="Chunk Number", yaxis_title="Emotion Score", template='plotly_dark')
    fig.show()

def normalize_results_for_json(all_results):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()  # Convert NumPy scalars to Python scalars
        if isinstance(item, dict):
            return {k: convert(v) for k, v in item.items()}
        if isinstance(item, list):
            return [convert(i) for i in item]
        return item

    return convert(all_results)

def calculate_emotional_inconsistency_score(all_results):
    """
    Calculates the Emotional Inconsistency (EI) Score for each chunk.
    The EI score measures the variability in emotions over time.
    """
    for i, result in enumerate(all_results):
        if i == 0:
            # No previous result to compare, so set EI score to 0
            result['ei_score'] = 0
        else:
            previous_emotions = all_results[i - 1]['emotions'][0]
            current_emotions = result['emotions'][0]

            # Calculate the sum of absolute differences between emotion scores
            ei_score = sum(abs(current_emotions[j]['score'] - previous_emotions[j]['score'])
                           for j in range(len(current_emotions)))

            result['ei_score'] = ei_score

    return all_results

if __name__ == "__main__":
    input_source = "https://www.youtube.com/watch?v=-BIDA_6t3VA"  # Replace with the actual input source
    source_type = "youtube"  # Choose from 'url', 'file', 'youtube'
    input_source = "https://www.cnn.com/2024/08/28/politics/supreme-court-biden-student-loan/index.html"
    source_type = "url"

    transcript = get_input_text(input_source, source_type)

    print("Splitting transcript into chunks...")
    chunks = chunk_text(transcript, chunk_size=40)

    print("Performing emotion analysis on each chunk...")
    all_results = perform_emotion_analysis(chunks)

    print("Calculating emotional baseline...")
    emotion_baseline = calculate_emotional_baseline(all_results)

    print("Calculating Emotional Baseline Deviation (EBD) Score...")
    all_results = calculate_ebd_score(all_results, emotion_baseline)

    print("Detecting anomalies using K-Nearest Neighbors...")
    all_results = detect_anomalies_knn(all_results)

    print("Detecting anomalies using Isolation Forest...")
    all_results = detect_anomalies_isolation_forest(all_results)

    print("Detecting anomalies using Z-Score...")
    all_results = detect_anomalies_zscore(all_results)

    print("Calculating Emotional Inconsistency (EI) Score...")
    all_results = calculate_emotional_inconsistency_score(all_results)

    print("Calculating Content-Emotion Correlation (CEC) Score...")
    all_results = calculate_cec_score(all_results)

    print("Calculating overall Deception Score...")
    all_results = calculate_deception_score(all_results)

    # Normalize for JSON
    all_results = normalize_results_for_json(all_results)

    # Example: Output the results as JSON if needed
    print(json.dumps(all_results, indent=4))

    # Generate Deception Summary
    deception_summary = generate_deception_summary(all_results)
    print(json.dumps(deception_summary, indent=4))

    print("Creating visualizations...")
    plot_dominant_emotions_bar(all_results)
    plot_emotion_scores_bar_over_time(all_results)
    plot_overall_emotion_distribution_pie(all_results)
    plot_anomalies_scatter(all_results, anomaly_type='knn')
    plot_anomalies_scatter(all_results, anomaly_type='isolation_forest')
    plot_anomalies_scatter(all_results, anomaly_type='zscore')

    # Plot deception score over time
    plot_deception_score_over_time(all_results)

    # Output deception score and summary in a formatted way
    print("\nDeception Summary:")
    print(f"{'Total Chunks Processed:':<30}\t{deception_summary['total_chunks_processed']}")
    print(f"{'Overall Deception Score:':<30}\t{deception_summary['overall_deception_score']:.4f}")
    print(f"{'Frequency of High Deception:':<30}\t{deception_summary['frequency_of_high_deception']}")
    print(f"{'Proportion of High Deception:':<30}\t{deception_summary['proportion_of_high_deception']:.2%}\n")

    print(f"{'Average Deception Score:':<30}\t{deception_summary['average_deception_score']:.4f}")
    print(f"{'Standard Deviation of Scores:':<30}\t{deception_summary['std_deception_score']:.4f}\n")

    print(f"{'Max Deception Score:':<30}\t{deception_summary['max_deception_score']:.4f}")
    print(f"{'Max Deception Chunk Number:':<30}\t{deception_summary['max_deception_chunk']}")
    print(f"{'Max Deception Text:':<30}\t{deception_summary['max_deception_text']}\n")

    print(f"{'Min Deception Score:':<30}\t{deception_summary['min_deception_score']:.4f}")
    print(f"{'Min Deception Chunk Number:':<30}\t{deception_summary['min_deception_chunk']}")
    print(f"{'Min Deception Text:':<30}\t{deception_summary['min_deception_text']}\n")

    print(f"{'Median Deception Score:':<30}\t{(deception_summary['min_deception_score'] + deception_summary['max_deception_score'])/2:.4f}")
    print(f"{'Median Deception Chunk Number:':<30}\t{deception_summary['median_deception_chunk']}")
    print(f"{'Median Deception Text:':<30}\t{deception_summary['median_deception_text']}\n")

