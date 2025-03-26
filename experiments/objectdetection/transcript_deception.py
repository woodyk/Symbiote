#!/usr/bin/env python3
#
# transcript_deception.py

#nltk.download('averaged_perceptron_tagger_eng')

import os
import re
import json
import nltk
import numpy as np
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from transformers import pipeline
from sklearn.neighbors import LocalOutlierFactor
from youtube_transcript_api import YouTubeTranscriptApi

# Ensure you have the necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def preprocess_text(transcript):
    # Step 1: Sentence Tokenization
    sentences = sent_tokenize(transcript)

    processed_sentences = []
    for sentence in sentences:
        # Step 2: Text Normalization (lowercasing, removing punctuation)
        sentence = sentence.lower()
        sentence = re.sub(r'[^\w\s]', '', sentence)

        # Step 3: Word Tokenization
        words = word_tokenize(sentence)

        # Step 4: Remove Stop Words
        filtered_words = [word for word in words if word not in stopwords.words('english')]

        # Step 5: POS Tagging
        pos_tags = pos_tag(filtered_words)

        processed_sentences.append({
            "original_sentence": sentence,
            "filtered_words": filtered_words,
            "pos_tags": pos_tags
        })

    return processed_sentences

def perform_sentiment_analysis(sentences):
    sentiment_analyzer = pipeline("sentiment-analysis")
    emotions = [sentiment_analyzer(sentence['original_sentence']) for sentence in sentences]
    return emotions

def detect_emotional_anomalies(emotions, baseline_emotion):
    deviations = [abs(emotion[0]['score'] - baseline_emotion.get(emotion[0]['label'], 0)) for emotion in emotions]
    return deviations

def detect_contextual_anomalies(sentences, sentiments):
    lof = LocalOutlierFactor(n_neighbors=5)
    sentiment_scores = [sentiment[0]['score'] for sentiment in sentiments]
    anomaly_labels = lof.fit_predict(np.array(sentiment_scores).reshape(-1, 1))
    return anomaly_labels

def calculate_deception_score(emotions, anomalies, sentences):
    scores = []
    for emotion, anomaly, sentence in zip(emotions, anomalies, sentences):
        # Normalize the scores to fall between 0 and 1
        emotion_score = emotion[0]['score'] / 1.0  # Assuming sentiment score is already between 0 and 1
        anomaly_score = (anomaly + 1) / 2.0  # Convert anomaly score (-1 or 1) to 0 or 1

        # Combine the scores with equal weighting
        deception_score = (emotion_score + anomaly_score) / 2.0

        scores.append({
            "text": sentence['original_sentence'],  # Include the original text
            "emotion": emotion[0]['label'],
            "emotion_score": emotion_score,  # Already between 0 and 1
            "anomaly_score": anomaly_score,  # Now between 0 and 1
            "deception_score": deception_score  # Between 0 and 1
        })
    return scores

def analyze_transcript(transcript):
    sentences = preprocess_text(transcript)
    emotions = perform_sentiment_analysis(sentences)
    baseline_emotion = calculate_baseline_emotion(emotions)
    emotional_anomalies = detect_emotional_anomalies(emotions, baseline_emotion)
    contextual_anomalies = detect_contextual_anomalies(sentences, emotions)

    deception_scores = calculate_deception_score(emotions, contextual_anomalies, sentences)

    # Calculate summary statistics
    all_scores = [score["deception_score"] for score in deception_scores]
    avg_score = float(np.mean(all_scores))
    max_score = float(np.max(all_scores))
    min_score = float(np.min(all_scores))
    std_score = float(np.std(all_scores))

    return {
        "average_deception_score": avg_score,
        "max_deception_score": max_score,
        "min_deception_score": min_score,
        "std_deception_score": std_score,
        "detailed_scores": deception_scores,
        "summary": "Detailed summary of findings..."
    }

def calculate_baseline_emotion(emotions):
    emotion_scores = defaultdict(list)
    for emotion in emotions:
        emotion_scores[emotion[0]['label']].append(emotion[0]['score'])

    baseline_emotion = {emotion: float(np.mean(scores)) for emotion, scores in emotion_scores.items()}
    return baseline_emotion

def get_youtube_transcript(youtube_url):
    video_id = youtube_url.split("v=")[-1] if "v=" in youtube_url else youtube_url.split("/")[-1]
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = " ".join([item['text'] for item in transcript_list])
    return transcript

# Example usage
youtube_url = "https://www.youtube.com/watch?v=MyJHlhST62E"  # Replace with the actual YouTube video URL
youtube_url = "https://www.youtube.com/watch?v=iC-wRBsAhEs"
transcript = get_youtube_transcript(youtube_url)
deception_summary = analyze_transcript(transcript)
print(json.dumps(deception_summary, indent=4))

