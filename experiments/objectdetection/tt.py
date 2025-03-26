#!/usr/bin/env python3
#
# tt.py


from transformers import pipeline
import warnings
import json
from typing import List, Dict
import re

warnings.simplefilter(action='ignore', category=FutureWarning)

def split_text_into_sentences(text: str) -> List[str]:
    """
    Splits the input text into a list of sentences based on punctuation, handling cases with no spaces after punctuation.
    :param text: The input text.
    :return: A list of sentences.
    """
    sentence_endings = re.compile(r'(?<=[.!?])(?=[A-Z]|\s|$)')
    sentences = sentence_endings.split(text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def analyze_emotions(text: str) -> List[Dict[str, str]]:
    """
    Processes each sentence in the text to extract dominant emotions.
    :param text: The input text.
    :return: A list of dictionaries containing sentences and their dominant emotions.
    """
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    sentiment_analyzer = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    sentences = split_text_into_sentences(text)
    
    analysis_results = []
    for sentence in sentences:
        emotion_outputs = classifier([sentence])
        sentiment_output = sentiment_analyzer(sentence)
        
        dominant_emotion = max(emotion_outputs[0], key=lambda x: x['score'])
        sentiment_label = sentiment_output[0]['label'].lower()
        
        analysis_results.append({
            "sentence": sentence,
            "dominant_emotion": dominant_emotion['label'],
            "emotion_score": dominant_emotion['score'],
            "sentiment": sentiment_label,
            "sentiment_score": sentiment_output[0]['score']
        })
    
    return analysis_results

def measure_intent(analysis_results: List[Dict[str, str]]) -> str:
    """
    Measures intent based on changes in emotion and content.
    :param analysis_results: The list of analyzed sentences with emotions and sentiments.
    :return: A string indicating the inferred intent.
    """
    intent_score = 0
    for i, result in enumerate(analysis_results):
        emotion = result['dominant_emotion']
        sentiment = result['sentiment']
        
        # Evaluate based on mismatches or consistent patterns
        if sentiment == 'positive' and emotion in ['anger', 'disgust', 'fear', 'sadness']:
            intent_score -= 1  # Negative emotion with positive content, may indicate hiding something
        elif sentiment == 'negative' and emotion in ['joy', 'surprise']:
            intent_score -= 1  # Positive emotion with negative content, may indicate deception
        elif sentiment == 'neutral' and emotion in ['anger', 'disgust', 'fear', 'sadness']:
            intent_score -= 1  # Negative emotion with neutral content, may indicate discomfort
        elif sentiment == 'neutral' and emotion in ['joy', 'surprise']:
            intent_score += 1  # Positive emotion with neutral content, could indicate honesty
        elif sentiment == 'negative' and emotion in ['anger', 'disgust', 'fear', 'sadness']:
            intent_score += 1  # Negative emotion with negative content, could indicate truthfulness
        elif sentiment == 'positive' and emotion in ['joy', 'surprise']:
            intent_score += 1  # Positive emotion with positive content, could indicate truthfulness
        
        # Additional checks can be added here to refine the intent measurement
    
    if intent_score > 2:
        return "Truthful"
    elif intent_score < -2:
        return "Deceptive"
    else:
        return "Uncertain"

# Example usage:
text = """ """
analysis_results = analyze_emotions(text)
intent = measure_intent(analysis_results)
print(json.dumps(analysis_results, indent=4))
print(f"Inferred intent: {intent}")

