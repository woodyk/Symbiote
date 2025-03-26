#!/usr/bin/env python3
#
# fake_news_checker.py

import re
import os
import requests
import json
import sys
import logging
import warnings
from bs4 import BeautifulSoup
from transformers import pipeline, GPT2TokenizerFast
from huggingface_hub import login
from typing import List, Dict

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)
api_token = os.getenv("HUGGINGFACE_API_KEY")
login(token=api_token)

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



def split_text_into_chunks(text: str, max_tokens: int = 500) -> List[str]:
    """
    Splits the input text into chunks of a specified maximum number of tokens.
    
    :param text: The input text to be split.
    :param max_tokens: The maximum number of tokens per chunk.
    :return: A list of text chunks, each containing up to `max_tokens` tokens.
    """

    # Initialize the tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Tokenize the entire text
    tokens = tokenizer.encode(text)
    
    # Split the tokens into chunks
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    
    # Decode each chunk back into text
    text_chunks = [tokenizer.decode(chunk) for chunk in chunks]
    
    return text_chunks

def download_text_from_url(url: str) -> str:
    """
    Downloads the text from a given URL.
    
    :param url: The URL of the webpage to download text from.
    :return: The extracted text as a string.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all paragraphs from the page
        paragraphs = soup.find_all('p')
        text = "\n".join([para.get_text() for para in paragraphs])
        
        return text
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while trying to download the text: {e}")
        return ""

def check_fake_news(text: str, model_name: str = "hamzab/roberta-fake-news-classification") -> List[dict]:
    """
    Checks if the provided text is classified as fake news.
    
    :param text: The text to be checked.
    :param model_name: The model name to use for fake news detection.
    :return: A list of results from the fake news detection model.
    """
    # Load the text-classification model
    clf = pipeline("text-classification", model=model_name, tokenizer=model_name)
    
    # Split the text into manageable chunks
    chunks = split_text_into_chunks(text)
    
    # Analyze each chunk
    results = []
    for chunk in chunks:
        result = clf(chunk)
        results.append(result)
    
    return results

if __name__ == "__main__":
    url = sys.argv[1]

    # Download text from the provided URL
    text = download_text_from_url(url)

    MODEL = "jy46604790/Fake-News-Bert-Detect"
    MODEL = "yanzcc/FakeNewsClassifier_Longformer"
    MODEL = "hamzab/roberta-fake-news-classification"
    MODEL = "vishalk4u/liar_binaryclassifier_bert_cased"
    MODEL = "Zain6699/intent-classifier-establish_credibility"
    MODEL = "armansakif/bengali-fake-news"
    MODEL = "eligapris/lie-detection-sentiment-analysis" # login needed
    MODEL = "openai-community/roberta-base-openai-detector" # detect if written by AI
    MODEL = "dlentr/lie_detection_distilbert"
    MODEL = "Giyaseddin/distilbert-base-cased-finetuned-fake-and-real-news-dataset"

    if text:
        # Check for fake news
        results = check_fake_news(text, model_name=MODEL)

        print(json.dumps(results, indent=4))
        #for i, result in enumerate(results):
        #    print(f"{result}")
    else:
        print("Failed to download or process the text.")

    analysis_results = analyze_emotions(text)
    intent = measure_intent(analysis_results)
    print(json.dumps(analysis_results, indent=4))
    print(f"Inferred intent: {intent}")
