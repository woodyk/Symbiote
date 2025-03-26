#!/usr/bin/env python3
#
# intent_detection_from_text.py

import os
import requests
import json
import logging
import math
import numpy as np
import re
from collections import defaultdict
from bs4 import BeautifulSoup
from transformers import pipeline, GPT2TokenizerFast, AutoTokenizer
from transformers import logging as hf_logging
from huggingface_hub import login
from typing import List, Dict
import warnings
import sys
import contextlib

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set transformers and other libraries logging level to ERROR
hf_logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# Suppress TensorFlow and other potential logs from deep learning libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Suppress stdout entirely using context manager
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

class IntentDetector:
    def __init__(self, api_key: str = None, tokenizers_parallelism: str = 'false'):
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if self.api_key:
            self._authenticate()
        os.environ["TOKENIZERS_PARALLELISM"] = tokenizers_parallelism

        # Initialize models
        with suppress_stdout():
            self.emotion_model = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
            self.sentiment_model = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            self.fake_news_model = pipeline("text-classification", model="hamzab/roberta-fake-news-classification")
            self.ai_detection_model = pipeline("text-classification", model="openai-community/roberta-base-openai-detector")
            self.tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
            self.max_length = self.tokenizer.model_max_length

    def _authenticate(self):
        with suppress_stdout():
            login(token=self.api_key, add_to_git_credential=False)

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyzes the text for fake news potential, AI-generated text potential, and emotions,
        and consolidates the results into a single deception score.
        """
        fake_news_score = self._process_text(text, self.fake_news_model, fake_label="FAKE")
        ai_detection_score = self._process_text(text, self.ai_detection_model, fake_label="AI-GENERATED")
        emotion_results = self.analyze_emotions(text)
        intent_results = self.measure_intent(emotion_results)

        # Combine the results into a single deception score
        deception_score = self._calculate_deception_score(fake_news_score, ai_detection_score, intent_results)

        return {
            "deception_score": deception_score,
            "fake_news_probability": fake_news_score,
            "ai_generated_probability": ai_detection_score,
            "intent_analysis": intent_results
        }

    def _process_text(self, text: str, model, fake_label: str) -> float:
        """
        Generic method to process text using a given model and calculate an average score.
        """
        chunks = self.split_text_into_chunks(text)
        scores = []
        for chunk in chunks:
            result = model(chunk)
            scores.append(result[0]['score'] if result[0]['label'] == fake_label else 1 - result[0]['score'])
        return np.mean(scores)

    def analyze_emotions(self, text: str) -> List[Dict[str, Dict[str, float]]]:
        sentences = self.split_text_into_sentences(text)
        analysis_results = []
        for sentence in sentences:
            emotion_outputs = self.emotion_model(sentence[:self.max_length])[0]
            sentiment_output = self.sentiment_model(sentence[:self.max_length])[0]

            emotion_scores = {emotion['label']: emotion['score'] for emotion in emotion_outputs}
            analysis_results.append({
                "sentence": sentence,
                "emotion_scores": emotion_scores,
                "sentiment": sentiment_output['label'].lower(),
                "sentiment_score": sentiment_output['score']
            })
        return analysis_results

    def measure_intent(self, analysis_results: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        intent_score = 100
        previous_sentiment = None
        previous_emotion = None

        for i, result in enumerate(analysis_results):
            dominant_emotion = self.get_dominant_emotion(result['emotion_scores'])
            dominant_emotion_score = result['emotion_scores'][dominant_emotion]
            sentiment = result['sentiment']
            entropy = self.calculate_entropy(result['emotion_scores'])

            intent_score = self.adjust_intent_score(
                intent_score, sentiment, dominant_emotion, dominant_emotion_score, entropy,
                i, previous_sentiment, previous_emotion, result
            )

            previous_sentiment = sentiment
            previous_emotion = dominant_emotion

        intent_score = self.apply_final_adjustments(intent_score, analysis_results)

        inferred_intent_score = round(intent_score / 100, 4)
        dominant_intent = "truthful" if inferred_intent_score >= 0.5 else "deceptive"

        return {
            "inferred_intent_score": inferred_intent_score,
            "dominant_intent": dominant_intent,
            "emotion_scores": self.aggregate_emotion_scores(analysis_results),
            "sentiment": self.determine_dominant_sentiment(analysis_results),
            "sentiment_stats": self.calculate_aggregated_sentiment_stats(analysis_results)
        }

    def _calculate_deception_score(self, fake_news_score: float, ai_detection_score: float, intent_results: Dict[str, float]) -> float:
        """
        Combine the results from fake news analysis, AI text detection, and intent analysis to calculate a final deception score.
        The final score is a weighted combination of these factors.
        """
        intent_score = intent_results["inferred_intent_score"]

        # A possible weighting scheme could be:
        weights = {
            'fake_news': 0.4,
            'ai_detection': 0.3,
            'intent': 0.3
        }

        combined_score = (weights['fake_news'] * fake_news_score +
                          weights['ai_detection'] * ai_detection_score +
                          weights['intent'] * (1 - intent_score))

        return round(combined_score, 4)

    def adjust_intent_score(self, intent_score, sentiment, dominant_emotion, dominant_emotion_score, entropy, i, previous_sentiment, previous_emotion, result):
        if sentiment == 'positive' and dominant_emotion in ['anger', 'disgust', 'fear', 'sadness', 'disapproval', 'realization']:
            intent_score -= 2.5 * dominant_emotion_score
        elif sentiment == 'negative' and dominant_emotion in ['joy', 'surprise', 'admiration', 'amusement', 'approval']:
            intent_score -= 2.5 * dominant_emotion_score
        elif sentiment == 'neutral' and dominant_emotion in ['anger', 'disgust', 'fear', 'sadness', 'disapproval', 'realization']:
            intent_score -= 2 * dominant_emotion_score
        elif sentiment == 'neutral' and dominant_emotion in ['joy', 'surprise', 'admiration', 'amusement', 'approval']:
            intent_score += 2 * dominant_emotion_score
        elif sentiment == 'negative' and dominant_emotion in ['anger', 'disgust', 'fear', 'sadness', 'disapproval', 'realization']:
            intent_score += 2.5 * dominant_emotion_score
        elif sentiment == 'positive' and dominant_emotion in ['joy', 'surprise', 'admiration', 'amusement', 'approval']:
            intent_score += 2.5 * dominant_emotion_score

        # Adjust based on emotional distribution (entropy)
        if entropy > 1.5:
            intent_score -= 1
        elif entropy < 0.5:
            intent_score += 1

        # Adjust based on narrow gaps between dominant and secondary emotions
        emotion_scores = sorted(result['emotion_scores'].values(), reverse=True)
        if len(emotion_scores) > 1:
            second_highest_score = emotion_scores[1]
            if dominant_emotion_score - second_highest_score < 0.1:
                intent_score -= 1

        # Adjust based on changes in sentiment between sentences
        if i > 0:
            if sentiment != previous_sentiment or dominant_emotion != previous_emotion:
                intent_score -= 1.0
            if sentiment == previous_sentiment and dominant_emotion == previous_emotion:
                intent_score += 0.5

        return intent_score

    def apply_final_adjustments(self, intent_score, analysis_results):
        sentiment_stats = self.calculate_aggregated_sentiment_stats(analysis_results)
        if sentiment_stats['positive']['average'] > 0.7 and sentiment_stats['negative']['average'] > 0.7:
            intent_score -= 0.5

        aggregated_emotions = self.aggregate_emotion_scores(analysis_results)
        if aggregated_emotions['neutral']['average'] > 0.75:
            if any(aggregated_emotions[emotion]['average'] > 0.05 for emotion in ['anger', 'disgust', 'sadness']):
                intent_score -= 0.75

        for emotion in aggregated_emotions:
            if aggregated_emotions[emotion]['average'] < 0.05 and aggregated_emotions[emotion]['std_dev'] > 0.15:
                intent_score -= 0.5

        return max(min(intent_score, 100), 0)

    def get_dominant_emotion(self, emotion_scores: Dict[str, float]) -> str:
        sorted_emotions = sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)
        if sorted_emotions[0][0] == 'neutral' and len(sorted_emotions) > 1:
            return sorted_emotions[1][0]
        return sorted_emotions[0][0]

    def calculate_entropy(self, emotion_scores: Dict[str, float]) -> float:
        total_score = sum(emotion_scores.values())
        if total_score == 0:
            return 0.0
        return -sum((score / total_score) * math.log2(score / total_score) for score in emotion_scores.values() if score > 0)

    def aggregate_emotion_scores(self, analysis_results: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        emotion_data = defaultdict(list)
        for result in analysis_results:
            for emotion, score in result['emotion_scores'].items():
                emotion_data[emotion].append(score)
        return {emotion: {"min": np.min(scores), "max": np.max(scores), "average": np.mean(scores), "std_dev": np.std(scores)} for emotion, scores in emotion_data.items()}

    def calculate_aggregated_sentiment_stats(self, analysis_results: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        sentiment_data = defaultdict(list)
        for result in analysis_results:
            sentiment_data[result['sentiment']].append(result['sentiment_score'])
        return {sentiment: {"min": np.min(scores), "max": np.max(scores), "average": np.mean(scores), "std_dev": np.std(scores)} for sentiment, scores in sentiment_data.items()}

    def determine_dominant_sentiment(self, analysis_results: List[Dict[str, Dict[str, float]]]) -> str:
        sentiment_weights = defaultdict(float)
        for result in analysis_results:
            sentiment_weights[result['sentiment']] += result['sentiment_score']
        return max(sentiment_weights, key=sentiment_weights.get)

    def split_text_into_sentences(self, text: str) -> List[str]:
        sentence_endings = re.compile(r'(?<=[.!?])(?=[A-Z]|\s|$)')
        return [sentence.strip() for sentence in sentence_endings.split(text) if sentence.strip()]

    def split_text_into_chunks(self, text: str, max_tokens: int = 500) -> List[str]:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokens = tokenizer.encode(text)
        return [tokenizer.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]

    def download_text_from_url(self, url: str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            return "\n".join([para.get_text() for para in paragraphs])
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while trying to download the text: {e}")
            return ""

# Example usage
if __name__ == "__main__":
    # Use environment variable if no API key is provided explicitly
    detector = IntentDetector()

    url = "https://www.cnn.com/2024/08/31/politics/harris-slams-trump-arlington-national-cemetery/index.html"
    url = "https://www.cnn.com/2024/08/31/middleeast/israel-recovers-bodies-gaza-intl-latam/index.html"
    url = "https://www.theguardian.com/commentisfree/2020/sep/08/robot-wrote-this-article-gpt-3"
    text = detector.download_text_from_url(url)

    if text:
        with suppress_stdout():
            result = detector.analyze_text(text)
        print(json.dumps(result, indent=4))
    else:
        print("Failed to download or process the text.")

