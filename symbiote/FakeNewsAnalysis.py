#!/usr/bin/env python3
#
# FakeNewsAnalysis.py

import os
import requests
import json
import logging
import math
import numpy as np
import re
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
from transformers import pipeline, GPT2TokenizerFast, AutoTokenizer
from transformers import logging as hf_logging
from huggingface_hub import login
from typing import List, Dict
import warnings
import sys
import contextlib
from textstat import textstat
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from rich.console import Console
console = Console()
print = console.print
log = console.log

# Download necessary NLTK data
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')

# Suppress all warnings and set logging to ERROR level
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
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

class FakeNewsDetector:
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
        ai_detection_score_model = self._process_text(text, self.ai_detection_model, fake_label="AI-GENERATED")

        # Enhanced AI Detection without models
        ai_features = self.extract_ai_features(text)
        heuristic_ai_score, ai_explanations = self.heuristic_ai_score(ai_features)

        # Combine model-based and heuristic AI scores
        combined_ai_score = (ai_detection_score_model * 0.5) + (heuristic_ai_score * 0.5)

        # Additional Heuristic Features for Fake News and Deception
        advanced_features = self.extract_advanced_features(text)
        advanced_score, advanced_explanations = self.heuristic_advanced_score(advanced_features)

        # Combine advanced heuristic scores with existing scores
        combined_fake_news_score = (fake_news_score * 0.7) + (advanced_score * 0.3)

        emotion_results = self.analyze_emotions(text)
        intent_results = self.measure_intent(emotion_results)
        complexity_score = self.analyze_text_complexity(text)

        # Combine the results into a single deception score
        deception_score, explanation = self._calculate_deception_score(
            combined_fake_news_score, combined_ai_score, intent_results, complexity_score, ai_explanations, advanced_explanations
        )

        return {
            "deception_score": deception_score,
            "fake_news_probability": combined_fake_news_score,
            "ai_generated_probability": combined_ai_score,
            "intent_analysis": intent_results,
            "complexity_score": complexity_score,
            "explanation": explanation
        }

    def _process_text(self, text: str, model, fake_label: str) -> float:
        """
        Generic method to process text using a given model and calculate an average score.
        """
        chunks = self.split_text_into_chunks(text)
        scores = []
        for chunk in chunks:
            result = model(chunk)
            if result[0]['label'] == fake_label:
                scores.append(result[0]['score'])  # Use the score as is for the fake label
            else:
                scores.append(1 - result[0]['score'])  # Invert the score for non-fake labels
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
        explanations = []
        previous_sentiment, previous_emotion = None, None

        for i, result in enumerate(analysis_results):
            dominant_emotion = self.get_dominant_emotion(result['emotion_scores'])
            entropy = self.calculate_entropy(result['emotion_scores'])

            intent_score, explanation = self.adjust_intent_score(
                intent_score, result['sentiment'], dominant_emotion, result['emotion_scores'][dominant_emotion],
                entropy, i, previous_sentiment, previous_emotion, result
            )
            explanations.extend(explanation)
            previous_sentiment, previous_emotion = result['sentiment'], dominant_emotion

        intent_score = self.apply_final_adjustments(intent_score, analysis_results)
        inferred_intent_score = round(intent_score / 100, 4)

        # Correcting the threshold for determining dominant intent
        dominant_intent = "deceptive" if inferred_intent_score > 0.5 else "truthful"

        return {
            "inferred_intent_score": inferred_intent_score,
            "dominant_intent": dominant_intent,
            "emotion_scores": self.aggregate_emotion_scores(analysis_results),
            "sentiment": self.determine_dominant_sentiment(analysis_results),
            "sentiment_stats": self.calculate_aggregated_sentiment_stats(analysis_results),
            "explanation": explanations
        }

    def _calculate_deception_score(self, fake_news_score: float, ai_detection_score: float, intent_results: Dict[str, float], complexity_score: float, ai_explanations: List[str], advanced_explanations: List[str]) -> (float, List[str]):
        """
        Combine all features, including new ones, to calculate a final deception score.
        """
        intent_score = intent_results["inferred_intent_score"]

        # Detailed explanation of the score calculation
        explanation = []

        # Fake News Analysis
        explanation.append(f"Fake News Analysis: The text was analyzed for potential fake news. The resulting score was {fake_news_score:.4f}, where 1 indicates a high likelihood of being fake and 0 indicates a high likelihood of being truthful.")

        # AI Writing Detection
        explanation.append(f"AI Writing Detection: The text was checked to see if it was generated by AI. The resulting combined AI score was {ai_detection_score:.4f}, combining model-based and heuristic analysis.")

        # Append heuristic AI explanations
        for ai_explanation in ai_explanations:
            explanation.append(f"AI Detection Adjustment: {ai_explanation}")

        # Advanced Heuristic Features Explanations
        for adv_explanation in advanced_explanations:
            explanation.append(f"Advanced Feature Adjustment: {adv_explanation}")

        # Intent Analysis
        explanation.append(f"Intent Analysis: The text's sentiment and emotion were analyzed to infer intent. The inferred intent score was {intent_score:.4f}, where 0 indicates truthful intent and 1 indicates deceptive intent.")

        # Text Complexity Analysis
        explanation.append(f"Text Complexity Analysis: The text was evaluated for complexity using readability metrics and sentence/word length. The complexity score was {complexity_score:.4f}, where higher values indicate higher complexity.")

        # Weighted combination
        combined_score = (0.35 * fake_news_score +
                          0.35 * ai_detection_score +
                          0.2 * intent_score +
                          0.1 * complexity_score)

        explanation.append(f"The final deception score is calculated by weighting the scores from each analysis: "
                           f"35% Fake News Analysis, 35% AI Writing Detection (enhanced), 20% Intent Analysis, and 10% Text Complexity. "
                           f"The final deception score is {combined_score:.4f}, where 1 indicates high likelihood of deception and 0 indicates high likelihood of truthfulness.")

        return round(combined_score, 4), explanation

    def adjust_intent_score(self, intent_score, sentiment, dominant_emotion, dominant_emotion_score, entropy, i, previous_sentiment, previous_emotion, result):
        explanation = []
        if sentiment == 'positive' and dominant_emotion in ['anger', 'disgust', 'fear', 'sadness', 'disapproval', 'realization']:
            intent_score -= 2.5 * dominant_emotion_score
            explanation.append(f"Adjusted down due to inconsistent sentiment ('{sentiment}') and emotion ('{dominant_emotion}').")
        elif sentiment == 'negative' and dominant_emotion in ['joy', 'surprise', 'admiration', 'amusement', 'approval']:
            intent_score -= 2.5 * dominant_emotion_score
            explanation.append(f"Adjusted down due to inconsistent sentiment ('{sentiment}') and emotion ('{dominant_emotion}').")
        elif sentiment == 'neutral' and dominant_emotion in ['anger', 'disgust', 'fear', 'sadness', 'disapproval', 'realization']:
            intent_score -= 2 * dominant_emotion_score
            explanation.append(f"Adjusted down due to neutral sentiment with negative emotion ('{dominant_emotion}').")
        elif sentiment == 'neutral' and dominant_emotion in ['joy', 'surprise', 'admiration', 'amusement', 'approval']:
            intent_score += 2 * dominant_emotion_score
            explanation.append(f"Adjusted up due to neutral sentiment with positive emotion ('{dominant_emotion}').")
        elif sentiment == 'negative' and dominant_emotion in ['anger', 'disgust', 'fear', 'sadness', 'disapproval', 'realization']:
            intent_score += 2.5 * dominant_emotion_score
            explanation.append(f"Adjusted up due to consistent negative sentiment and emotion ('{dominant_emotion}').")
        elif sentiment == 'positive' and dominant_emotion in ['joy', 'surprise', 'admiration', 'amusement', 'approval']:
            intent_score += 2.5 * dominant_emotion_score
            explanation.append(f"Adjusted up due to consistent positive sentiment and emotion ('{dominant_emotion}').")

        # Adjust based on emotional distribution (entropy)
        if entropy > 1.5:
            intent_score -= 1
            explanation.append("Adjusted down due to high entropy (emotional variability).")
        elif entropy < 0.5:
            intent_score += 1
            explanation.append("Adjusted up due to low entropy (emotional consistency).")

        # Adjust based on narrow gaps between dominant and secondary emotions
        emotion_scores = sorted(result['emotion_scores'].values(), reverse=True)
        if len(emotion_scores) > 1:
            second_highest_score = emotion_scores[1]
            if dominant_emotion_score - second_highest_score < 0.1:
                intent_score -= 1
                explanation.append("Adjusted down due to narrow gap between dominant and secondary emotions.")

        # Adjust based on changes in sentiment between sentences
        if i > 0:
            if sentiment != previous_sentiment or dominant_emotion != previous_emotion:
                intent_score -= 1.0
                explanation.append("Adjusted down due to changes in sentiment/emotion consistency.")
            if sentiment == previous_sentiment and dominant_emotion == previous_emotion:
                intent_score += 0.5
                explanation.append("Adjusted up due to consistent sentiment/emotion across sentences.")

        return intent_score, explanation

    def apply_final_adjustments(self, intent_score, analysis_results):
        sentiment_stats = self.calculate_aggregated_sentiment_stats(analysis_results)
        if sentiment_stats.get('positive', {}).get('average', 0) > 0.7 and sentiment_stats.get('negative', {}).get('average', 0) > 0.7:
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

        entropy = 0.0
        for score in emotion_scores.values():
            if score > 0:
                probability = score / total_score
                entropy_contribution = probability * math.log2(probability)
                entropy -= entropy_contribution

        return entropy

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
        return max(sentiment_weights, key=sentiment_weights.get) if sentiment_weights else "neutral"

    def analyze_text_complexity(self, text: str) -> float:
        """
        Analyzes the complexity of the text using readability metrics and average sentence/word length.
        """
        # Flesch-Kincaid readability score (higher values indicate higher complexity)
        fk_score = textstat.flesch_kincaid_grade(text)

        # Average sentence length (higher values indicate higher complexity)
        sentences = self.split_text_into_sentences(text)
        avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences) if sentences else 0

        # Average word length (higher values indicate higher complexity)
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        # Normalize the scores (example: using a 0-1 range)
        fk_score_normalized = np.clip((fk_score - 1) / 12, 0, 1)
        avg_sentence_length_normalized = np.clip(avg_sentence_length / 20, 0, 1)
        avg_word_length_normalized = np.clip(avg_word_length / 10, 0, 1)

        # Weighted average complexity score
        complexity_score = (0.4 * fk_score_normalized +
                            0.3 * avg_sentence_length_normalized +
                            0.3 * avg_word_length_normalized)

        return round(complexity_score, 4)

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
            log(f"An error occurred while trying to download the text: {e}")
            return ""

    # --- Enhanced AI Detection Methods ---

    # Linguistic Feature Extraction Methods
    def calculate_ttr(self, text: str) -> float:
        words = re.findall(r'\w+', text.lower())
        unique_words = set(words)
        return len(unique_words) / len(words) if words else 0

    def average_sentence_length_feature(self, sentences: List[str]) -> float:
        word_counts = [len(sentence.split()) for sentence in sentences]
        return np.mean(word_counts) if word_counts else 0

    def punctuation_density(self, text: str) -> float:
        punctuations = re.findall(r'[.!?,;:"\'-]', text)
        return len(punctuations) / len(text) if text else 0

    def ngram_repetition(self, text: str, n: int = 3) -> float:
        words = text.lower().split()
        ngrams = zip(*[words[i:] for i in range(n)])
        ngram_counts = Counter(ngrams)
        repetitions = sum(1 for count in ngram_counts.values() if count > 1)
        total_ngrams = sum(ngram_counts.values())  # Corrected line
        return repetitions / total_ngrams if total_ngrams else 0

    def shannon_entropy(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0
        counts = Counter(words)
        probabilities = [count / len(words) for count in counts.values()]
        return -sum(p * math.log2(p) for p in probabilities if p > 0)

    def pos_tag_distribution(self, text: str) -> Dict[str, float]:
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        chunks = nltk.ne_chunk(pos_tags, binary=False)
        tag_counts = defaultdict(int)
        for chunk in chunks:
            if isinstance(chunk, nltk.Tree):
                tag_counts[chunk.label()] += 1
            else:
                tag_counts[chunk[1]] += 1
        total = sum(tag_counts.values())
        return {tag: count / total for tag, count in tag_counts.items()} if total else {}

    def topic_consistency(self, text: str) -> float:
        # Simplistic approach: check if the same keywords/topics are present throughout
        sentences = self.split_text_into_sentences(text)
        if not sentences:
            return 0
        first_sentence_words = set(re.findall(r'\w+', sentences[0].lower()))
        if not first_sentence_words:
            return 0
        consistency_scores = []
        for sentence in sentences[1:]:
            words = set(re.findall(r'\w+', sentence.lower()))
            overlap = first_sentence_words.intersection(words)
            consistency_scores.append(len(overlap) / len(first_sentence_words) if first_sentence_words else 0)
        return np.mean(consistency_scores) if consistency_scores else 0

    def detect_contradictions(self, text: str) -> float:
        contradictory_pairs = [
            ("however", "therefore"),
            ("but", "so"),
            ("although", "thus"),
            ("nevertheless", "consequently"),
            ("despite", "because"),
            ("though", "yet")
        ]
        contradictions = 0
        for word1, word2 in contradictory_pairs:
            if word1 in text.lower() and word2 in text.lower():
                contradictions += 1
        return contradictions / len(contradictory_pairs) if contradictory_pairs else 0

    def personal_pronoun_ratio(self, text: str) -> float:
        personal_pronouns = ['i', 'we', 'my', 'our', 'us', 'me']
        words = re.findall(r'\w+', text.lower())
        pronoun_count = sum(1 for word in words if word in personal_pronouns)
        return pronoun_count / len(words) if words else 0

    # --- Advanced Heuristic Feature Methods ---

    # Additional Feature Extraction Methods
    def extract_advanced_features(self, text: str) -> Dict[str, float]:
        features = {}
        features['passive_voice_ratio'] = self.passive_voice_ratio(text)
        features['lexical_density'] = self.lexical_density(text)
        features['sentiment_shifts'] = self.sentiment_shift_analysis(self.analyze_emotions(text))
        features['named_entity_consistency'] = self.named_entity_consistency(text)
        features['sensationalist_language'] = self.sensationalist_language_score(text)
        features['logical_fallacies'] = self.logical_fallacy_score(text)
        features['loaded_language'] = self.loaded_language_score(text)
        return features

    # Heuristic Scoring for Advanced Features
    def heuristic_advanced_score(self, features: Dict[str, float]) -> (float, List[str]):
        # Assign weights to each advanced feature based on assumed importance
        weights = {
            'passive_voice_ratio': 0.10,
            'lexical_density': 0.10,
            'sentiment_shifts': 0.15,
            'named_entity_consistency': 0.15,
            'sensationalist_language': 0.20,
            'logical_fallacies': 0.15,
            'loaded_language': 0.15
        }

        # Initialize score and explanations
        score = 0
        explanations = []

        # Feature-based scoring
        score += weights['passive_voice_ratio'] * features.get('passive_voice_ratio', 0)
        explanations.append(f"Passive Voice Ratio contributes {weights['passive_voice_ratio']} * {features.get('passive_voice_ratio', 0):.4f} = {weights['passive_voice_ratio'] * features.get('passive_voice_ratio', 0):.4f} to Fake News score.")

        score += weights['lexical_density'] * features.get('lexical_density', 0)
        explanations.append(f"Lexical Density contributes {weights['lexical_density']} * {features.get('lexical_density', 0):.4f} = {weights['lexical_density'] * features.get('lexical_density', 0):.4f} to Fake News score.")

        score += weights['sentiment_shifts'] * features.get('sentiment_shifts', 0)
        explanations.append(f"Sentiment Shifts contributes {weights['sentiment_shifts']} * {features.get('sentiment_shifts', 0):.4f} = {weights['sentiment_shifts'] * features.get('sentiment_shifts', 0):.4f} to Fake News score.")

        score += weights['named_entity_consistency'] * features.get('named_entity_consistency', 0)
        explanations.append(f"Named Entity Consistency contributes {weights['named_entity_consistency']} * {features.get('named_entity_consistency', 0):.4f} = {weights['named_entity_consistency'] * features.get('named_entity_consistency', 0):.4f} to Fake News score.")

        score += weights['sensationalist_language'] * features.get('sensationalist_language', 0)
        explanations.append(f"Sensationalist Language contributes {weights['sensationalist_language']} * {features.get('sensationalist_language', 0):.4f} = {weights['sensationalist_language'] * features.get('sensationalist_language', 0):.4f} to Fake News score.")

        score += weights['logical_fallacies'] * features.get('logical_fallacies', 0)
        explanations.append(f"Logical Fallacies contributes {weights['logical_fallacies']} * {features.get('logical_fallacies', 0):.4f} = {weights['logical_fallacies'] * features.get('logical_fallacies', 0):.4f} to Fake News score.")

        score += weights['loaded_language'] * features.get('loaded_language', 0)
        explanations.append(f"Loaded Language contributes {weights['loaded_language']} * {features.get('loaded_language', 0):.4f} = {weights['loaded_language'] * features.get('loaded_language', 0):.4f} to Fake News score.")

        # Normalize the score to 0-1
        score = np.clip(score, 0, 1)

        explanations.append(f"Heuristic Advanced Fake News Score (normalized to 0-1): {score:.4f}")

        return score, explanations

    # --- End of Enhanced AI Detection Methods ---

    # --- Additional Heuristic Feature Methods ---

    # Passive Voice Ratio
    def passive_voice_ratio(self, text: str) -> float:
        sentences = nltk.sent_tokenize(text)
        passive_count = 0
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            tags = nltk.pos_tag(tokens)
            chunks = nltk.ne_chunk(tags, binary=False)
            # Simple heuristic: look for forms of 'to be' + past participle (VBN)
            for i in range(len(tags)-1):
                if tags[i][0].lower() in ['is', 'are', 'was', 'were', 'be', 'been', 'being'] and tags[i+1][1] == 'VBN':
                    passive_count += 1
                    break
        return passive_count / len(sentences) if sentences else 0

    # Lexical Density
    def lexical_density(self, text: str) -> float:
        tokens = nltk.word_tokenize(text)
        tags = nltk.pos_tag(tokens)
        content_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                       'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
        content_words = [word for word, tag in tags if tag in content_tags]
        return len(content_words) / len(tokens) if tokens else 0

    # Sentiment Shift Analysis
    def sentiment_shift_analysis(self, emotion_results: List[Dict[str, Dict[str, float]]]) -> float:
        sentiment_scores = [result['sentiment_score'] if result['sentiment'] == 'positive' else -result['sentiment_score'] for result in emotion_results]
        if len(sentiment_scores) < 2:
            return 0.0
        shifts = 0
        for i in range(1, len(sentiment_scores)):
            if abs(sentiment_scores[i] - sentiment_scores[i-1]) > 0.5:  # Threshold can be adjusted
                shifts += 1
        return shifts / (len(sentiment_scores) - 1)

    # Named Entity Consistency
    def named_entity_consistency(self, text: str) -> float:
        sentences = nltk.sent_tokenize(text)
        entities = []
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            tags = pos_tag(tokens)
            chunks = nltk.ne_chunk(tags, binary=False)
            for chunk in chunks:
                if isinstance(chunk, nltk.Tree):
                    entity = ' '.join(c[0] for c in chunk.leaves())
                    entities.append((chunk.label(), entity))
        if not entities:
            return 1.0  # No entities to evaluate
        # Count occurrences of each entity
        entity_counts = Counter(entities)
        max_count = max(entity_counts.values())
        consistency = max_count / len(sentences) if len(sentences) else 1.0
        return consistency

    # Sensationalist Language Score
    def sensationalist_language_score(self, text: str) -> float:
        sensationalist_keywords = [
            "shocking", "unbelievable", "incredible", "amazing", "breaking",
            "exclusive", "alert", "urgent", "crazy", "disastrous", "horrific",
            "catastrophic", "explosive", "deadly", "terrifying", "massive"
        ]
        words = re.findall(r'\w+', text.lower())
        count = sum(1 for word in words if word in sensationalist_keywords)
        return count / len(words) if words else 0

    # Logical Fallacy Score
    def logical_fallacy_score(self, text: str) -> float:
        fallacy_phrases = [
            "you too", "if you don't believe me", "everyone knows",
            "nobody cares", "isn't it obvious", "the fact that",
            "without a doubt", "undeniably", "clearly", "obviously",
            "the problem is", "it's simple", "the only reason"
        ]
        text_lower = text.lower()
        count = sum(text_lower.count(phrase) for phrase in fallacy_phrases)
        return count / len(text_lower.split()) if text_lower.split() else 0

    # Loaded Language Score
    def loaded_language_score(self, text: str) -> float:
        loaded_words = [
            "disgusting", "horrible", "fantastic", "terrible", "wonderful",
            "hate", "love", "kill", "destroy", "fraud", "betrayal",
            "genocide", "catastrophe", "miracle", "extremist", "radical",
            "awful", "brilliant", "hideous", "fabulous", "appalling"
        ]
        words = re.findall(r'\w+', text.lower())
        count = sum(1 for word in words if word in loaded_words)
        return count / len(words) if words else 0

    # --- End of Additional Heuristic Feature Methods ---

    # --- Advanced Fake News Heuristic Scoring ---

    def heuristic_advanced_score(self, features: Dict[str, float]) -> (float, List[str]):
        # Assign weights to each advanced feature based on assumed importance
        weights = {
            'passive_voice_ratio': 0.10,
            'lexical_density': 0.10,
            'sentiment_shifts': 0.15,
            'named_entity_consistency': 0.15,
            'sensationalist_language': 0.20,
            'logical_fallacies': 0.15,
            'loaded_language': 0.15
        }

        # Initialize score and explanations
        score = 0
        explanations = []

        # Feature-based scoring
        score += weights['passive_voice_ratio'] * features.get('passive_voice_ratio', 0)
        explanations.append(f"Passive Voice Ratio contributes {weights['passive_voice_ratio']} * {features.get('passive_voice_ratio', 0):.4f} = {weights['passive_voice_ratio'] * features.get('passive_voice_ratio', 0):.4f} to Fake News score.")

        score += weights['lexical_density'] * features.get('lexical_density', 0)
        explanations.append(f"Lexical Density contributes {weights['lexical_density']} * {features.get('lexical_density', 0):.4f} = {weights['lexical_density'] * features.get('lexical_density', 0):.4f} to Fake News score.")

        score += weights['sentiment_shifts'] * features.get('sentiment_shifts', 0)
        explanations.append(f"Sentiment Shifts contributes {weights['sentiment_shifts']} * {features.get('sentiment_shifts', 0):.4f} = {weights['sentiment_shifts'] * features.get('sentiment_shifts', 0):.4f} to Fake News score.")

        score += weights['named_entity_consistency'] * features.get('named_entity_consistency', 0)
        explanations.append(f"Named Entity Consistency contributes {weights['named_entity_consistency']} * {features.get('named_entity_consistency', 0):.4f} = {weights['named_entity_consistency'] * features.get('named_entity_consistency', 0):.4f} to Fake News score.")

        score += weights['sensationalist_language'] * features.get('sensationalist_language', 0)
        explanations.append(f"Sensationalist Language contributes {weights['sensationalist_language']} * {features.get('sensationalist_language', 0):.4f} = {weights['sensationalist_language'] * features.get('sensationalist_language', 0):.4f} to Fake News score.")

        score += weights['logical_fallacies'] * features.get('logical_fallacies', 0)
        explanations.append(f"Logical Fallacies contributes {weights['logical_fallacies']} * {features.get('logical_fallacies', 0):.4f} = {weights['logical_fallacies'] * features.get('logical_fallacies', 0):.4f} to Fake News score.")

        score += weights['loaded_language'] * features.get('loaded_language', 0)
        explanations.append(f"Loaded Language contributes {weights['loaded_language']} * {features.get('loaded_language', 0):.4f} = {weights['loaded_language'] * features.get('loaded_language', 0):.4f} to Fake News score.")

        # Normalize the score to 0-1
        score = np.clip(score, 0, 1)

        explanations.append(f"Heuristic Advanced Fake News Score (normalized to 0-1): {score:.4f}")

        return score, explanations

    # --- End of Advanced Fake News Heuristic Scoring ---

    # --- Enhanced AI Detection Methods ---

    # Linguistic Feature Extraction Methods
    def calculate_ttr(self, text: str) -> float:
        words = re.findall(r'\w+', text.lower())
        unique_words = set(words)
        return len(unique_words) / len(words) if words else 0

    def average_sentence_length_feature(self, sentences: List[str]) -> float:
        word_counts = [len(sentence.split()) for sentence in sentences]
        return np.mean(word_counts) if word_counts else 0

    def punctuation_density(self, text: str) -> float:
        punctuations = re.findall(r'[.!?,;:"\'-]', text)
        return len(punctuations) / len(text) if text else 0

    def ngram_repetition(self, text: str, n: int = 3) -> float:
        words = text.lower().split()
        ngrams = zip(*[words[i:] for i in range(n)])
        ngram_counts = Counter(ngrams)
        repetitions = sum(1 for count in ngram_counts.values() if count > 1)
        total_ngrams = sum(ngram_counts.values())  # Corrected line
        return repetitions / total_ngrams if total_ngrams else 0

    def shannon_entropy(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0
        counts = Counter(words)
        probabilities = [count / len(words) for count in counts.values()]
        return -sum(p * math.log2(p) for p in probabilities if p > 0)

    def pos_tag_distribution(self, text: str) -> Dict[str, float]:
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        chunks = nltk.ne_chunk(pos_tags, binary=False)
        tag_counts = defaultdict(int)
        for chunk in chunks:
            if isinstance(chunk, nltk.Tree):
                entity = ' '.join(c[0] for c in chunk.leaves())
                tag_counts[chunk.label()] += 1
            else:
                tag_counts[chunk[1]] += 1
        total = sum(tag_counts.values())
        return {tag: count / total for tag, count in tag_counts.items()} if total else {}

    def topic_consistency(self, text: str) -> float:
        # Simplistic approach: check if the same keywords/topics are present throughout
        sentences = self.split_text_into_sentences(text)
        if not sentences:
            return 0
        first_sentence_words = set(re.findall(r'\w+', sentences[0].lower()))
        if not first_sentence_words:
            return 0
        consistency_scores = []
        for sentence in sentences[1:]:
            words = set(re.findall(r'\w+', sentence.lower()))
            overlap = first_sentence_words.intersection(words)
            consistency_scores.append(len(overlap) / len(first_sentence_words) if first_sentence_words else 0)
        return np.mean(consistency_scores) if consistency_scores else 0

    def detect_contradictions(self, text: str) -> float:
        contradictory_pairs = [
            ("however", "therefore"),
            ("but", "so"),
            ("although", "thus"),
            ("nevertheless", "consequently"),
            ("despite", "because"),
            ("though", "yet")
        ]
        contradictions = 0
        for word1, word2 in contradictory_pairs:
            if word1 in text.lower() and word2 in text.lower():
                contradictions += 1
        return contradictions / len(contradictory_pairs) if contradictory_pairs else 0

    def personal_pronoun_ratio(self, text: str) -> float:
        personal_pronouns = ['i', 'we', 'my', 'our', 'us', 'me']
        words = re.findall(r'\w+', text.lower())
        pronoun_count = sum(1 for word in words if word in personal_pronouns)
        return pronoun_count / len(words) if words else 0

    # Feature Extraction for AI Detection
    def extract_ai_features(self, text: str) -> Dict[str, float]:
        sentences = self.split_text_into_sentences(text)
        features = {}
        features['ttr'] = self.calculate_ttr(text)
        features['avg_sentence_length'] = self.average_sentence_length_feature(sentences)
        features['punctuation_density'] = self.punctuation_density(text)
        features['ngram_repetition'] = self.ngram_repetition(text, n=3)
        features['shannon_entropy'] = self.shannon_entropy(text)
        pos_dist = self.pos_tag_distribution(text)
        # Simplify POS distribution by focusing on noun and verb ratios
        noun_ratio = pos_dist.get('NN', 0) + pos_dist.get('NNS', 0)
        verb_ratio = pos_dist.get('VB', 0) + pos_dist.get('VBD', 0) + pos_dist.get('VBG', 0) + pos_dist.get('VBN', 0) + pos_dist.get('VBP', 0) + pos_dist.get('VBZ', 0)
        features['noun_ratio'] = noun_ratio
        features['verb_ratio'] = verb_ratio
        features['topic_consistency'] = self.topic_consistency(text)
        features['contradictions'] = self.detect_contradictions(text)
        features['personal_pronoun_ratio'] = self.personal_pronoun_ratio(text)
        return features

    # Heuristic Scoring for AI Detection
    def heuristic_ai_score(self, features: Dict[str, float]) -> (float, List[str]):
        # Assign weights to each feature based on assumed importance
        weights = {
            'ttr': 0.15,
            'avg_sentence_length': 0.15,
            'punctuation_density': 0.10,
            'ngram_repetition': 0.15,
            'shannon_entropy': 0.15,
            'noun_ratio': 0.10,
            'verb_ratio': 0.10,
            'topic_consistency': 0.05,
            'contradictions': 0.03,
            'personal_pronoun_ratio': 0.02
        }

        # Initialize score and explanations
        score = 0
        explanations = []

        # Feature-based scoring
        score += weights['ttr'] * features.get('ttr', 0)
        explanations.append(f"TTR (Type-Token Ratio) contributes {weights['ttr']} * {features.get('ttr', 0):.4f} = {weights['ttr'] * features.get('ttr', 0):.4f} to AI score.")

        # Normalize avg_sentence_length assuming 30 words as benchmark for high complexity
        normalized_avg_sentence_length = features.get('avg_sentence_length', 0) / 30
        normalized_avg_sentence_length = np.clip(normalized_avg_sentence_length, 0, 1)
        score += weights['avg_sentence_length'] * normalized_avg_sentence_length
        explanations.append(f"Average Sentence Length contributes {weights['avg_sentence_length']} * {normalized_avg_sentence_length:.4f} = {weights['avg_sentence_length'] * normalized_avg_sentence_length:.4f} to AI score.")

        score += weights['punctuation_density'] * features.get('punctuation_density', 0)
        explanations.append(f"Punctuation Density contributes {weights['punctuation_density']} * {features.get('punctuation_density', 0):.4f} = {weights['punctuation_density'] * features.get('punctuation_density', 0):.4f} to AI score.")

        score += weights['ngram_repetition'] * features.get('ngram_repetition', 0)
        explanations.append(f"N-gram Repetition contributes {weights['ngram_repetition']} * {features.get('ngram_repetition', 0):.4f} = {weights['ngram_repetition'] * features.get('ngram_repetition', 0):.4f} to AI score.")

        # Normalize shannon_entropy assuming a maximum of 10
        normalized_shannon_entropy = features.get('shannon_entropy', 0) / 10
        normalized_shannon_entropy = np.clip(normalized_shannon_entropy, 0, 1)
        score += weights['shannon_entropy'] * normalized_shannon_entropy
        explanations.append(f"Shannon Entropy contributes {weights['shannon_entropy']} * {normalized_shannon_entropy:.4f} = {weights['shannon_entropy'] * normalized_shannon_entropy:.4f} to AI score.")

        # Noun and Verb ratios
        score += weights['noun_ratio'] * features.get('noun_ratio', 0)
        explanations.append(f"Noun Ratio contributes {weights['noun_ratio']} * {features.get('noun_ratio', 0):.4f} = {weights['noun_ratio'] * features.get('noun_ratio', 0):.4f} to AI score.")

        score += weights['verb_ratio'] * features.get('verb_ratio', 0)
        explanations.append(f"Verb Ratio contributes {weights['verb_ratio']} * {features.get('verb_ratio', 0):.4f} = {weights['verb_ratio'] * features.get('verb_ratio', 0):.4f} to AI score.")

        score += weights['topic_consistency'] * features.get('topic_consistency', 0)
        explanations.append(f"Topic Consistency contributes {weights['topic_consistency']} * {features.get('topic_consistency', 0):.4f} = {weights['topic_consistency'] * features.get('topic_consistency', 0):.4f} to AI score.")

        score += weights['contradictions'] * features.get('contradictions', 0)
        explanations.append(f"Contradictions contributes {weights['contradictions']} * {features.get('contradictions', 0):.4f} = {weights['contradictions'] * features.get('contradictions', 0):.4f} to AI score.")

        # Less personal pronouns may indicate AI
        pronoun_score = (1 - features.get('personal_pronoun_ratio', 0))  # Higher if fewer pronouns
        score += weights['personal_pronoun_ratio'] * pronoun_score
        explanations.append(f"Personal Pronoun Ratio contributes {weights['personal_pronoun_ratio']} * {pronoun_score:.4f} = {weights['personal_pronoun_ratio'] * pronoun_score:.4f} to AI score.")

        # Ensure the score is between 0 and 1
        score = np.clip(score, 0, 1)

        explanations.append(f"Heuristic AI Score (normalized to 0-1): {score:.4f}")

        return score, explanations

    # --- End of Enhanced AI Detection Methods ---

    # --- Advanced Heuristic Feature Methods ---

    # Additional Feature Extraction Methods
    def extract_advanced_features(self, text: str) -> Dict[str, float]:
        features = {}
        features['passive_voice_ratio'] = self.passive_voice_ratio(text)
        features['lexical_density'] = self.lexical_density(text)
        features['sentiment_shifts'] = self.sentiment_shift_analysis(self.analyze_emotions(text))
        features['named_entity_consistency'] = self.named_entity_consistency(text)
        features['sensationalist_language'] = self.sensationalist_language_score(text)
        features['logical_fallacies'] = self.logical_fallacy_score(text)
        features['loaded_language'] = self.loaded_language_score(text)
        return features

    # Heuristic Scoring for Advanced Features
    def heuristic_advanced_score(self, features: Dict[str, float]) -> (float, List[str]):
        # Assign weights to each advanced feature based on assumed importance
        weights = {
            'passive_voice_ratio': 0.10,
            'lexical_density': 0.10,
            'sentiment_shifts': 0.15,
            'named_entity_consistency': 0.15,
            'sensationalist_language': 0.20,
            'logical_fallacies': 0.15,
            'loaded_language': 0.15
        }

        # Initialize score and explanations
        score = 0
        explanations = []

        # Feature-based scoring
        score += weights['passive_voice_ratio'] * features.get('passive_voice_ratio', 0)
        explanations.append(f"Passive Voice Ratio contributes {weights['passive_voice_ratio']} * {features.get('passive_voice_ratio', 0):.4f} = {weights['passive_voice_ratio'] * features.get('passive_voice_ratio', 0):.4f} to Fake News score.")

        score += weights['lexical_density'] * features.get('lexical_density', 0)
        explanations.append(f"Lexical Density contributes {weights['lexical_density']} * {features.get('lexical_density', 0):.4f} = {weights['lexical_density'] * features.get('lexical_density', 0):.4f} to Fake News score.")

        score += weights['sentiment_shifts'] * features.get('sentiment_shifts', 0)
        explanations.append(f"Sentiment Shifts contributes {weights['sentiment_shifts']} * {features.get('sentiment_shifts', 0):.4f} = {weights['sentiment_shifts'] * features.get('sentiment_shifts', 0):.4f} to Fake News score.")

        score += weights['named_entity_consistency'] * features.get('named_entity_consistency', 0)
        explanations.append(f"Named Entity Consistency contributes {weights['named_entity_consistency']} * {features.get('named_entity_consistency', 0):.4f} = {weights['named_entity_consistency'] * features.get('named_entity_consistency', 0):.4f} to Fake News score.")

        score += weights['sensationalist_language'] * features.get('sensationalist_language', 0)
        explanations.append(f"Sensationalist Language contributes {weights['sensationalist_language']} * {features.get('sensationalist_language', 0):.4f} = {weights['sensationalist_language'] * features.get('sensationalist_language', 0):.4f} to Fake News score.")

        score += weights['logical_fallacies'] * features.get('logical_fallacies', 0)
        explanations.append(f"Logical Fallacies contributes {weights['logical_fallacies']} * {features.get('logical_fallacies', 0):.4f} = {weights['logical_fallacies'] * features.get('logical_fallacies', 0):.4f} to Fake News score.")

        score += weights['loaded_language'] * features.get('loaded_language', 0)
        explanations.append(f"Loaded Language contributes {weights['loaded_language']} * {features.get('loaded_language', 0):.4f} = {weights['loaded_language'] * features.get('loaded_language', 0):.4f} to Fake News score.")

        # Normalize the score to 0-1
        score = np.clip(score, 0, 1)

        explanations.append(f"Heuristic Advanced Fake News Score (normalized to 0-1): {score:.4f}")

        return score, explanations

    # --- End of Advanced Heuristic Feature Methods ---

    # --- Enhanced AI Detection Methods ---

    # Linguistic Feature Extraction Methods
    def calculate_ttr(self, text: str) -> float:
        words = re.findall(r'\w+', text.lower())
        unique_words = set(words)
        return len(unique_words) / len(words) if words else 0

    def average_sentence_length_feature(self, sentences: List[str]) -> float:
        word_counts = [len(sentence.split()) for sentence in sentences]
        return np.mean(word_counts) if word_counts else 0

    def punctuation_density(self, text: str) -> float:
        punctuations = re.findall(r'[.!?,;:"\'-]', text)
        return len(punctuations) / len(text) if text else 0

    def ngram_repetition(self, text: str, n: int = 3) -> float:
        words = text.lower().split()
        ngrams = zip(*[words[i:] for i in range(n)])
        ngram_counts = Counter(ngrams)
        repetitions = sum(1 for count in ngram_counts.values() if count > 1)
        total_ngrams = sum(ngram_counts.values())  # Corrected line
        return repetitions / total_ngrams if total_ngrams else 0

    def shannon_entropy(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0
        counts = Counter(words)
        probabilities = [count / len(words) for count in counts.values()]
        return -sum(p * math.log2(p) for p in probabilities if p > 0)

    def pos_tag_distribution(self, text: str) -> Dict[str, float]:
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        tag_counts = defaultdict(int)
        for _, tag in pos_tags:
            tag_counts[tag] += 1
        total = sum(tag_counts.values())
        return {tag: count / total for tag, count in tag_counts.items()} if total else {}

    def topic_consistency(self, text: str) -> float:
        # Simplistic approach: check if the same keywords/topics are present throughout
        sentences = self.split_text_into_sentences(text)
        if not sentences:
            return 0
        first_sentence_words = set(re.findall(r'\w+', sentences[0].lower()))
        if not first_sentence_words:
            return 0
        consistency_scores = []
        for sentence in sentences[1:]:
            words = set(re.findall(r'\w+', sentence.lower()))
            overlap = first_sentence_words.intersection(words)
            consistency_scores.append(len(overlap) / len(first_sentence_words) if first_sentence_words else 0)
        return np.mean(consistency_scores) if consistency_scores else 0

    def detect_contradictions(self, text: str) -> float:
        contradictory_pairs = [
            ("however", "therefore"),
            ("but", "so"),
            ("although", "thus"),
            ("nevertheless", "consequently")
        ]
        contradictions = 0
        for word1, word2 in contradictory_pairs:
            if word1 in text.lower() and word2 in text.lower():
                contradictions += 1
        return contradictions / len(contradictory_pairs) if contradictory_pairs else 0

    def personal_pronoun_ratio(self, text: str) -> float:
        personal_pronouns = ['i', 'we', 'my', 'our', 'us', 'me']
        words = re.findall(r'\w+', text.lower())
        pronoun_count = sum(1 for word in words if word in personal_pronouns)
        return pronoun_count / len(words) if words else 0

    # Feature Extraction for AI Detection
    def extract_ai_features(self, text: str) -> Dict[str, float]:
        sentences = self.split_text_into_sentences(text)
        features = {}
        features['ttr'] = self.calculate_ttr(text)
        features['avg_sentence_length'] = self.average_sentence_length_feature(sentences)
        features['punctuation_density'] = self.punctuation_density(text)
        features['ngram_repetition'] = self.ngram_repetition(text, n=3)
        features['shannon_entropy'] = self.shannon_entropy(text)
        pos_dist = self.pos_tag_distribution(text)
        # Simplify POS distribution by focusing on noun and verb ratios
        noun_ratio = pos_dist.get('NN', 0) + pos_dist.get('NNS', 0)
        verb_ratio = pos_dist.get('VB', 0) + pos_dist.get('VBD', 0) + pos_dist.get('VBG', 0) + pos_dist.get('VBN', 0) + pos_dist.get('VBP', 0) + pos_dist.get('VBZ', 0)
        features['noun_ratio'] = noun_ratio
        features['verb_ratio'] = verb_ratio
        features['topic_consistency'] = self.topic_consistency(text)
        features['contradictions'] = self.detect_contradictions(text)
        features['personal_pronoun_ratio'] = self.personal_pronoun_ratio(text)
        return features

    # Heuristic Scoring for AI Detection
    def heuristic_ai_score(self, features: Dict[str, float]) -> (float, List[str]):
        # Assign weights to each feature based on assumed importance
        weights = {
            'ttr': 0.15,
            'avg_sentence_length': 0.15,
            'punctuation_density': 0.10,
            'ngram_repetition': 0.15,
            'shannon_entropy': 0.15,
            'noun_ratio': 0.10,
            'verb_ratio': 0.10,
            'topic_consistency': 0.05,
            'contradictions': 0.03,
            'personal_pronoun_ratio': 0.02
        }

        # Initialize score and explanations
        score = 0
        explanations = []

        # Feature-based scoring
        score += weights['ttr'] * features.get('ttr', 0)
        explanations.append(f"TTR (Type-Token Ratio) contributes {weights['ttr']} * {features.get('ttr', 0):.4f} = {weights['ttr'] * features.get('ttr', 0):.4f} to AI score.")

        # Normalize avg_sentence_length assuming 30 words as benchmark for high complexity
        normalized_avg_sentence_length = features.get('avg_sentence_length', 0) / 30
        normalized_avg_sentence_length = np.clip(normalized_avg_sentence_length, 0, 1)
        score += weights['avg_sentence_length'] * normalized_avg_sentence_length
        explanations.append(f"Average Sentence Length contributes {weights['avg_sentence_length']} * {normalized_avg_sentence_length:.4f} = {weights['avg_sentence_length'] * normalized_avg_sentence_length:.4f} to AI score.")

        score += weights['punctuation_density'] * features.get('punctuation_density', 0)
        explanations.append(f"Punctuation Density contributes {weights['punctuation_density']} * {features.get('punctuation_density', 0):.4f} = {weights['punctuation_density'] * features.get('punctuation_density', 0):.4f} to AI score.")

        score += weights['ngram_repetition'] * features.get('ngram_repetition', 0)
        explanations.append(f"N-gram Repetition contributes {weights['ngram_repetition']} * {features.get('ngram_repetition', 0):.4f} = {weights['ngram_repetition'] * features.get('ngram_repetition', 0):.4f} to AI score.")

        # Normalize shannon_entropy assuming a maximum of 10
        normalized_shannon_entropy = features.get('shannon_entropy', 0) / 10
        normalized_shannon_entropy = np.clip(normalized_shannon_entropy, 0, 1)
        score += weights['shannon_entropy'] * normalized_shannon_entropy
        explanations.append(f"Shannon Entropy contributes {weights['shannon_entropy']} * {normalized_shannon_entropy:.4f} = {weights['shannon_entropy'] * normalized_shannon_entropy:.4f} to AI score.")

        # Noun and Verb ratios
        score += weights['noun_ratio'] * features.get('noun_ratio', 0)
        explanations.append(f"Noun Ratio contributes {weights['noun_ratio']} * {features.get('noun_ratio', 0):.4f} = {weights['noun_ratio'] * features.get('noun_ratio', 0):.4f} to AI score.")

        score += weights['verb_ratio'] * features.get('verb_ratio', 0)
        explanations.append(f"Verb Ratio contributes {weights['verb_ratio']} * {features.get('verb_ratio', 0):.4f} = {weights['verb_ratio'] * features.get('verb_ratio', 0):.4f} to AI score.")

        score += weights['topic_consistency'] * features.get('topic_consistency', 0)
        explanations.append(f"Topic Consistency contributes {weights['topic_consistency']} * {features.get('topic_consistency', 0):.4f} = {weights['topic_consistency'] * features.get('topic_consistency', 0):.4f} to AI score.")

        score += weights['contradictions'] * features.get('contradictions', 0)
        explanations.append(f"Contradictions contributes {weights['contradictions']} * {features.get('contradictions', 0):.4f} = {weights['contradictions'] * features.get('contradictions', 0):.4f} to AI score.")

        # Less personal pronouns may indicate AI
        pronoun_score = (1 - features.get('personal_pronoun_ratio', 0))  # Higher if fewer pronouns
        score += weights['personal_pronoun_ratio'] * pronoun_score
        explanations.append(f"Personal Pronoun Ratio contributes {weights['personal_pronoun_ratio']} * {pronoun_score:.4f} = {weights['personal_pronoun_ratio'] * pronoun_score:.4f} to AI score.")

        # Ensure the score is between 0 and 1
        score = np.clip(score, 0, 1)

        explanations.append(f"Heuristic AI Score (normalized to 0-1): {score:.4f}")

        return score, explanations

    # --- End of Enhanced AI Detection Methods ---

# Example usage
if __name__ == "__main__":
    # Use environment variable if no API key is provided explicitly
    detector = FakeNewsDetector()

    url = "https://theonion.com/dozens-of-pregnant-women-caught-in-hanging-snare-nets-above-texas-hospital-entrance/"
    url = "https://www.cnn.com/politics/live-news/trump-harris-election-09-15-24/index.html"
    url = "https://www.npr.org/2024/09/11/g-s1-22023/debate-harris-trump-takeaways"
    text = detector.download_text_from_url(url)

    if text:
        with suppress_stdout():
            result = detector.analyze_text(text)
        print(json.dumps(result, indent=4))
        print(result['deception_score'])
    else:
        print("Failed to download or process the text.")

