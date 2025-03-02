#!/usr/bin/env python3
#
# DeceptionDetection.py
"""
Project Overview:
The `DeceptionDetection.py` script is a sophisticated tool designed to analyze 
text or URLs for potential indicators of deception. This solution leverages 
Natural Language Processing (NLP), statistical analysis, and machine learning 
techniques to identify linguistic, emotional, and syntactic patterns commonly 
associated with deceptive behavior. It offers a detailed and holistic analysis 
of input text or web-based content.

Purpose and Goals:
- To detect and quantify deception in written communication using evidence-based 
  linguistic markers and statistical features.
- To provide interpretability by generating a detailed explanation of the 
  deception score based on multiple analysis dimensions.
- To facilitate extensibility for future research or real-world applications 
  in domains like security, journalism, or fraud detection.

Key Features:
1. **Deception Pattern Recognition**:
   - A comprehensive set of regex-based deception patterns including:
     - Hedging statements, verbal fillers, and certainty words.
     - Non-contracted denials and overly formal language.
     - Chronological storytelling and excessive detail.
     - Linguistic minimization, repetition, and negations.
   - These patterns are aggregated into scores reflecting their frequency and 
     impact on the overall deception score.

2. **Sentiment and Emotional Analysis**:
   - Leverages Hugging Face Transformers for sentiment analysis and emotion 
     classification.
   - Identifies sentiment shifts and emotional inconsistencies in the text to 
     assess narrative reliability.

3. **Linguistic and Syntactic Features**:
   - Evaluates lexical diversity, syntactic complexity, passive voice usage, 
     and modal verb frequency.
   - Combines these linguistic markers to quantify how text complexity relates 
     to potential deception.

4. **Anomaly Detection**:
   - Uses statistical techniques like Isolation Forest and Z-Score to identify 
     anomalies in sentence structure and length.
   - Detects outlier sentences that may signal fabricated or deceptive content.

5. **Readability Scoring**:
   - Employs Flesch-Kincaid grade level to measure text readability and normalize 
     its contribution to the deception score.

6. **Integration with Web Content**:
   - Processes both plain text and web pages by extracting visible content using 
     BeautifulSoup.
   - Can analyze URLs directly, making it versatile for news, social media, 
     and other web-based text sources.

7. **Comprehensive Scoring and Explanation**:
   - Generates a deception score by combining multiple dimensions:
     - Readability, tone variability, linguistic complexity, emotional patterns, 
       and deception markers.
   - Provides an interpretive explanation for each contributing factor, offering 
     transparency and insight into the analysis.

Methodologies:
- **NLP Pipelines**: Utilizes Hugging Face Transformers for sentiment and emotion 
  analysis, and NLTK for tokenization and linguistic feature extraction.
- **Regex-Based Detection**: Patterns are implemented to capture behavioral and 
  linguistic markers of deception.
- **Statistical Modeling**: Applies machine learning techniques (e.g., Isolation 
  Forest) and statistical scores (e.g., Z-Score) for anomaly detection.
- **Contextual Scoring**: Combines individual feature scores with weighted 
  contributions to calculate the overall deception score.

Extensibility:
- Adding New Deception Patterns:
  - Define new regex patterns and integrate them into the `self.deception_patterns` 
    dictionary.
- Supporting Additional Models:
  - Replace or extend Hugging Face pipelines with domain-specific models for 
    improved sentiment or emotion analysis.
- Scaling for Larger Texts or Datasets:
  - Adapt machine learning algorithms and pre-processing techniques for 
    batch processing or streaming analysis.

Best Practices:
- Ensure input text is preprocessed to remove noise (e.g., HTML tags) for 
  improved accuracy.
- Use the `analyze_text` method as a high-level entry point for analysis to 
  ensure all components are properly utilized.
- For web-based analysis, validate URL accessibility and expected content structure 
  before running the tool.

Reusable Prompt for Extending:
- "Develop a deception detection model focusing on [specific linguistic or emotional 
  patterns]. Use multi-dimensional scoring and statistical methods to enhance 
  reliability and interpretability."

Personal Style Alignment:
- The project reflects a modular and extensible design, prioritizing clarity and 
  reusability in its structure.
- Heavy use of helper methods ensures functionality is encapsulated, facilitating 
  updates and debugging.
- The explanation system adds an interpretability layer, aligning the tool with 
  user-centric applications like fraud prevention, journalism integrity, or legal 
  investigations.

Usage Example:
```
detector = DeceptionDetector()
text_or_url = "Honestly, I didn't take the money. To the best of my knowledge, the document was misplaced."
result = detector.analyze_text(text_or_url)
print(json.dumps(result, indent=4))
```
"""

import re
import json
import requests
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textstat import textstat
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore
from transformers import pipeline
import os
import warnings

# Suppress NLTK output and Hugging Face FutureWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=UserWarning)

# Set the environment variable to suppress the Hugging Face tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DeceptionDetector:
    def __init__(self):
        """Initialize the deception detector with necessary components and configurations."""
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.emotion_analyzer = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
        self.deception_patterns = self._initialize_deception_patterns()

    def _initialize_deception_patterns(self):
        """Internal method to initialize deception patterns as regular expressions."""
        patterns = {
            "overemphasizing_truthfulness": re.compile(r"\b(honestly|to be honest|believe me|i swear|let me be clear|trust me)\b", re.IGNORECASE),
            "non_contracted_denials": re.compile(r"\b(i did not|he does not|she did not|they did not|it is not)\b", re.IGNORECASE),
            "hedging_statements": re.compile(r"\b(as far as i know|to the best of my knowledge|i believe|maybe|possibly|likely|probably)\b", re.IGNORECASE),
            "avoidance_of_pronouns": re.compile(r"\b(the document was|the item was|the task was)\b", re.IGNORECASE),
            "excessive_detail": re.compile(r"\b(first|then|after that|next)\b", re.IGNORECASE),
            "euphemisms": re.compile(r"\b(take|borrow|misplace|involved|accidentally)\b", re.IGNORECASE),
            "repeated_question": re.compile(r"\b(did i|do you mean)\b", re.IGNORECASE),
            "defensive_responses": re.compile(r"\b(why would you|what do you mean|how could you)\b", re.IGNORECASE),
            "verbal_fillers": re.compile(r"\b(um|uh|you know|like)\b", re.IGNORECASE),
            "certainty_words": re.compile(r"\b(always|never|absolutely|definitely|certainly)\b", re.IGNORECASE),
            "lack_of_specificity": re.compile(r"\b(something|stuff|things|someone|somebody|somewhere)\b", re.IGNORECASE),
            "chronological_storytelling": re.compile(r"\b(first|second|third|after that|then)\b", re.IGNORECASE),
            "negation": re.compile(r"\b(did not|didn't)\b", re.IGNORECASE),
            "minimization": re.compile(r"\b(just a|only a|small)\b", re.IGNORECASE),
            "repetition": re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE),
            "unexpected_details": re.compile(r"\b(unnecessary detail|irrelevant detail|unrelated)\b", re.IGNORECASE),
            "overly_formal": re.compile(r"\b(hereby|therefore|henceforth)\b", re.IGNORECASE),
            "first_person_pronouns": re.compile(r"\b(i|me|my|mine|we|us|our|ours|myself|ourselves)\b", re.IGNORECASE),
            "qualifiers": re.compile(r"\b(very|really|extremely|absolutely|definitely|certainly|truly|surely|completely|utterly|highly|perfectly|deeply|incredibly|totally|significantly|greatly|quite|rather|fairly|somewhat|slightly|pretty|kind of|sort of|basically)\b", re.IGNORECASE)
        }
        return patterns

    def analyze_text(self, input_text_or_url):
        """
        Analyze input text or URL to detect deception.
        Automatically determines if the input is a URL or text.
        """
        text = self._get_text_from_input(input_text_or_url)
        if not text:
            return {"error": "Failed to process the input."}

        sentences, words = self._tokenize_text(text)

        readability_score = self._calculate_readability(text)
        lexical_diversity = self._calculate_lexical_diversity(words)
        sentiment_chain = self._analyze_sentiment_chain(sentences)
        tone_variability = self._calculate_variability(sentiment_chain)
        sentiment_shift_score = self._detect_sentiment_shifts(sentiment_chain)
        linguistic_features = self._analyze_linguistic_features(sentences, words)
        syntactic_complexity = self._analyze_syntactic_complexity(sentences, words)
        deception_marker_scores, deception_pattern_aggregate = self._analyze_deception_markers(sentences)
        emotional_consistency_score, emotional_sentence_scores = self._analyze_emotional_patterns(sentences)

        sentence_scores = self._calculate_sentence_scores(
            sentences, lexical_diversity, linguistic_features, deception_marker_scores
        )

        outlier_scores = self._detect_anomalies(sentences)

        deception_score = self._calculate_deception_score(
            readability_score, lexical_diversity, tone_variability, linguistic_features, syntactic_complexity,
            sentiment_shift_score, outlier_scores, deception_marker_scores, emotional_consistency_score
        )

        explanation = self._generate_explanation(
            readability_score, lexical_diversity, tone_variability, linguistic_features, syntactic_complexity,
            sentiment_shift_score, outlier_scores, deception_marker_scores, emotional_consistency_score
        )

        top_deceptive_sentences = self._extract_top_deceptive_sentences(sentence_scores)

        result = {
            "deception_score": deception_score,
            "readability_score": float(readability_score),
            "lexical_diversity": float(lexical_diversity),
            "tone_variability": float(tone_variability),
            "linguistic_features": {k: float(v) for k, v in linguistic_features.items()},
            "syntactic_complexity": float(syntactic_complexity),
            "sentiment_shift_score": float(sentiment_shift_score),
            "emotional_consistency_score": float(emotional_consistency_score),
            "outlier_scores": {k: float(v) for k, v in outlier_scores.items()},
            "deception_pattern_aggregate": deception_pattern_aggregate,
            "explanation": explanation,
            "top_deceptive_sentences": [(sentence, float(score)) for sentence, score in top_deceptive_sentences]
        }

        return result

    def _get_text_from_input(self, input_text_or_url):
        """Determine if the input is a URL or plain text, and extract the text content accordingly."""
        if re.match(r'^https?://', input_text_or_url):
            text = self._download_text_from_url(input_text_or_url)
        else:
            text = input_text_or_url
        return text

    def _download_text_from_url(self, url):
        """Download and extract text content from a given URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            extracted_text = "\n".join([para.get_text() for para in paragraphs])
        except requests.exceptions.RequestException as e:
            log(f"An error occurred while trying to download the text: {e}")
            extracted_text = ""
        return extracted_text

    def _tokenize_text(self, text):
        """Tokenize the text into sentences and words, filtering out stop words."""
        sentences = sent_tokenize(text)
        words = [word for word in word_tokenize(text) if word.isalpha() and word not in self.stop_words]
        return sentences, words

    def _calculate_readability(self, text):
        """Calculate the readability score of the text using the Flesch-Kincaid grade level."""
        fk_grade = textstat.flesch_kincaid_grade(text)
        normalized_readability = np.clip(fk_grade / 12, 0, 1)
        return normalized_readability

    def _calculate_lexical_diversity(self, words):
        """Calculate the lexical diversity of the text."""
        lexical_diversity = len(set(words)) / len(words) if words else 0
        return lexical_diversity

    def _analyze_sentiment_chain(self, sentences):
        """Analyze the sentiment of each sentence and return a list of sentiment scores."""
        sentiment_chain = [
            self.sentiment_analyzer(sentence)[0]['score'] if self.sentiment_analyzer(sentence)[0]['label'] == 'POSITIVE' else -self.sentiment_analyzer(sentence)[0]['score']
            for sentence in sentences
        ]
        return sentiment_chain

    def _calculate_variability(self, values):
        """Calculate the standard deviation of a list of values, used for tone and sentiment variability."""
        if len(values) > 1:
            variability = np.std(values)
        else:
            variability = 0
        normalized_variability = np.clip(variability, 0, 1)
        return normalized_variability

    def _detect_sentiment_shifts(self, sentiment_chain):
        """Detect significant shifts in sentiment throughout the text."""
        shifts = sum(1 for i in range(1, len(sentiment_chain)) if abs(sentiment_chain[i] - sentiment_chain[i - 1]) > 0.5)
        sentiment_shift_score = np.clip(shifts / len(sentiment_chain), 0, 1)
        return sentiment_shift_score

    def _analyze_linguistic_features(self, sentences, words):
        """Analyze linguistic features such as passive voice, modal verbs, and negations."""
        tagged_words = nltk.pos_tag(words)
        passive_voice_count = sum(1 for _, tag in tagged_words if tag == 'VBN')
        modal_verb_count = sum(1 for _, tag in tagged_words if tag in ['MD'])
        negation_count = sum(1 for word in words if word.lower() in ['not', 'no', 'never', 'n’t'])

        linguistic_features = self._normalize_counts(passive_voice_count, modal_verb_count, negation_count, total=len(sentences))
        return linguistic_features

    def _analyze_syntactic_complexity(self, sentences, words):
        """Analyze the syntactic complexity of the text based on the use of complex structures."""
        tagged_words = nltk.pos_tag(words)
        complex_sentence_count = sum(1 for _, tag in tagged_words if tag in ['VBN', 'VBG', 'IN'])
        syntactic_complexity = np.clip(complex_sentence_count / len(sentences), 0, 1)
        return syntactic_complexity

    def _analyze_deception_markers(self, sentences):
        """Analyze text for combined behavioral markers and deception patterns."""
        deception_scores = []
        deception_pattern_aggregate = {pattern: 0 for pattern in self.deception_patterns.keys()}

        for sentence in sentences:
            score = 0
            for pattern_name, pattern in self.deception_patterns.items():
                if pattern.search(sentence):
                    score += 1
                    deception_pattern_aggregate[pattern_name] += 1
            deception_scores.append(score)

        normalized_scores = np.clip(np.array(deception_scores) / len(self.deception_patterns), 0, 1)
        return normalized_scores, deception_pattern_aggregate

    def _analyze_emotional_patterns(self, sentences):
        """Analyze the emotional patterns in the text using a pre-trained emotion classification model."""
        emotional_sentence_scores = []
        for sentence in sentences:
            model_outputs = self.emotion_analyzer(sentence)
            if isinstance(model_outputs, list) and len(model_outputs) > 0 and isinstance(model_outputs[0], list):
                model_outputs = model_outputs[0]
            dominant_emotion = max(model_outputs, key=lambda x: x['score'])
            emotional_sentence_scores.append(dominant_emotion['score'])

        emotional_consistency_score = self._detect_emotional_anomalies(emotional_sentence_scores)
        return emotional_consistency_score, emotional_sentence_scores

    def _detect_emotional_anomalies(self, emotional_scores):
        """Detect emotional anomalies within the emotional pattern of the text."""
        emotional_scores = np.array(emotional_scores).reshape(-1, 1)
        n_samples = len(emotional_scores)
        
        # If there's only one or too few sentences, return 0 as anomaly score
        if n_samples < 2:
            return 0  # or some other default value indicating no anomalies detected
        
        n_neighbors = min(20, n_samples - 1)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
        outlier_labels = lof.fit_predict(emotional_scores)
        anomaly_score = np.mean(outlier_labels == -1)
        clipped_anomaly_score = np.clip(anomaly_score, 0, 1)
        return clipped_anomaly_score


    def _normalize_counts(self, *counts, total):
        """Normalize counts by the total number of sentences."""
        normalized_counts = {f"feature_{i+1}": np.clip(count / total, 0, 1) for i, count in enumerate(counts)}
        return normalized_counts

    def _detect_anomalies(self, sentences):
        """Detect anomalies in sentence lengths using Isolation Forest and Z-Score."""
        sentence_lengths = np.array([len(sentence.split()) for sentence in sentences]).reshape(-1, 1)
        scaler = StandardScaler()
        scaled_lengths = scaler.fit_transform(sentence_lengths)

        isolation_forest = IsolationForest(contamination=0.1)
        outlier_labels = isolation_forest.fit_predict(scaled_lengths)
        z_scores = zscore(scaled_lengths)

        outlier_scores = {
            "isolation_forest_outliers": np.clip(np.mean(outlier_labels == -1), 0, 1),
            "z_score_outliers": np.clip(np.mean(np.abs(z_scores) > 2), 0, 1)
        }
        return outlier_scores

    def _calculate_sentence_scores(self, sentences, lexical_diversity, linguistic_features, deception_marker_scores):
        """Calculate overall deception-related scores for each sentence."""
        sentence_scores = [
            (
                sentence,
                0.4 * self._average_features(linguistic_features) +
                0.2 * lexical_diversity +
                0.2 * deception_marker_scores[i] +
                0.2 * deception_marker_scores[i]
            )
            for i, sentence in enumerate(sentences)
        ]
        return sentence_scores

    def _average_features(self, features):
        """Average multiple feature scores."""
        if isinstance(features, dict):
            average_score = sum(features.values()) / len(features)
        elif isinstance(features, np.ndarray):
            average_score = np.mean(features)
        else:
            raise TypeError("Unsupported feature type for averaging.")
        return average_score

    def _extract_top_deceptive_sentences(self, sentence_scores, top_n=10):
        """Extract the top N sentences most indicative of deception."""
        top_deceptive_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:top_n]
        return top_deceptive_sentences

    def _calculate_deception_score(self, readability, lexical_diversity, tone_variability, linguistic_features, syntactic_complexity, sentiment_shift_score, outlier_scores, deception_marker_scores, emotional_consistency_score):
        """Calculate the overall deception score for the text."""
        deception_score = (
            0.1 * readability +
            0.1 * tone_variability +
            0.1 * (1 - lexical_diversity) +
            0.1 * (outlier_scores['isolation_forest_outliers'] + outlier_scores['z_score_outliers']) / 2 +
            0.1 * self._average_features(linguistic_features) +
            0.1 * syntactic_complexity +
            0.1 * sentiment_shift_score +
            0.1 * emotional_consistency_score +
            0.15 * np.mean(deception_marker_scores)
        )

        rounded_deception_score = round(deception_score, 4)
        return rounded_deception_score

    def _generate_explanation(self, readability, lexical_diversity, tone_variability, linguistic_features, syntactic_complexity, sentiment_shift_score, outlier_scores, deception_marker_scores, emotional_consistency_score):
        """Generate a detailed explanation for the calculated deception score."""
        explanation = {
            "Readability Impact": f"Text readability score normalized to {readability:.4f}, {'increasing' if readability > 0.67 else 'moderately impacting' if readability > 0.33 else 'lowering'} the deception score. High readability scores can increase the perceived complexity, which may be a sign of deception.",
            "Lexical Diversity Impact": f"Lexical diversity is {lexical_diversity:.4f}, {'decreasing' if lexical_diversity > 0.5 else 'increasing'} the deception score. Lower diversity can indicate repetitive language, which may suggest a higher likelihood of deception.",
            "Tone Variability Impact": f"Tone variability is {tone_variability:.4f}, {'lowering' if tone_variability < 0.5 else 'increasing'} the deception score. Consistent tone may lower the likelihood of deception, while shifts in tone can increase it.",
            "Linguistic Features Impact": f"Detected {linguistic_features['feature_1']:.4f} normalized instances of passive voice, {linguistic_features['feature_2']:.4f} modal verbs, and {linguistic_features['feature_3']:.4f} negations per sentence. These linguistic features {'may contribute to a higher deception score' if linguistic_features['feature_1'] > 0.33 else 'have a minimal effect on the deception score'}.",
            "Syntactic Complexity Impact": f"Syntactic complexity score is {syntactic_complexity:.4f}, {'increasing' if syntactic_complexity > 0.5 else 'lowering'} the deception score. Highly complex sentence structures can be a sign of intentional obfuscation.",
            "Sentiment Shift Impact": f"Detected {sentiment_shift_score:.4f} significant sentiment shifts throughout the text. These shifts can indicate inconsistency in the narrative, contributing to a higher deception score.",
            "Anomaly Detection Impact": f"{outlier_scores['isolation_forest_outliers']:.4f} proportion of sentences flagged as outliers by Isolation Forest and {outlier_scores['z_score_outliers']:.4f} by Z-Score, {'increasing' if outlier_scores['isolation_forest_outliers'] > 0.1 else 'slightly impacting'} the deception score. Anomalies in sentence structure can suggest potential deception.",
            "Emotional Consistency Impact": f"Emotional consistency score is {emotional_consistency_score:.4f}. {'High emotional variability or anomalies' if emotional_consistency_score > 0.5 else 'Stable emotional patterns'} affect the deception score accordingly.",
            "Deception Pattern Impact": f"Deception pattern score is {np.mean(deception_marker_scores):.4f}. Frequent use of deceptive language patterns significantly increases the deception score."
        }
        return explanation

# Example Usage
if __name__ == "__main__":
    detector = DeceptionDetector()

    # Analyze text or URL
    #input_text_or_url = "Honestly, I didn't take the money. To the best of my knowledge, the document was misplaced."
    #result = detector.analyze_text(input_text_or_url)
    #print(json.dumps(result, indent=4))

    # Example with a URL
    input_text_or_url = "https://www.msnbc.com/opinion/msnbc-opinion/trump-trading-cards-nft-america-first-rcna168999"
    input_text_or_url = "https://theonion.com/dozens-of-pregnant-women-caught-in-hanging-snare-nets-above-texas-hospital-entrance/"
    input_text_or_url = "https://www.foxnews.com/transcript/fox-news-sunday-july-21-2024"
    result = detector.analyze_text(input_text_or_url)
    print(json.dumps(result, indent=4))

