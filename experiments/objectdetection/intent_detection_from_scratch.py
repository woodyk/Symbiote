#!/usr/bin/env python3
#
# intent_detection_from_scratch.py

import re
import json
import requests
import numpy as np
from bs4 import BeautifulSoup
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textstat import textstat
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore
from transformers import pipeline

nltk.download('punkt')
nltk.download('stopwords')

class DeceptionDetector:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.emotion_analyzer = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

    def analyze_text(self, text):
        sentences = self.split_text_into_sentences(text)
        words = self.split_text_into_words(text)

        readability_score = self.calculate_readability(text)
        lexical_diversity = self.calculate_lexical_diversity(words)
        sentiment_chain = self.analyze_sentiment_chain(sentences)
        tone_variability = self.analyze_tone_variability(sentiment_chain)
        sentiment_shift_score = self.detect_sentiment_shifts(sentiment_chain)
        linguistic_features = self.analyze_linguistic_features(sentences)
        syntactic_complexity = self.analyze_syntactic_complexity(sentences)
        behavioral_markers = self.analyze_behavioral_markers(sentences)
        emotional_consistency_score, emotional_sentence_scores = self.analyze_emotional_patterns(sentences)

        sentence_scores = self.calculate_sentence_scores(
            sentences, emotional_sentence_scores, lexical_diversity, linguistic_features, behavioral_markers
        )

        outlier_scores = self.detect_anomalies(sentences)

        deception_score = self.calculate_deception_score(
            readability_score, lexical_diversity, tone_variability, linguistic_features, syntactic_complexity,
            behavioral_markers, sentiment_shift_score, outlier_scores, emotional_consistency_score
        )

        explanation = self.generate_explanation(
            readability_score, lexical_diversity, tone_variability, linguistic_features, syntactic_complexity,
            behavioral_markers, sentiment_shift_score, outlier_scores, emotional_consistency_score
        )

        top_deceptive_sentences = self.extract_top_deceptive_sentences(sentence_scores)

        return {
            "deception_score": deception_score,
            "readability_score": readability_score,
            "lexical_diversity": lexical_diversity,
            "tone_variability": tone_variability,
            "linguistic_features": linguistic_features,
            "syntactic_complexity": syntactic_complexity,
            "behavioral_markers": behavioral_markers,
            "sentiment_shift_score": sentiment_shift_score,
            "outlier_scores": outlier_scores,
            "emotional_consistency_score": emotional_consistency_score,
            "explanation": explanation,
            "top_deceptive_sentences": top_deceptive_sentences
        }

    def split_text_into_sentences(self, text):
        sentences = sent_tokenize(text)
        return sentences

    def split_text_into_words(self, text):
        words = [word for word in word_tokenize(text) if word.isalpha() and word not in self.stop_words]
        return words

    def calculate_readability(self, text):
        fk_grade = textstat.flesch_kincaid_grade(text)
        normalized_readability = np.clip(fk_grade / 12, 0, 1)
        return normalized_readability

    def calculate_lexical_diversity(self, words):
        lexical_diversity = len(set(words)) / len(words) if len(words) > 0 else 0
        return lexical_diversity

    def analyze_sentiment_chain(self, sentences):
        sentiment_chain = []
        for sentence in sentences:
            result = self.sentiment_analyzer(sentence)[0]
            score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
            sentiment_chain.append(score)
        return sentiment_chain

    def analyze_tone_variability(self, sentiment_chain):
        if len(sentiment_chain) > 1:
            tone_variability = np.std(sentiment_chain)
            normalized_tone_variability = np.clip(tone_variability, 0, 1)
        else:
            normalized_tone_variability = 0
        return normalized_tone_variability

    def detect_sentiment_shifts(self, sentiment_chain):
        shifts = 0
        threshold = 0.5
        for i in range(1, len(sentiment_chain)):
            if abs(sentiment_chain[i] - sentiment_chain[i - 1]) > threshold:
                shifts += 1
        sentiment_shift_score = np.clip(shifts / len(sentiment_chain), 0, 1)
        return sentiment_shift_score

    def analyze_linguistic_features(self, sentences):
        passive_voice_count = 0
        modal_verb_count = 0
        negation_count = 0

        for sentence in sentences:
            words = word_tokenize(sentence)
            tagged_words = nltk.pos_tag(words)

            passive_voice_count += sum(1 for word, tag in tagged_words if tag == 'VBN')
            modal_verb_count += sum(1 for word, tag in tagged_words if tag in ['MD'])
            negation_count += sum(1 for word in words if word.lower() in ['not', 'no', 'never', 'n’t'])

        sentence_count = len(sentences)
        normalized_passive_voice = np.clip(passive_voice_count / sentence_count, 0, 1)
        normalized_modal_verbs = np.clip(modal_verb_count / sentence_count, 0, 1)
        normalized_negations = np.clip(negation_count / sentence_count, 0, 1)

        linguistic_features = {
            "passive_voice_count": normalized_passive_voice,
            "modal_verb_count": normalized_modal_verbs,
            "negation_count": normalized_negations,
            "sentence_count": sentence_count
        }
        return linguistic_features

    def analyze_syntactic_complexity(self, sentences):
        complex_sentence_count = 0
        for sentence in sentences:
            words = word_tokenize(sentence)
            tagged_words = nltk.pos_tag(words)
            complex_sentence_count += sum(1 for word, tag in tagged_words if tag in ['VBN', 'VBG', 'IN'])

        syntactic_complexity = complex_sentence_count / len(sentences)
        normalized_syntactic_complexity = np.clip(syntactic_complexity, 0, 1)
        return normalized_syntactic_complexity

    def analyze_behavioral_markers(self, sentences):
        first_person_pronouns = [
            'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 'myself', 'ourselves'
        ]

        qualifiers = [
            'very', 'really', 'extremely', 'absolutely', 'definitely', 'certainly', 'truly',
            'surely', 'completely', 'utterly', 'highly', 'perfectly', 'deeply', 'incredibly',
            'totally', 'significantly', 'greatly', 'quite', 'rather', 'fairly', 'somewhat',
            'slightly', 'pretty', 'kind of', 'sort of', 'basically'
        ]

        hedging_words = [
            'maybe', 'possibly', 'perhaps', 'could', 'might', 'can', 'may', 'seem', 'appear',
            'likely', 'suggest', 'assume', 'arguably', 'probably', 'apparently', 'presumably',
            'somewhat', 'kind of', 'sort of', 'supposedly', 'I think', 'I believe', 'I guess',
            'I suppose', 'I feel', 'I assume', 'I imagine', 'I suspect', 'it seems', 'it appears',
            'it’s possible', 'as far as I know', 'as far as I can tell'
        ]

        first_person_count = 0
        qualifier_count = 0
        hedging_count = 0

        for sentence in sentences:
            words = word_tokenize(sentence)

            first_person_count += sum(1 for word in words if word.lower() in first_person_pronouns)
            qualifier_count += sum(1 for word in words if word.lower() in qualifiers)
            hedging_count += sum(1 for word in words if word.lower() in hedging_words)

        sentence_count = len(sentences)
        normalized_first_person = np.clip(first_person_count / sentence_count, 0, 1)
        normalized_qualifiers = np.clip(qualifier_count / sentence_count, 0, 1)
        normalized_hedging = np.clip(hedging_count / sentence_count, 0, 1)

        behavioral_markers = {
            "first_person_count": normalized_first_person,
            "qualifier_count": normalized_qualifiers,
            "hedging_count": normalized_hedging
        }
        return behavioral_markers

    def analyze_emotional_patterns(self, sentences):
        emotional_sentence_scores = []
        for sentence in sentences:
            model_outputs = self.emotion_analyzer(sentence)
            if isinstance(model_outputs, list) and len(model_outputs) > 0 and isinstance(model_outputs[0], list):
                model_outputs = model_outputs[0]
            dominant_emotion = max(model_outputs, key=lambda x: x['score'])
            emotional_sentence_scores.append(dominant_emotion['score'])

        emotional_consistency_score = self.detect_emotional_anomalies(emotional_sentence_scores)
        return emotional_consistency_score, emotional_sentence_scores

    def detect_emotional_anomalies(self, emotional_scores):
        emotional_scores = np.array(emotional_scores).reshape(-1, 1)
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        outlier_labels = lof.fit_predict(emotional_scores)
        anomaly_score = np.mean(outlier_labels == -1)
        clipped_anomaly_score = np.clip(anomaly_score, 0, 1)
        return clipped_anomaly_score

    def detect_anomalies(self, sentences):
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        sentence_lengths = np.array(sentence_lengths).reshape(-1, 1)

        scaler = StandardScaler()
        scaled_lengths = scaler.fit_transform(sentence_lengths)

        isolation_forest = IsolationForest(contamination=0.1)
        outlier_labels = isolation_forest.fit_predict(scaled_lengths)

        z_scores = zscore(scaled_lengths)

        isolation_forest_outliers = int(np.sum(outlier_labels == -1)) / len(sentences)
        z_score_outliers = int(np.sum(np.abs(z_scores) > 2)) / len(sentences)

        clipped_isolation_forest_outliers = np.clip(isolation_forest_outliers, 0, 1)
        clipped_z_score_outliers = np.clip(z_score_outliers, 0, 1)

        outlier_scores = {
            "isolation_forest_outliers": clipped_isolation_forest_outliers,
            "z_score_outliers": clipped_z_score_outliers
        }
        return outlier_scores

    def calculate_sentence_scores(self, sentences, emotional_sentence_scores, lexical_diversity, linguistic_features, behavioral_markers):
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            sentence_emotion_score = emotional_sentence_scores[i]
            sentence_linguistic_score = self.calculate_individual_sentence_linguistic_score(linguistic_features, i)
            sentence_behavioral_score = self.calculate_individual_sentence_behavioral_score(behavioral_markers, i)
            sentence_score = 0.5 * sentence_emotion_score + 0.25 * sentence_linguistic_score + 0.25 * sentence_behavioral_score
            sentence_scores.append((sentence, sentence_score))
        return sentence_scores

    def calculate_individual_sentence_linguistic_score(self, linguistic_features, sentence_index):
        linguistic_score = (linguistic_features['passive_voice_count'] +
                            linguistic_features['modal_verb_count'] +
                            linguistic_features['negation_count']) / 3
        return linguistic_score

    def calculate_individual_sentence_behavioral_score(self, behavioral_markers, sentence_index):
        behavioral_score = (behavioral_markers['first_person_count'] +
                            behavioral_markers['qualifier_count'] +
                            behavioral_markers['hedging_count']) / 3
        return behavioral_score

    def extract_top_deceptive_sentences(self, sentence_scores, top_n=3):
        sorted_scores = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
        top_deceptive_sentences = sorted_scores[:top_n]
        return top_deceptive_sentences

    def calculate_deception_score(self, readability, lexical_diversity, tone_variability, linguistic_features, syntactic_complexity, behavioral_markers, sentiment_shift_score, outlier_scores, emotional_consistency_score):
        complexity_score = readability

        outlier_score = (outlier_scores['isolation_forest_outliers'] + outlier_scores['z_score_outliers']) / 2

        linguistic_score = (linguistic_features['passive_voice_count'] +
                            linguistic_features['modal_verb_count'] +
                            linguistic_features['negation_count']) / 3

        behavioral_score = (behavioral_markers['first_person_count'] +
                            behavioral_markers['qualifier_count'] +
                            behavioral_markers['hedging_count']) / 3

        deception_score = (0.15 * complexity_score +
                           0.1 * tone_variability +
                           0.1 * (1 - lexical_diversity) +
                           0.1 * outlier_score +
                           0.1 * linguistic_score +
                           0.1 * syntactic_complexity +
                           0.1 * behavioral_score +
                           0.15 * sentiment_shift_score +
                           0.1 * emotional_consistency_score)

        rounded_deception_score = round(deception_score, 4)
        return rounded_deception_score

    def generate_explanation(self, readability, lexical_diversity, tone_variability, linguistic_features, syntactic_complexity, behavioral_markers, sentiment_shift_score, outlier_scores, emotional_consistency_score):
        explanation = {
            "Readability Impact": (
                f"Text readability score normalized to {readability:.4f}, "
                f"{'increasing' if readability > 0.67 else 'moderately impacting' if readability > 0.33 else 'lowering'} the deception score. "
                f"High readability scores can increase the perceived complexity, which may be a sign of deception."
            ),
            "Lexical Diversity Impact": (
                f"Lexical diversity is {lexical_diversity:.4f}, "
                f"{'decreasing' if lexical_diversity > 0.5 else 'increasing'} the deception score. "
                f"Lower diversity can indicate repetitive language, which may suggest a higher likelihood of deception."
            ),
            "Tone Variability Impact": (
                f"Tone variability is {tone_variability:.4f}, "
                f"{'lowering' if tone_variability < 0.5 else 'increasing'} the deception score. "
                f"Consistent tone may lower the likelihood of deception, while shifts in tone can increase it."
            ),
            "Linguistic Features Impact": (
                f"Detected {linguistic_features['passive_voice_count']:.4f} normalized instances of passive voice, "
                f"{linguistic_features['modal_verb_count']:.4f} modal verbs, and {linguistic_features['negation_count']:.4f} negations per sentence. "
                f"These linguistic features {'may contribute to a higher deception score' if linguistic_features['passive_voice_count'] > 0.33 else 'have a minimal effect on the deception score'}."
            ),
            "Syntactic Complexity Impact": (
                f"Syntactic complexity score is {syntactic_complexity:.4f}, "
                f"{'increasing' if syntactic_complexity > 0.5 else 'lowering'} the deception score. "
                f"Highly complex sentence structures can be a sign of intentional obfuscation."
            ),
            "Behavioral Markers Impact": (
                f"Detected {behavioral_markers['first_person_count']:.4f} instances of first-person pronouns, "
                f"{behavioral_markers['qualifier_count']:.4f} qualifiers, and {behavioral_markers['hedging_count']:.4f} hedging language per sentence. "
                f"These markers {'contribute to a higher deception score' if behavioral_markers['hedging_count'] > 0.33 else 'have a minimal effect on the deception score'}."
            ),
            "Sentiment Shift Impact": (
                f"Detected {sentiment_shift_score:.4f} significant sentiment shifts throughout the text. "
                f"These shifts can indicate inconsistency in the narrative, contributing to a higher deception score."
            ),
            "Anomaly Detection Impact": (
                f"{outlier_scores['isolation_forest_outliers']:.4f} proportion of sentences flagged as outliers by Isolation Forest and "
                f"{outlier_scores['z_score_outliers']:.4f} by Z-Score, "
                f"{'increasing' if outlier_scores['isolation_forest_outliers'] > 0.1 else 'slightly impacting'} the deception score. "
                f"Anomalies in sentence structure can suggest potential deception."
            ),
            "Emotional Consistency Impact": (
                f"Emotional consistency score is {emotional_consistency_score:.4f}. "
                f"{'High emotional variability or anomalies' if emotional_consistency_score > 0.5 else 'Stable emotional patterns'} "
                f"affect the deception score accordingly."
            )
        }
        return explanation

    def download_text_from_url(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            extracted_text = "\n".join([para.get_text() for para in paragraphs])
            return extracted_text
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while trying to download the text: {e}")
            return ""

    def analyze_url(self, url):
        text = self.download_text_from_url(url)
        if text:
            return self.analyze_text(text)
        else:
            return {"error": "Failed to download or process the text from the URL."}


# Example Usage
if __name__ == "__main__":
    detector = DeceptionDetector()

    # Analyze a given text directly
    # text = "Your example text goes here. Analyze this text to detect any potential deception using non-AI-based methods."
    # result = detector.analyze_text(text)
    # print("Deception Score:", result['deception_score'])
    # print("Explanation:", result['explanation'])
    # print("Top Deceptive Sentences:", result['top_deceptive_sentences'])

    # Analyze text from a URL
    url = "https://www.msnbc.com/opinion/msnbc-opinion/trump-trading-cards-nft-america-first-rcna168999"
    url_result = detector.analyze_url(url)
    if "error" not in url_result:
        print(json.dumps(url_result, indent=4))
        print("Deception Score from URL:", url_result['deception_score'])
        print("Top Deceptive Sentences from URL:", url_result['top_deceptive_sentences'])
    else:
        print(url_result['error'])

