#!/usr/bin/env python3
#
# deception4.py

import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter, defaultdict
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import statistics
import math

# Download required NLTK data files
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

class DeceptionAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()

    def preprocess_text(self, text):
        # Remove noise and punctuation
        text = re.sub(r'\[[^]]*\]', '', text)  # Remove contents within brackets
        text = re.sub(r'\s+', ' ', text)       # Remove extra whitespace
        text = re.sub(r'[^\w\s]', '', text)    # Remove punctuation

        # Tokenize and lemmatize
        tokens = word_tokenize(text.lower())
        lemmas = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words]
        return ' '.join(lemmas)

    def get_ngrams(self, text, n):
        tokens = word_tokenize(text)
        ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        return Counter(ngrams)

    def get_part_of_speech(self, pos_tags):
        pos_count = {}
        for word, pos in pos_tags:
            if pos[0] not in pos_count:
                pos_count[pos[0]] = 0
            pos_count[pos[0]] += 1
        return pos_count

    def calculate_syntactic_complexity(self, sentences):
        complexity = 0
        for sentence in sentences:
            doc = nlp(sentence)
            complexity += sum(1 for token in doc if token.dep_ == 'ROOT')
        return complexity / len(sentences)

    def calculate_sentence_similarity(self, text):
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return 0.0  # Cannot compute similarity with one or zero sentences
        vectors = self.vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(vectors)
        sum_similarity = sum(sum(row) for row in similarity_matrix) - len(sentences)
        avg_similarity = sum_similarity / (len(sentences) * (len(sentences) - 1))
        return avg_similarity

    def analyze_sentiment(self, text):
        sentences = sent_tokenize(text)
        sentiment_scores = []
        emotional_intensity = []
        high_sentiment_sentences = []

        for sentence in sentences:
            sentiment = self.sia.polarity_scores(sentence)
            sentiment_scores.append(sentiment['compound'])
            emotional_intensity.append(abs(sentiment['compound']))
            if abs(sentiment['compound']) > 0.5:
                high_sentiment_sentences.append(sentence.strip())

        if not sentiment_scores:
            return 0.5, []  # Neutral sentiment score

        avg_sentiment = statistics.mean(sentiment_scores)
        avg_intensity = statistics.mean(emotional_intensity)

        # Count sentiment shifts between sentences
        sentiment_shifts = sum(
            1 for i in range(1, len(sentiment_scores))
            if (sentiment_scores[i] > 0 and sentiment_scores[i-1] < 0) or
               (sentiment_scores[i] < 0 and sentiment_scores[i-1] > 0)
        )

        # Normalize sentiment shift proportion
        sentiment_shift_proportion = sentiment_shifts / (len(sentiment_scores) - 1) if len(sentiment_scores) > 1 else 0

        # Compute sentiment-based deception score
        sentiment_score = 0.5 + (
            0.3 * (avg_sentiment / 2) +  # avg_sentiment ranges from -1 to 1
            0.2 * sentiment_shift_proportion +
            0.3 * avg_intensity +
            0.2 * (len(sentiment_scores) / len(text))
        )
        # Ensure the score is between 0 and 1
        sentiment_score = max(0.0, min(1.0, sentiment_score))
        return sentiment_score, high_sentiment_sentences

    def detect_contradictions(self, sentences):
        contradiction_count = 0
        total_pairs = 0
        contradictory_pairs = []

        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                total_pairs += 1
                sent_i = sentences[i].strip().lower()
                sent_j = sentences[j].strip().lower()
                # Simple heuristic for contradiction detection
                if sent_i == sent_j:
                    continue  # Skip identical sentences
                if ('not' in sent_i and 'not' not in sent_j) or \
                   ('not' in sent_j and 'not' not in sent_i):
                    contradiction_count += 1
                    contradictory_pairs.append((sentences[i].strip(), sentences[j].strip()))
        contradiction_proportion = contradiction_count / total_pairs if total_pairs > 0 else 0
        return contradiction_proportion, contradictory_pairs

    def analyze_entity_sentiment(self, text):
        doc = nlp(text)
        entity_sentiments = defaultdict(list)
        entity_sentiment_statements = defaultdict(list)

        for sentence in doc.sents:
            entities = [ent.text for ent in sentence.ents]
            sentiment = self.sia.polarity_scores(sentence.text)['compound']
            for entity in entities:
                entity_sentiments[entity].append(sentiment)
                entity_sentiment_statements[entity].append(sentence.text.strip())

        # Compute average sentiment per entity
        avg_entity_sentiments = {}
        for entity, sentiments in entity_sentiments.items():
            avg_entity_sentiments[entity] = statistics.mean(sentiments)

        # Calculate sentiment variance among entities
        if avg_entity_sentiments:
            sentiment_values = list(avg_entity_sentiments.values())
            sentiment_variance = statistics.variance(sentiment_values) if len(sentiment_values) > 1 else 0
        else:
            sentiment_variance = 0

        return sentiment_variance, entity_sentiment_statements

    def count_temporal_references(self, words, sentences):
        temporal_words = {'today', 'yesterday', 'tomorrow', 'now', 'then', 'soon', 'later', 'immediately', 'eventually'}
        temporal_count = sum(1 for word in words if word in temporal_words)
        temporal_sentences = [sentence.strip() for sentence in sentences if any(word in sentence.lower() for word in temporal_words)]
        temporal_proportion = temporal_count / len(words) if len(words) > 0 else 0
        return temporal_proportion, temporal_sentences

    def detect_passive_voice(self, sentences):
        passive_count = 0
        passive_sentences = []
        for sentence in sentences:
            doc = nlp(sentence)
            for token in doc:
                if token.dep_ == 'auxpass':
                    passive_count += 1
                    passive_sentences.append(sentence.strip())
                    break  # Count one per sentence
        passive_proportion = passive_count / len(sentences) if len(sentences) > 0 else 0
        return passive_proportion, passive_sentences

    def compute_deception_score(self, text):
        explanations = []
        contributing_statements = []

        prep_text = self.preprocess_text(text)
        words = word_tokenize(prep_text)
        total_words = len(words)
        sentences = sent_tokenize(text)
        total_sentences = len(sentences)

        if total_words == 0 or total_sentences == 0:
            return 0.0, ["The text is empty or has no valid content."], []

        pos_tags = nltk.pos_tag(words)
        pos_count = self.get_part_of_speech(pos_tags)

        unigrams = Counter(words)
        bigrams = self.get_ngrams(prep_text, 2)
        trigrams = self.get_ngrams(prep_text, 3)

        syntactic_complexity = self.calculate_syntactic_complexity(sentences)
        sentence_similarity = self.calculate_sentence_similarity(text)
        sentiment_score, high_sentiment_sentences = self.analyze_sentiment(text)
        avg_sentence_length = total_words / total_sentences

        # New features
        contradiction_proportion, contradictory_pairs = self.detect_contradictions(sentences)
        entity_sentiment_variance, entity_sentiment_statements = self.analyze_entity_sentiment(text)
        temporal_reference_proportion, temporal_sentences = self.count_temporal_references(words, sentences)
        passive_voice_proportion, passive_sentences = self.detect_passive_voice(sentences)

        # Word lists
        first_person_singular = {'i', 'me', 'my', 'mine', 'myself'}
        negative_emotions = {'hate', 'angry', 'sad', 'bad', 'terrible', 'awful', 'horrible', 'disgusting'}
        exclusive_words = {'but', 'except', 'without', 'excluding', 'however', 'nevertheless'}
        modal_verbs = {'could', 'would', 'should', 'might', 'may', 'maybe', 'possibly', 'likely'}
        motion_verbs = {'go', 'walk', 'run', 'move', 'travel', 'went', 'gone'}
        certainty_words = {'always', 'never', 'definitely', 'certainly', 'absolutely'}
        pronouns = {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'}
        fillers = {'um', 'uh', 'like', 'you know', 'so', 'actually'}
        uncertainty_words = {'possibly', 'maybe', 'it seems', 'i think', 'in my opinion'}

        # Count frequencies and track contributing words/sentences
        feature_contributions = defaultdict(list)

        fp_count = sum(1 for word in words if word in first_person_singular)
        if fp_count < (total_words * 0.05):  # Threshold for low first-person pronoun usage
            feature_contributions['fp_proportion'].append("Low use of first-person pronouns.")

        neg_emotion_words = [word for word in words if word in negative_emotions]
        neg_emotion_count = len(neg_emotion_words)
        if neg_emotion_words:
            feature_contributions['neg_emotion_proportion'].extend(neg_emotion_words)

        excl_words = [word for word in words if word in exclusive_words]
        excl_count = len(excl_words)
        if excl_count < (total_words * 0.01):  # Threshold for low exclusive words usage
            feature_contributions['excl_proportion'].append("Low use of exclusive words.")

        modal_words = [word for word in words if word in modal_verbs]
        modal_count = len(modal_words)
        if modal_words:
            feature_contributions['modal_proportion'].extend(modal_words)

        motion_words = [word for word in words if word in motion_verbs]
        motion_count = len(motion_words)
        if motion_count < (total_words * 0.01):  # Threshold for low motion verbs usage
            feature_contributions['motion_proportion'].append("Low use of motion verbs.")

        certainty_words_used = [word for word in words if word in certainty_words]
        certainty_count = len(certainty_words_used)
        if certainty_words_used:
            feature_contributions['certainty_proportion'].extend(certainty_words_used)

        pronoun_count = sum(1 for word in words if word in pronouns)
        if pronoun_count < (total_words * 0.1):  # Threshold for low pronoun usage
            feature_contributions['pronoun_proportion'].append("Low use of pronouns.")

        filler_words = [word for word in words if word in fillers]
        filler_count = len(filler_words)
        if filler_words:
            feature_contributions['filler_proportion'].extend(filler_words)

        uncertainty_words_used = [word for word in words if word in uncertainty_words]
        uncertainty_count = len(uncertainty_words_used)
        if uncertainty_words_used:
            feature_contributions['uncertainty_proportion'].extend(uncertainty_words_used)

        # POS counts
        verb_count = sum(1 for word, pos in pos_tags if pos.startswith('VB'))
        noun_count = sum(1 for word, pos in pos_tags if pos.startswith('NN'))
        adjective_count = sum(1 for word, pos in pos_tags if pos.startswith('JJ'))
        adverb_count = sum(1 for word, pos in pos_tags if pos.startswith('RB'))

        # Compute proportions
        fp_proportion = fp_count / total_words
        neg_emotion_proportion = neg_emotion_count / total_words
        excl_proportion = excl_count / total_words
        modal_proportion = modal_count / total_words
        motion_proportion = motion_count / total_words
        certainty_proportion = certainty_count / total_words
        pronoun_proportion = pronoun_count / total_words
        filler_proportion = filler_count / total_words
        uncertainty_proportion = uncertainty_count / total_words
        verb_proportion = verb_count / total_words
        noun_proportion = noun_count / total_words
        adjective_proportion = adjective_count / total_words
        adverb_proportion = adverb_count / total_words
        lexical_diversity = len(set(words)) / total_words

        # Bigram and trigram diversity
        total_bigrams = sum(bigrams.values())
        unique_bigrams = len(bigrams)
        bigram_diversity = unique_bigrams / total_bigrams if total_bigrams > 0 else 1

        total_trigrams = sum(trigrams.values())
        unique_trigrams = len(trigrams)
        trigram_diversity = unique_trigrams / total_trigrams if total_trigrams > 0 else 1

        # Features associated with deception
        features = {
            'fp_proportion': 1 - fp_proportion,                   # Lower is more deceptive
            'neg_emotion_proportion': neg_emotion_proportion,     # Higher is more deceptive
            'excl_proportion': 1 - excl_proportion,               # Lower is more deceptive
            'modal_proportion': modal_proportion,                 # Higher is more deceptive
            'motion_proportion': 1 - motion_proportion,           # Lower is more deceptive
            'certainty_proportion': certainty_proportion,         # Higher is more deceptive
            'pronoun_proportion': 1 - pronoun_proportion,         # Lower is more deceptive
            'filler_proportion': filler_proportion,               # Higher is more deceptive
            'uncertainty_proportion': uncertainty_proportion,     # Higher is more deceptive
            'lexical_diversity': 1 - lexical_diversity,           # Lower diversity may indicate deception
            'bigram_diversity': 1 - bigram_diversity,             # Lower diversity may indicate deception
            'trigram_diversity': 1 - trigram_diversity,           # Lower diversity may indicate deception
            'syntactic_complexity': 1 - (syntactic_complexity / (syntactic_complexity + 1)),  # Lower is more deceptive
            'sentence_similarity': sentence_similarity,           # Higher is more deceptive
            'sentiment_score': sentiment_score,                   # Higher is more deceptive
            'verb_proportion': verb_proportion,                   # Higher may be more deceptive
            'noun_proportion': 1 - noun_proportion,               # Lower may be more deceptive
            'adjective_proportion': adjective_proportion,         # Higher may be more deceptive
            'adverb_proportion': adverb_proportion,               # Higher may be more deceptive
            'avg_sentence_length': 1 - (avg_sentence_length / (avg_sentence_length + 1)),  # Shorter sentences are more deceptive
            'contradiction_proportion': contradiction_proportion, # Higher indicates more contradictions
            'entity_sentiment_variance': entity_sentiment_variance, # Higher variance may indicate bias
            'temporal_reference_proportion': 1 - temporal_reference_proportion, # Lower temporal references may be deceptive
            'passive_voice_proportion': passive_voice_proportion, # Higher passive voice may be deceptive
        }

        # Weights for each feature (adjustable)
        weights = {
            'fp_proportion': 2.0,
            'neg_emotion_proportion': 1.5,
            'excl_proportion': 2.0,
            'modal_proportion': 1.5,
            'motion_proportion': 1.5,
            'certainty_proportion': 2.0,
            'pronoun_proportion': 2.0,
            'filler_proportion': 2.0,
            'uncertainty_proportion': 1.5,
            'lexical_diversity': 1.0,
            'bigram_diversity': 1.0,
            'trigram_diversity': 1.0,
            'syntactic_complexity': 2.0,
            'sentence_similarity': 2.0,
            'sentiment_score': 2.0,
            'verb_proportion': 0.5,
            'noun_proportion': 0.5,
            'adjective_proportion': 0.5,
            'adverb_proportion': 0.5,
            'avg_sentence_length': 1.0,
            'contradiction_proportion': 2.0,
            'entity_sentiment_variance': 1.5,
            'temporal_reference_proportion': 1.5,
            'passive_voice_proportion': 1.5,
        }

        # Calculate weighted sum of feature scores
        weighted_scores = []
        for feature in features:
            weight = weights[feature]
            value = features[feature]
            weighted_value = weight * value
            weighted_scores.append(weighted_value)

            # Generate explanation
            if value > 0:
                explanation = f"Feature '{feature}' has a value of {value:.2f} with a weight of {weight}, contributing {weighted_value:.2f} to the score."
                explanations.append(explanation)

                # Include contributing statements or words if available
                if feature in feature_contributions and feature_contributions[feature]:
                    contributions = feature_contributions[feature]
                    explanations.append(f"Contributing elements for '{feature}': {', '.join(contributions)}")

                elif feature == 'contradiction_proportion' and contradictory_pairs:
                    explanations.append(f"Contradictory statements detected: {contradictory_pairs}")

                elif feature == 'passive_voice_proportion' and passive_sentences:
                    explanations.append(f"Sentences using passive voice: {passive_sentences}")
                    contributing_statements.extend(passive_sentences)

                elif feature == 'sentiment_score' and high_sentiment_sentences:
                    explanations.append(f"Sentences with high emotional intensity: {high_sentiment_sentences}")
                    contributing_statements.extend(high_sentiment_sentences)

                elif feature == 'temporal_reference_proportion' and not temporal_sentences:
                    explanations.append("Few or no temporal references found.")
                elif feature == 'entity_sentiment_variance' and entity_sentiment_statements:
                    for entity, statements in entity_sentiment_statements.items():
                        explanations.append(f"Statements about '{entity}': {statements}")
                        contributing_statements.extend(statements)

        total_weight = sum(weights.values())
        total_score = sum(weighted_scores) / total_weight

        # Ensure the score is between 0 and 1
        total_score = max(0.0, min(1.0, total_score))

        explanations.append(f"The final deception score is {total_score:.2f} on a scale from 0 (not deceptive) to 1 (highly deceptive).")

        # Remove duplicates from contributing statements
        contributing_statements = list(set(contributing_statements))

        return total_score, explanations, contributing_statements

    def score(self, text):
        """
        Calculates the deception score of the provided text, provides explanations, and lists contributing statements.

        Parameters:
        text (str): The text to analyze.

        Returns:
        tuple: (Deception score between 0 and 1, List of explanations, List of contributing statements)
        """
        return self.compute_deception_score(text)

# Instantiate the analyzer
analyzer = DeceptionAnalyzer()

# Sample text to analyze
text = """

A judge on Thursday threw out three charges in the sweeping Georgia election subversion case, including two charges that former President Donald Trump faces.

The decision hasn’t yet been formally applied to Trump because his case has been paused pending appeals.

In a separate ruling, Fulton County Judge Scott McAfee also upheld the marquee racketeering charge in the case, which Trump is also facing.

Trump lawyer Steve Sadow hailed the rulings as a victory.

“President Trump and his legal team in Georgia have prevailed once again,” Sadow said in a statement. “The trial court has decided that counts 15 and 27 in the indictment must be quashed/dismissed.”

McAfee threw out one count of filing false documents and one count of conspiring to file false documents, both stemming from the Trump campaign’s efforts to put forward a slate of fake GOP electors in Georgia. Trump was only named in the conspiracy count.

In the ruling, McAfee also threw out a separate count of filing false documents, which Trump was charged with. That count relates to untrue statements about supposed voter fraud that were included in one of Trump’s lawsuits in December 2020 that attempted to negate the election results.

These rulings only narrowly took effect for former Trump lawyer John Eastman and Georgia state Sen. Shawn Still, who were involved in the 2020 fake electors plot. Their cases are not currently paused. Trump was only named in two of the three charges that McAfee threw out Thursday.

The awkward ruling from McAfee comes as he has only partial jurisdiction over the 2020 election meddling case. Trump and most of his remaining co-defendants are seeking to have an appeals court disqualify Fulton County District Attorney Fani Willis from overseeing the case. So, their cases are paused with McAfee, who presides in a lower court. But two defendants, Eastman and Still, opted to move ahead with their cases in the lower court rather than join the appeal over whether Willis should be prosecuting the case.

Willis, a Democrat, originally secured a 13-count indictment against Trump last summer, related to his multi-pronged attempts to overturn his 2020 defeat in the Peach State. McAfee already threw out three of Trump’s charges in March.

Willis’ office declined to comment. McAfee’s chambers did not immediately return a request for additional clarification.
"""

# Calculate the deception score and get explanations
score, explanations, contributing_statements = analyzer.score(text)

print(f"Deception Score: {score:.2f}\n")
print("Explanations:")
for explanation in explanations:
    print("- " + explanation)

print("\nStatements that caused the deception score to increase:")
for statement in contributing_statements:
    print("- " + statement)

