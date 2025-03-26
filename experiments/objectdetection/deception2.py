#!/usr/bin/env python3
#
# deception2.py

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from collections import Counter
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def deception_score(text):
    """
    Analyzes the input text for features associated with deceptive language
    and returns a deception score between 0 and 1.

    Parameters:
    text (str): The text to be analyzed.

    Returns:
    float: A deception score between 0 and 1.
    """
    # Preprocessing
    text = text.lower()
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    total_words = len(words)
    if total_words == 0:
        return 0.0  # Return 0 for empty text

    stop_words = set(stopwords.words('english'))
    words_no_stop = [word for word in words if word.isalpha() and word not in stop_words]
    total_words_no_stop = len(words_no_stop)

    # POS Tagging
    pos_tags = pos_tag(words_no_stop)
    pos_counts = Counter(tag for word, tag in pos_tags)

    # Define word lists (expanded)
    first_person_singular = {'i', 'me', 'my', 'mine', 'myself'}
    first_person_plural = {'we', 'us', 'our', 'ours', 'ourselves'}
    second_person = {'you', 'your', 'yours', 'yourself', 'yourselves'}
    third_person = {'he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'theirs'}
    exclusive_words = {'but', 'except', 'without', 'excluding', 'however', 'nevertheless', 'yet', 'although'}
    negative_emotions = {'hate', 'angry', 'sad', 'bad', 'terrible', 'awful', 'regret', 'sorry', 'disappoint'}
    causation_words = {'because', 'since', 'therefore', 'so', 'hence', 'thus'}
    certainty_words = {'always', 'never', 'definitely', 'certainly', 'undoubtedly'}
    tentative_words = {'maybe', 'perhaps', 'possibly', 'might', 'could', 'seems'}
    sensory_perception = {'see', 'hear', 'feel', 'touch', 'smell', 'taste'}
    fillers = {'um', 'uh', 'like', 'you know', 'actually'}
    modal_verbs = {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would'}

    # Count frequencies
    fp_singular_count = sum(1 for word in words if word in first_person_singular)
    fp_plural_count = sum(1 for word in words if word in first_person_plural)
    second_person_count = sum(1 for word in words if word in second_person)
    third_person_count = sum(1 for word in words if word in third_person)
    excl_count = sum(1 for word in words if word in exclusive_words)
    neg_emotion_count = sum(1 for word in words if word in negative_emotions)
    causation_count = sum(1 for word in words if word in causation_words)
    certainty_count = sum(1 for word in words if word in certainty_words)
    tentative_count = sum(1 for word in words if word in tentative_words)
    sensory_count = sum(1 for word in words if word in sensory_perception)
    filler_count = sum(1 for word in words if word in fillers)
    modal_count = sum(1 for word in words if word in modal_verbs)

    # Syntactic features
    avg_sentence_length = total_words / len(sentences)
    complex_word_count = sum(1 for word in words_no_stop if len(word) > 6)
    avg_word_length = sum(len(word) for word in words_no_stop) / total_words_no_stop
    noun_count = pos_counts.get('NN', 0) + pos_counts.get('NNS', 0)
    verb_count = pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) + pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)
    adjective_count = pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + pos_counts.get('JJS', 0)
    adverb_count = pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + pos_counts.get('RBS', 0)
    pronoun_count = pos_counts.get('PRP', 0) + pos_counts.get('PRP$', 0)

    # Compute proportions
    fp_singular_prop = fp_singular_count / total_words
    fp_plural_prop = fp_plural_count / total_words
    second_person_prop = second_person_count / total_words
    third_person_prop = third_person_count / total_words
    excl_prop = excl_count / total_words
    neg_emotion_prop = neg_emotion_count / total_words
    causation_prop = causation_count / total_words
    certainty_prop = certainty_count / total_words
    tentative_prop = tentative_count / total_words
    sensory_prop = sensory_count / total_words
    filler_prop = filler_count / total_words
    modal_prop = modal_count / total_words
    complex_word_prop = complex_word_count / total_words_no_stop
    noun_prop = noun_count / total_words_no_stop
    verb_prop = verb_count / total_words_no_stop
    adjective_prop = adjective_count / total_words_no_stop
    adverb_prop = adverb_count / total_words_no_stop
    pronoun_prop = pronoun_count / total_words_no_stop

    # Features associated with deception:
    # - Fewer first-person singular pronouns
    # - More negative emotion words
    # - Fewer exclusive words
    # - Fewer sensory perception words
    # - More tentative words
    # - More fillers
    # - Simpler sentence structure (shorter sentences, fewer complex words)
    # - Different POS usage (e.g., fewer adjectives)

    # Weights for each feature (adjusted for importance)
    weights = {
        'fp_singular': 1.5,
        'neg_emotion': 1.5,
        'excl': 1.0,
        'sensory': 1.0,
        'tentative': 1.5,
        'filler': 1.0,
        'sentence_length': 1.0,
        'complex_word': 1.0,
        'adjective': 1.0,
        'adverb': 1.0,
        'pronoun': 1.0,
        'modal': 1.0,
    }

    # Individual feature scores (scaled between 0 and 1)
    fp_singular_score = 1 - fp_singular_prop  # Lower is more deceptive
    neg_emotion_score = neg_emotion_prop      # Higher is more deceptive
    excl_score = 1 - excl_prop                # Lower is more deceptive
    sensory_score = 1 - sensory_prop          # Lower is more deceptive
    tentative_score = tentative_prop          # Higher is more deceptive
    filler_score = filler_prop                # Higher is more deceptive
    sentence_length_score = 1 - min(avg_sentence_length / 30, 1)  # Shorter sentences are more deceptive
    complex_word_score = 1 - complex_word_prop  # Fewer complex words are more deceptive
    adjective_score = 1 - adjective_prop      # Fewer adjectives are more deceptive
    adverb_score = adverb_prop                # More adverbs are more deceptive
    pronoun_score = pronoun_prop              # More pronouns are more deceptive
    modal_score = modal_prop                  # More modal verbs are more deceptive

    # Calculate weighted average
    total_weight = sum(weights.values())
    total_score = (
        weights['fp_singular'] * fp_singular_score +
        weights['neg_emotion'] * neg_emotion_score +
        weights['excl'] * excl_score +
        weights['sensory'] * sensory_score +
        weights['tentative'] * tentative_score +
        weights['filler'] * filler_score +
        weights['sentence_length'] * sentence_length_score +
        weights['complex_word'] * complex_word_score +
        weights['adjective'] * adjective_score +
        weights['adverb'] * adverb_score +
        weights['pronoun'] * pronoun_score +
        weights['modal'] * modal_score
    ) / total_weight

    # Ensure the score is between 0 and 1
    total_score = max(0.0, min(1.0, total_score))

    return total_score

text = "I didn't see what happened, but maybe someone else did. You know, things like this are quite unexpected."
score = deception_score(text)
print(f"Deception Score: {score:.2f}")

