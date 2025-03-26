#!/usr/bin/env python3
#
# sentence_checker.py

import re
from nltk import pos_tag, word_tokenize
from nltk.tokenize import sent_tokenize
import nltk

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# List of predefined patterns
patterns = [
    # Simple Sentence (S + V)
    (r'^(NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ)$', "Simple Sentence (S + V)"),

    # Simple Sentence with Object (S + V + O)
    (r'^(NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ) (NN|NNS|NNP|NNPS|PRP|JJ|DT|CD)$', "Simple Sentence with Object (S + V + O)"),

    # Simple Sentence with Complement (S + V + C)
    (r'^(NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ) (NN|NNS|NNP|NNPS|JJ|DT|CD)$', "Simple Sentence with Complement (S + V + C)"),

    # Double Object Sentence (S + V + IO + DO)
    (r'^(NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ) (NN|NNS|NNP|NNPS|PRP) (NN|NNS|NNP|NNPS|PRP|JJ|DT|CD)$', "Double Object Sentence (S + V + IO + DO)"),

    # Simple Sentence with Adverb (S + V + Adv)
    (r'^(NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ) (RB|RBR|RBS)$', "Simple Sentence with Adverb (S + V + Adv)"),

    # Simple Sentence with Adjective (S + V + Adj)
    (r'^(NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ) (JJ|JJR|JJS)$', "Simple Sentence with Adjective (S + V + Adj)"),

    # Compound Sentence (S + V, FANBOYS S + V)
    (r'^(NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ) , (CC) (NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ)$', "Compound Sentence (S + V, FANBOYS S + V)"),

    # Complex Sentence with Subordinate Clause (S + V + Subordinate Clause)
    (r'^(NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ) (IN|WDT|WRB|WP|WP$) (NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ)$', "Complex Sentence with Subordinate Clause (S + V + Subordinate Clause)"),

    # Complex Sentence with Relative Clause (S + Relative Pronoun + V + O)
    (r'^(NN|NNS|NNP|NNPS|PRP) (WDT|WP|WP$|WRB) (VB|VBD|VBG|VBN|VBP|VBZ) (NN|NNS|NNP|NNPS|PRP)$', "Complex Sentence with Relative Clause (S + Relative Pronoun + V + O)"),

    # Compound-Complex Sentence (S + V + O, Subordinate Clause, S + V + O)
    (r'^(NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ) (NN|NNS|NNP|NNPS|PRP) , (IN|WDT|WRB|WP|WP$) (NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ) , (NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ) (NN|NNS|NNP|NNPS|PRP)$', "Compound-Complex Sentence (S + V + O, Subordinate Clause, S + V + O)"),

    # Yes/No Question (Auxiliary Verb + S + V)
    (r'^(VB|VBD|VBG|VBN|VBP|VBZ) (NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ)$', "Yes/No Question (Auxiliary Verb + S + V)"),

    # Wh- Question (Wh-word + Auxiliary Verb + S + V)
    (r'^(WDT|WP|WP$|WRB) (VB|VBD|VBG|VBN|VBP|VBZ) (NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ)$', "Wh- Question (Wh-word + Auxiliary Verb + S + V)"),

    # Imperative Sentence (You) + V + O
    (r'^(VB|VBP) (NN|NNS|NNP|NNPS|PRP)$', "Imperative Sentence (You) + V + O"),

    # Negative Imperative Sentence (Don't + V + O)
    (r'^Don\'t (VB|VBP) (NN|NNS|NNP|NNPS|PRP)$', "Negative Imperative Sentence (Don't + V + O)"),

    # Prepositional Phrase (S + V + PP)
    (r'^(DT|JJ|NN|NNS|NNP|NNPS|PRP)+ (VB|VBD|VBG|VBN|VBP|VBZ) (IN) (DT|JJ|NN|NNS|NNP|NNPS|PRP)+$', "Prepositional Phrase (S + V + PP)"),

    # Adjective Phrase (S + AdjP + V + O)
    (r'^(NN|NNS|NNP|NNPS|PRP) (JJ|JJR|JJS) (VB|VBD|VBG|VBN|VBP|VBZ) (NN|NNS|NNP|NNPS|PRP)$', "Adjective Phrase (S + AdjP + V + O)"),

    # Adverbial Phrase (S + V + AdvP)
    (r'^(NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ) (RB|RBR|RBS)$', "Adverbial Phrase (S + V + AdvP)"),

    # Noun Phrase (NP + V + O)
    (r'^(DT|JJ|JJR|JJS|PRP$|RB|RBR|RBS|NN|NNS|NNP|NNPS) (VB|VBD|VBG|VBN|VBP|VBZ) (NN|NNS|NNP|NNPS|PRP)$', "Noun Phrase (NP + V + O)"),

    # Participle Phrase (S + V + Participle Phrase)
    (r'^(NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ) (VBG|VBN)$', "Participle Phrase (S + V + Participle Phrase)"),

    # First Conditional (If + S + V, S + V)
    (r'^If (NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ), (NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ)$', "First Conditional (If + S + V, S + V)"),

    # Second Conditional (If + S + V (past), S + V (would))
    (r'^If (NN|NNS|NNP|NNPS|PRP) (VBD|VBN|VBP|VBZ), (NN|NNS|NNP|NNPS|PRP) (MD) (VB)$', "Second Conditional (If + S + V (past), S + V (would))"),

    # Third Conditional (If + S + V (past perfect), S + V (would have))
    (r'^If (NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ) (VBN|VBD), (NN|NNS|NNP|NNPS|PRP) (MD) (VB) (VBN)$', "Third Conditional (If + S + V (past perfect), S + V (would have))"),

    # Mixed Conditional (If + S + V (past perfect), S + V (would))
    (r'^If (NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ) (VBN|VBD), (NN|NNS|NNP|NNPS|PRP) (MD) (VB)$', "Mixed Conditional (If + S + V (past perfect), S + V (would))"),

    # Simple Passive (S + be + V-ed)
    (r'^(NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ) (VBN)$', "Simple Passive (S + be + V-ed)"),

    # Passive with Modal (S + modal + be + V-ed)
    (r'^(NN|NNS|NNP|NNPS|PRP) (MD) (VB) (VBN)$', "Passive with Modal (S + modal + be + V-ed)"),

    # Passive with Indirect Object (S + be + V-ed + IO + by + O)
    (r'^(NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ) (VBN) (NN|NNS|NNP|NNPS|PRP) (IN) (NN|NNS|NNP|NNPS|PRP)$', "Passive with Indirect Object (S + be + V-ed + IO + by + O)"),

    # Gerund as Subject (Gerund + V)
    (r'^(VBG) (VB|VBD|VBG|VBN|VBP|VBZ)$', "Gerund as Subject (Gerund + V)"),

    # Infinitive as Subject (Infinitive + V)
    (r'^(TO) (VB|VBP) (VB|VBD|VBG|VBN|VBP|VBZ)$', "Infinitive as Subject (Infinitive + V)"),

    # Gerund as Object (S + V + Gerund)
    (r'^(NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ) (VBG)$', "Gerund as Object (S + V + Gerund)"),

    # Infinitive as Object (S + V + Infinitive)
    (r'^(NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ) (TO) (VB|VBP)$', "Infinitive as Object (S + V + Infinitive)"),

    # Prepositional Phrase (S + V + PP)
    (r'^(DT|JJ|NN|NNS|NNP|NNPS|PRP)+ (VB|VBD|VBG|VBN|VBP|VBZ) (IN) (DT|JJ|NN|NNS|NNP|NNPS|PRP)+$', "Prepositional Phrase (S + V + PP)"),
    (r'^(DT) (JJ) (NN) (NN) (VBZ) (IN) (DT) (JJ) (NN) (.)$', 'OTHER'),
]

def generate_regex_for_sentence(sentence):
    # Tokenize and tag the parts of speech
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)

    # Convert POS tags to a simplified string
    tag_string = " ".join([pos for word, pos in tagged])

    # Check against predefined patterns
    for pattern, description in patterns:
        if re.match(pattern, tag_string):
            print("Matched:")
            return pattern, description

    # If no predefined pattern matches, return the custom pattern
    custom_pattern = r'^' + " ".join(f'({pos})' for pos in tag_string.split()) + r'$'
    return custom_pattern, "OTHER"

def split_text_into_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

# Example usage
sentence = "The quick brown fox jumps over the lazy dog."
text = """Hello! How are you today? I hope you're doing well. This is a test sentence. Let's see how it works."""
sentences = split_text_into_sentences(text)
for sentence in sentences:
    pattern, description = generate_regex_for_sentence(sentence)
    print(f"(r'{pattern}', '{description}')")

