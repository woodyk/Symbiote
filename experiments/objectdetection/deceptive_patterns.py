#!/usr/bin/env python3
#
# deceptive_patterns.py

import re
from collections import Counter

# Updated and expanded lists of patterns associated with deception

NEGATIVE_WORDS = [
    "hate", "fear", "angry", "dislike", "upset", "regret", 
    "scared", "terrified", "nervous", "afraid", "anxious", 
    "worried", "furious", "irritated", "annoyed", "rage", 
    "resentful", "depressed", "miserable", "guilty", "ashamed", 
    "heartbroken", "repulsed", "nauseated", "horrified", "disgusted"
]

PASSIVE_WORDS = [
    "was", "were", "is being", "are being", "been", "be", 
    "is done", "was made", "has been", "was handled", "got", 
    "was taken", "was told", "it appears", "it seems", "it was said", 
    "it was reported"
]

VAGUE_WORDS = [
    "maybe", "perhaps", "kind of", "sort of", "around", 
    "possibly", "probably", "unsure", "likely", "could be", 
    "somewhat", "roughly", "approximately", "sometime", "eventually", 
    "later on", "soon", "a bit", "a few", "several", "a couple", 
    "some"
]

ABSOLUTE_WORDS = [
    "always", "never", "everyone", "nobody", "all", "none", 
    "forever", "constantly", "every time", "without fail", 
    "no one", "no way", "everybody", "everything", "entirely", 
    "completely", "definitely", "undoubtedly", "certainly", 
    "without question"
]

DISTANCING_WORDS = [
    "that guy", "that woman", "someone", "they", "them", 
    "those people", "a person", "he", "she", "that man", 
    "individual", "the other one", "the guy", "the girl", 
    "the person", "somebody", "one of them", "a friend", 
    "an acquaintance", "the person I know"
]

JUSTIFICATIONS = [
    "because", "since", "due to", "as a result", "so that", 
    "therefore", "hence", "thus", "for this reason", "accordingly", 
    "consequently", "to make sure", "in order to", "the reason is", 
    "so I could", "so I wouldn’t", "I had to", "I couldn’t", 
    "I did it because", "I wanted to make sure"
]

QUALIFIERS = [
    "possibly", "maybe", "could", "might", "potentially", 
    "seems like", "it appears", "it could be", "more or less", 
    "kind of", "sort of", "to some extent", "I guess", 
    "I'm not sure", "if", "perhaps", "probably", "somehow", 
    "I think", "I believe"
]

MINIMIZERS = [
    "just", "only", "merely", "hardly", "barely", 
    "no big deal", "nothing to worry about"
]

EVASIVE_PHRASES = [
    "you know", "like I said", "well", "honestly", 
    "to tell the truth", "if I’m being honest", 
    "let me be clear", "frankly"
]

POLITENESS = [
    "sir", "ma'am", "with all due respect", "thank you for your time", 
    "I apologize", "please", "kindly"
]

INTENSIFIERS = [
    "literally", "honestly", "absolutely", "truthfully", 
    "really", "definitely", "I swear", "trust me"
]

THIRD_PERSON_PRONOUNS = [
    "he", "she", "they", "someone", "that person", 
    "him", "her"
]


def detect_negative_language(text):
    words = text.lower().split()
    neg_count = sum(1 for word in words if word in NEGATIVE_WORDS)
    return neg_count

def detect_passive_voice(text):
    passive_count = len(re.findall(r'\b(?:' + '|'.join(PASSIVE_WORDS) + r')\b', text.lower()))
    return passive_count

def detect_vagueness(text):
    vague_count = sum(1 for word in text.split() if word.lower() in VAGUE_WORDS)
    return vague_count

def detect_absolutes(text):
    absolute_count = sum(1 for word in text.split() if word.lower() in ABSOLUTE_WORDS)
    return absolute_count

def detect_distancing_language(text):
    distancing_count = sum(1 for phrase in DISTANCING_WORDS if phrase.lower() in text.lower())
    return distancing_count

def detect_over_justification(text):
    justification_count = sum(1 for word in text.split() if word.lower() in JUSTIFICATIONS)
    return justification_count

def detect_qualifiers(text):
    qualifier_count = sum(1 for word in text.split() if word.lower() in QUALIFIERS)
    return qualifier_count

def detect_minimizers(text):
    minimizer_count = sum(1 for word in text.split() if word.lower() in MINIMIZERS)
    return minimizer_count

def detect_evasive_language(text):
    evasive_count = sum(1 for phrase in EVASIVE_PHRASES if phrase.lower() in text.lower())
    return evasive_count

def detect_politeness(text):
    politeness_count = sum(1 for word in text.split() if word.lower() in POLITENESS)
    return politeness_count

def detect_intensifiers(text):
    intensifier_count = sum(1 for word in text.split() if word.lower() in INTENSIFIERS)
    return intensifier_count

def detect_third_person_pronouns(text):
    third_person_count = sum(1 for word in text.split() if word.lower() in THIRD_PERSON_PRONOUNS)
    return third_person_count


def analyze_text_for_deception(text):
    # Analyze each pattern
    negative_score = detect_negative_language(text)
    passive_voice_score = detect_passive_voice(text)
    vagueness_score = detect_vagueness(text)
    absolute_score = detect_absolutes(text)
    distancing_score = detect_distancing_language(text)
    justification_score = detect_over_justification(text)
    qualifier_score = detect_qualifiers(text)
    minimizer_score = detect_minimizers(text)
    evasive_score = detect_evasive_language(text)
    politeness_score = detect_politeness(text)
    intensifier_score = detect_intensifiers(text)
    third_person_score = detect_third_person_pronouns(text)

    # Print the results
    analysis = {
        "Negative Language": negative_score,
        "Passive Voice": passive_voice_score,
        "Vagueness": vagueness_score,
        "Absolutes": absolute_score,
        "Distancing Language": distancing_score,
        "Over-Justification": justification_score,
        "Qualifiers": qualifier_score,
        "Minimizers": minimizer_score,
        "Evasive Language": evasive_score,
        "Politeness": politeness_score,
        "Intensifiers": intensifier_score,
        "Third-Person Pronouns": third_person_score
    }

    print("\nDeception Analysis Results:")
    for pattern, score in analysis.items():
        print(f"{pattern}: {score}")

    # Generate a summary
    total_score = sum(analysis.values())
    if total_score >= 7:
        print("\nThe text shows several patterns often associated with deception.")
    else:
        print("\nThe text does not show strong indicators of deception based on the analyzed patterns.")


if __name__ == "__main__":
    # Example usage
    user_text = """
    Former Vice President Dick Cheney said Friday that he will vote for Democrat Kamala Harris over fellow Republican Donald Trump in the November election, warning that the former president “can never be trusted with power again.”

“In our nation’s 248-year history, there has never been an individual who is a greater threat to our republic than Donald Trump,” Cheney said in a statement. “He tried to steal the last election using lies and violence to keep himself in power after the voters had rejected him. He can never be trusted with power again.”

“As citizens, we each have a duty to put country above partisanship to defend our Constitution. That is why I will be casting my vote for Vice President Kamala Harris,” he concluded.

Cheney’s daughter, former Wyoming Rep. Liz Cheney, first revealed earlier in the day that her father would be voting for the Democratic ticket during remarks at the Texas Tribune Festival in Austin, Texas. The former Republican congresswoman previously announced that she would be voting for Harris, citing “the danger that Donald Trump poses.”

Committee Vice Chair Rep. Liz Cheney (R-WY) attends the final meeting of the U.S. House Select Committee investigating the January 6 Attack on the U.S. Capitol, on Capitol Hill in Washington, U.S., December 19, 2022. REUTERS/Jonathan Ernst
Related article
Liz Cheney says she is voting for Harris for president
In her remarks Friday, Cheney unleashed a torrent of criticism against the Republican presidential ticket, calling Trump “a depraved human being” and labeling him and his running mate, Ohio Sen. JD Vance, “misogynistic pigs.”

“Every Republican, anybody who’s contemplating casting a vote for that ticket, you know, really needs to think about what they are enabling, what they’re embracing and the danger of electing people who will only honor election results if they agree with the outcome, and who are willing to set aside the Constitution,” she said in Austin. “And you know in the case of Donald Trump, promote, provoke, exploit violence in order to seize power.”

Trump, in turn, wrote on his Trump Social platform that “Dick Cheney is an irrelevant RINO, along with his daughter,” referring to the Cheneys as “Republican in name only.”

The former congresswoman, who previously told CNN she was committed to doing what was necessary to stop Trump from returning to the White House, said Friday that she expected to campaign against the former president in battleground states this fall. And she suggested that her father shared her views about Trump.

“If you think about the moment that we’re in and you think about how serious this moment is, you know, my dad believes – and he said publicly – that there’s never been an individual in our country who is as grave a threat to our democracy as Donald Trump is,” she said.

Dick Cheney’s support for Harris represents a stunning move for the staunch conservative who was vice president to George W. Bush and a longtime congressman from Wyoming who held several leadership roles in the House Republican Caucus.

The former vice president was critical of his party and Trump in the wake of the January 6 attack. In campaign ads for his daughter’s 2022 reelection effort, he called Trump a “threat to our republic” and a “coward.”

“He tried to steal the last election using lies and violence to keep himself in power after the voters had rejected him,” Dick Cheney said of the former president in the ad.

His daughter’s outspokenness against Trump and the former president’s efforts to overturn the 2020 election – including her vote to impeach him led to her losing her leadership position as the No. 3 House Republican. The GOP caucus ousted her as conference chair in May 2021 and replaced her with New York’s Elise Stefanik, a top Trump defender, several months after the January 6 attack.

Liz Cheney went on to serve as vice chair of the House select committee that investigated the Capitol riot and ultimately lost her seat in Congress to a Trump-backed Republican primary challenger.

She argued Friday that Trump and Vance were “doing everything they can to drive Reagan Republicans away” and that the duo “certainly do not reflect the importance of that Reagan philosophy of peace through strength and a strong national defense.”

Cheney said she watched parts of the Republican National Convention in July and took away that “it isn’t just that the party is trying to whitewash what Donald Trump did … the party is embracing it.”

Speaking about her decision to support Harris, Cheney said while she has “serious policy disagreements on a whole range of issues,” she feels that “those of us who believe in the defense of our democracy, in the defense of our Constitution, and the survival of our republic have a duty in this election cycle to come together to put those things above politics.”

In addition to backing Harris, Cheney also endorsed Democratic Rep. Colin Allred’s effort to unseat Republican Sen. Ted Cruz of Texas.

This headline and story have been updated with additional information."""
    analyze_text_for_deception(user_text)

