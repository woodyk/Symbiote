# Project: Symbiote



## Vision
Symbiote - an innovative personal data integration assistant that not only logs your life's milestones but also helps analyze your health, finances, personal behavior, and so much more. Symbiote creates a multi-generational legacy, allowing future generations to connect with their family's past like never before.

Imagine a world where you live beyond the constraints of time and space, where your experiences, knowledge, and wisdom are not lost but perpetuated and made accessible for generations. A world where your legacy is not just a collection of tangible assets, but an evolving digital tapestry of your life, enriched with your insights and personal growth.

In this world, your life's journey becomes an open book, not just for your descendants, but also for yourself. The lessons you've learned, the milestones you've achieved, and the adventures you've embarked on are at your fingertips, ready to be revisited or shared at any moment.

This is the world that Symbiote seeks to create. A world where each individual's life story becomes a part of an enduring digital heritage, shaping the wisdom of generations to come. Through Symbiote, we are not just remembering and preserving our past, but we are also forging a future that values the richness of human experience, the strength of shared knowledge, and the power of connected memories.

Welcome to the era where you are not just a fleeting moment in time, but a continuous, evolving narrative shaping the world one life story at a time. Welcome to Symbiote!

##### Author: Wadih Khairallah

## Donate:
BitCoin: 36Mg8UnAhVU5ZqSikQmbVXGG3rpYZRGNjC

## Running symbiote

# MacOs
```
brew install pipx protobuf libmagic tesseract
pipx install . --python $(which python3.11) 
symbiote -i
symbiote
```

# Ubuntu
```
apt-get install pipx portaudio19-dev libmagic1 tesseract-ocr
pipx install . --python $(which python3.11)
symbiote -i
symbiote
```

Add the following with your API keys to your .bashrc
```
export GOOGLE_API_KEY="your_google_api_key_here"
export GOOGLE_CX="your_google_search_code_here"
export OPENAI_API_KEY="your_openai_api_key_here"
```
For linux support on ubuntu be sure to add your user to the 'input' group.
