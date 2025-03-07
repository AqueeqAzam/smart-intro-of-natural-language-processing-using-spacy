import spacy

# 📌 1. Installing and Loading spaCy Model
# Definition: spaCy models provide pre-trained NLP capabilities.
# Usage: Used for text processing, AI chatbots, search engines.

# Install spaCy model (Run this in terminal if not installed)
# !python -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")  # Load small English model
print("✅ spaCy Model Loaded")


# 📌 2. Tokenization
# Definition: Splitting text into words, punctuation, or sentences.
# Usage: Used in text preprocessing, chatbots, text analysis.

text = "Hello, how are you doing today?"
doc = nlp(text)

print("\n📝 Tokenized Words:")
for token in doc:
    print(token.text)


# 📌 3. Part-of-Speech (POS) Tagging
# Definition: Identifies grammatical categories of words.
# Usage: Used in grammar checking, AI writing assistants.

print("\n🔠 POS Tagging:")
for token in doc:
    print(f"{token.text} → {token.pos_}")


# 📌 4. Named Entity Recognition (NER)
# Definition: Identifies names of people, organizations, dates, and places.
# Usage: Used in chatbots, news extraction, and finance.

text = "Apple was founded by Steve Jobs in California in 1976."
doc = nlp(text)

print("\n🏷️ Named Entities:")
for ent in doc.ents:
    print(f"{ent.text} → {ent.label_}")


# 📌 5. Dependency Parsing
# Definition: Analyzes the grammatical structure of a sentence.
# Usage: Used in AI voice assistants, language translation.

print("\n🔗 Dependency Parsing:")
for token in doc:
    print(f"{token.text} → Head: {token.head.text}, Dependency: {token.dep_}")


# 📌 6. Lemmatization
# Definition: Converts words to their base form.
# Usage: Used in search engines, AI writing tools.

print("\n📚 Lemmatization:")
for token in doc:
    print(f"{token.text} → {token.lemma_}")


# 📌 7. Stop Words Removal
# Definition: Removes common words that don’t add meaning.
# Usage: Used in search engines, text summarization.

filtered_tokens = [token.text for token in doc if not token.is_stop]
print("\n🚫 Tokens without Stop Words:", filtered_tokens)


# 📌 8. Text Similarity
# Definition: Compares similarity between texts.
# Usage: Used in recommendation systems, duplicate content detection.

text1 = nlp("I love programming.")
text2 = nlp("Coding is my passion.")

similarity = text1.similarity(text2)
print("\n🔍 Text Similarity Score:", similarity)


# 📌 9. Rule-Based Matching
# Definition: Finds patterns in text using rules.
# Usage: Used in chatbots, information extraction.

from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)
pattern = [{"LOWER": "apple"}]  # Match the word 'apple' (case insensitive)
matcher.add("APPLE_PATTERN", [pattern])

doc = nlp("I love Apple products.")
matches = matcher(doc)

print("\n🛠️ Rule-Based Matching Results:")
for match_id, start, end in matches:
    print(f"Matched: {doc[start:end].text}")


# 📌 10. Custom Named Entity Recognition (NER)
# Definition: Train spaCy to recognize new entity types.
# Usage: Used in domain-specific NLP (e.g., medical, finance).

nlp = spacy.blank("en")  # Create blank NLP model
ner = nlp.add_pipe("ner")  # Add Named Entity Recognition pipeline

# Add new entity label
ner.add_label("TECH_COMPANY")

# Manually annotate text for training
training_data = [("Tesla is an innovative company.", {"entities": [(0, 5, "TECH_COMPANY")]})]

# Example training process (skipped for simplicity)
print("\n🛠️ Custom NER Model Ready for Training!")


# 📌 11. Extracting Keywords from Text
# Definition: Identifies important words.
# Usage: Used in search engines, text summarization.

important_words = [token.text for token in doc if token.is_alpha and not token.is_stop]
print("\n🔑 Extracted Keywords:", important_words)


# 📌 12. Sentiment Analysis with spaCy & TextBlob
# Definition: Determines positive or negative sentiment.
# Usage: Used in social media monitoring, customer reviews.

from textblob import TextBlob

text = "I really love this product, it's amazing!"
sentiment_score = TextBlob(text).sentiment.polarity

print("\n😊 Sentiment Score:", sentiment_score)  # >0 = Positive, <0 = Negative


# 📌 13. Text Summarization with spaCy
# Definition: Reduces text length while keeping key points.
# Usage: Used in AI news summarization, content generation.

text = """SpaceX, founded by Elon Musk, is revolutionizing space travel. 
They have successfully launched reusable rockets, reducing costs significantly. 
The company's goal is to make Mars colonization a reality."""

doc = nlp(text)
summary = " ".join([sent.text for sent in doc.sents][:2])  # Extract first two sentences

print("\n📝 Summarized Text:", summary)


# 📌 14. Named Entity Visualization
# Definition: Displays named entities in a graphical format.
# Usage: Used in AI dashboards, NLP research.

from spacy import displacy

text = "Google was founded in 1998 in California."
doc = nlp(text)

print("\n🎨 Named Entity Visualization:")
displacy.render(doc, style="ent", jupyter=True)


# 📌 15. Dependency Tree Visualization
# Definition: Displays sentence structure visually.
# Usage: Used in NLP research, language analysis.

displacy.render(doc, style="dep", jupyter=True)
