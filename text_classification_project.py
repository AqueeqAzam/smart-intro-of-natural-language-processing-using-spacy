from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import spacy

# Sample data (for text classification, not NER)
texts = [
    "Uber blew through $1 million a week",  # Label 0
    "Apple is looking at buying U.K. startup",  # Label 1
    "SpaceX launched its latest rocket successfully",  # Label 1
    "Russia is in Asia",  # Label 0
    "Google is working on a new AI model",  # Label 1
    "Amazon announced new products in its latest event",  # Label 1
    "Tesla is manufacturing a new electric car in Germany",  # Label 1
    "France is a country in Europe",  # Label 0
    "The Great Wall is in China",  # Label 0
]
labels = [0, 1, 1, 0, 1, 1, 1, 0, 0]  # Updated labels


# Initialize spaCy and the TF-IDF Vectorizer
nlp = spacy.load("en_core_web_sm")
vectorizer = TfidfVectorizer()

# Example function to extract named entities
def extract_entities(text):
    doc = nlp(text)
    return " ".join([ent.text for ent in doc.ents])

# Vectorize the text and train a simple classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),      # Convert text to TF-IDF features
    ('clf', LogisticRegression())      # Train a classifier
])

# Train the classifier
pipeline.fit(texts, labels)

# Predict new text
new_text = "Russia is not in asia"
predicted_label = pipeline.predict([new_text])
print(f"Predicted Label: {predicted_label[0]}")
