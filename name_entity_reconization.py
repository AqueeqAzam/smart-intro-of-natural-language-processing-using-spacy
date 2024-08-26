import spacy

# Load the spaCy language model
nlp = spacy.load('en_core_web_sm')

def extract_entities(text):
  # Process the text with spaCy
  doc = nlp(text)
  
  # Extract entities
  entities = [(ent.text, ent.label_) for ent in doc.ents]
  return entities

# Sample text for entity extraction
text = " Apple is looking at buying U.K. startup for $1 billion.Elon Musk announced that SpaceX will send humans to Mars by 2024. The New York Times reported that the event will take place on August 26, 2024."
entities = extract_entities(text)
print(entities)
