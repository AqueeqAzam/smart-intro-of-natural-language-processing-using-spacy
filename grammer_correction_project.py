# !pip install language_tool_python
import spacy
import language_tool_python

# Initialize spaCy and LanguageTool
nlp = spacy.load("en_core_web_sm")
tool = language_tool_python.LanguageTool('en-US')

def preprocess_text(text):
    """
    Preprocess text using spaCy to remove unnecessary whitespace and tokenize.
    
    Parameters:
    text (str): The input text to preprocess.
    
    Returns:
    str: Preprocessed text.
    """
    doc = nlp(text)
    # Reconstruct the text without extra spaces
    return " ".join([token.text for token in doc if not token.is_punct and not token.is_space])

def correct_grammar(text):
    """
    Correct grammatical errors using LanguageTool.
    
    Parameters:
    text (str): The input text to correct.
    
    Returns:
    str: Text with grammatical errors corrected.
    """
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

def main():
    print("Grammar Correction Tool")
    
    # Input text
    input_text = input("Enter text with grammatical errors: ")
    
    # Preprocess the text
    preprocessed_text = preprocess_text(input_text)
    print("\nPreprocessed Text:")
    print(preprocessed_text)
    
    # Correct grammar
    corrected_text = correct_grammar(preprocessed_text)
    print("\nCorrected Text:")
    print(corrected_text)

if __name__ == "__main__":
    main()
