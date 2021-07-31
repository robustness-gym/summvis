def preprocess_text(text):
    # Ensure parentheses are probably separated by spaCy tokenizer for CNN/DailyMail dataset.
    return text.replace("(", "( ").replace(")", ") ")
