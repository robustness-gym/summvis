import re


def clean_text(text):
    split_punct = re.escape(r'()')
    return ' '.join(re.findall(rf"[^\s{split_punct}]+|[{split_punct}]", text))
    # Ensure parentheses are probably separated by spaCy tokenizer for CNN/DailyMail dataset.
    return text.replace("(", "( ").replace(")", ") ")

