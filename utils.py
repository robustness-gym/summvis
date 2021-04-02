import re


def preprocess_text(text):
    split_punct = re.escape(r'()')
    return ' '.join(re.findall(rf"[^\s{split_punct}]+|[{split_punct}]", text))
