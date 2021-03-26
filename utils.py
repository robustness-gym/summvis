import re


def preprocess_text(text):
    split_punct = re.escape(r'!"#$%&()*+,-\./:;<=>?@[\]^_`{|}~)')
    return ' '.join(re.findall(rf"[\w']+|[{split_punct}]", text))
