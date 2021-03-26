from unittest import TestCase

import spacy

from align import NGramAligner


class TestNGramAligner(TestCase):
    def test_ngram_match(self):
        # TODO Add more tests
        nlp = spacy.load("en_core_web_sm")
        source = nlp("a zebra elephant street the blue house.")
        targets = [
            nlp("a zebra elephant street the blue house."),
            nlp("blue")
        ]
        aligner = NGramAligner(10)
        print(aligner.align(source, targets))


