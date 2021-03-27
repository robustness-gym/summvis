# import streamlit as st
import heapq
import itertools
from collections import defaultdict
from operator import itemgetter
from typing import List, Dict, Tuple
from typing import Sequence

import numpy as np
import torch
from bert_score import BERTScorer
from spacy.tokens import Doc, Span
from spacy.tokens import Token
from toolz import itertoolz
from transformers import AutoTokenizer

from nltk import PorterStemmer


class BertscoreAligner():

    def __init__(
        self,
        threshold: float,
        top_k: int
    ):
        self.threshold = threshold
        self.top_k = top_k
        scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        self.model = scorer._model
        self._device = self.model.device
        self.baseline_val = scorer.baseline_vals[2].item()
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large", add_prefix_space=False)
        self.tokenizer_add_prefix = AutoTokenizer.from_pretrained("roberta-large", add_prefix_space=True)

    def align(
        self,
        source: Doc,
        targets: Sequence[Doc]
    ) -> List[Dict]:
        """Compute alignment from summary tokens to doc tokens with greatest semantic similarity
        Args:
            source: Source spaCy document
            targets: Target spaCy documents
        Returns: List of alignments, one for each target document
        """
        all_sents = list(source.sents) + list(itertools.chain.from_iterable(target.sents for target in targets))
        chunk_sizes = [_iter_len(source.sents)] + \
                      [_iter_len(target.sents) for target in targets]
        all_sents_token_embeddings = self._embed(all_sents)
        chunked_sents_token_embeddings = _split(all_sents_token_embeddings, chunk_sizes)
        source_sent_token_embeddings = chunked_sents_token_embeddings[0]
        source_token_embeddings = np.concatenate(source_sent_token_embeddings)
        for token_idx, token in enumerate(source):
            if token.is_stop or token.is_punct:
                source_token_embeddings[token_idx] = 0
        alignments = []
        for i, target in enumerate(targets):
            target_sent_token_embeddings = chunked_sents_token_embeddings[i + 1]
            target_token_embeddings = np.concatenate(target_sent_token_embeddings)
            for token_idx, token in enumerate(target):
                if token.is_stop or token.is_punct:
                    target_token_embeddings[token_idx] = 0
            alignment = defaultdict(list)
            # todo: ADD mask
            for score, target_idx, source_idx in self._emb_sim_sparse(
                target_token_embeddings,
                source_token_embeddings,
                self.threshold
            ):
                alignment[target_idx].append((source_idx, score))
            # TODO used argpartition to get nlargest
            for j in list(alignment):
                alignment[j] = heapq.nlargest(self.top_k, alignment[j], itemgetter(1))
            alignments.append(alignment)
        return alignments

    def _embed(
        self,
        sents: List[Span]
    ):
        # TODO:
        #  - Handle contraction tokenization better
        #  - Don't add prefix for end of sentence tokens

        # First produce an alignment between spacey tokens and roberta tokens
        token_alignments = []
        for sent in sents:
            token_alignment = []
            roberta_next_index = 0
            for i, token in enumerate(sent):
                # "Tokenize" each word individually, so as to track the alignment between spaCy/HF tokens
                if i == 0:
                    # The first token in the sentence is not prefixed by a space. Adding prefix to this can cause
                    # unexpected behavior is this is OOD
                    tokenizer = self.tokenizer
                else:
                    tokenizer = self.tokenizer_add_prefix
                n_roberta_tokens = len(tokenizer.tokenize(token.text))
                token_alignment.append(list(range(roberta_next_index, roberta_next_index + n_roberta_tokens)))
                roberta_next_index += n_roberta_tokens
            token_alignments.append(token_alignment)

        input_sents = [
            " ".join(t.text for t in sent) for sent in sents
        ]
        encoded_input = self.tokenizer(
            input_sents,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input.to(self._device))
            padded_embeddings = model_output[0].cpu()

        attention_mask = encoded_input['attention_mask']
        embeddings = []
        for emb_pad, mask in zip(padded_embeddings, attention_mask):
            emb_nopad = emb_pad[mask == 1]  # Remove embeddings at <pad> positions
            emb_nopad = emb_nopad[1:-1]  # Remove bos, eos
            embeddings.append(emb_nopad)
        spacy_embs_all = []
        for sent_embs, token_alignment in zip(embeddings, token_alignments):
            spacy_embs = []
            for roberta_tok_idxs in token_alignment:
                pooled_embs = sent_embs[roberta_tok_idxs].mean(dim=0)
                spacy_embs.append(pooled_embs.numpy())
            spacy_embs = np.stack(spacy_embs)
            spacy_embs = spacy_embs / np.linalg.norm(spacy_embs, axis=-1, keepdims=True)
            spacy_embs_all.append(spacy_embs)
        for embs, sent in zip(spacy_embs_all, sents):
            assert len(embs) == len(sent)

        return spacy_embs_all

    def _emb_sim_sparse(self, embs_1, embs_2, sim_threshold):
        sim = embs_1 @ embs_2.T
        sim = (sim - self.baseline_val) / (1 - self.baseline_val)
        keep = sim > sim_threshold
        keep_idxs_1, keep_idxs_2 = np.where(keep)
        keep_scores = sim[keep]
        return list(zip(keep_scores, keep_idxs_1, keep_idxs_2))


class StaticEmbeddingAligner():

    def __init__(
        self,
        threshold: float,
        top_k: int
    ):
        self.threshold = threshold
        self.top_k = top_k

    def align(
        self,
        source: Doc,
        targets: Sequence[Doc]
    ) -> List[Dict]:

        source_embedding = np.stack([t.vector / (t.vector_norm or 1) for t in source])
        for token_idx, token in enumerate(source):
            if token.is_stop or token.is_punct:
                source_embedding[token_idx] = 0
        alignments = []
        for target in targets:
            target_embedding = np.stack([t.vector / (t.vector_norm or 1) for t in target])
            for token_idx, token in enumerate(target):
                if token.is_stop or token.is_punct:
                    target_embedding[token_idx] = 0
            alignment = defaultdict(list)
            for target_token_idx, source_token_idx, sim in self._sim(
                target_embedding, source_embedding
            ):
                alignment[target_token_idx].append((source_token_idx, sim))
            for k in alignment:
                alignment[k] = heapq.nlargest(self.top_k, alignment[k], itemgetter(1))
            alignments.append(alignment)
        return alignments

    def _sim(self, embs1, embs2):
        sim = embs1 @ embs2.T
        keep = sim > self.threshold
        keep_idxs_1, keep_idxs_2 = np.where(keep)
        keep_scores = sim[keep]
        return list(zip(keep_idxs_1, keep_idxs_2, keep_scores))


class NGramAligner():

    def __init__(self, max_n):
        self.max_n = max_n
        self.stemmer = PorterStemmer()

    def align(
        self,
        source: Doc,
        targets: List[Doc],
    ) -> List[Dict]:

        alignments = []
        source_ngram_spans = self._get_ngram_spans(source)
        for target in targets:
            target_ngram_spans = self._get_ngram_spans(target)
            alignments.append(
                self._align_ngrams(target_ngram_spans, source_ngram_spans)
            )
        return alignments

    def _get_ngram_spans(
        self,
        doc: Doc,
    ):
        ngrams = []
        for n in range(1, self.max_n + 1):
            for sent in doc.sents:
                tokens = [t for t in sent if not (t.is_stop or t.is_punct)]
                ngrams.extend(_ngrams(tokens, n))

        def ngram_key(ngram):
            return tuple(self.stemmer.stem(token.text).lower() for token in ngram)

        key_to_ngrams = itertoolz.groupby(ngram_key, ngrams)
        key_to_spans = {}
        for k, grouped_ngrams in key_to_ngrams.items():
            key_to_spans[k] = [
                (ngram[0].i, ngram[-1].i + 1)
                for ngram in grouped_ngrams
            ]
        return key_to_spans

    def _align_ngrams(
        self,
        ngram_spans_1: Dict[Tuple[str], List[Tuple[int, int]]],
        ngram_spans_2: Dict[Tuple[str], List[Tuple[int, int]]]
    ) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """Align ngram spans between two documents
        Args:
            ngram_spans_1: Map from (normalized_token1, normalized_token2, ...) n-gram tuple to a list of token spans
                of format (start_pos, end_pos)
            ngram_spans_2: Same format as above, but for second text
        Returns: map from each (start, end) span in text 1 to list of aligned (start, end) spans in text 2
        """
        if not ngram_spans_1 or not ngram_spans_2:
            return {}
        max_span_end_1 = max(span[1] for span in itertools.chain.from_iterable(ngram_spans_1.values()))
        max_span_end_2 = max(span[1] for span in itertools.chain.from_iterable(ngram_spans_2.values()))
        token_is_available_1 = [True] * max_span_end_1  #
        token_is_available_2 = [True] * max_span_end_2
        matched_keys = list(set(ngram_spans_1.keys()) & set(ngram_spans_2.keys()))  # Matched normalized ngrams betwee
        matched_keys.sort(key=len, reverse=True)  # Process n-grams from longest to shortest

        alignment = defaultdict(list)  # Map from each matched span in text 1 to list of aligned spans in text 2
        for key in matched_keys:
            spans_1 = ngram_spans_1[key]
            spans_2 = ngram_spans_2[key]
            available_spans_1 = [span for span in spans_1 if all(token_is_available_1[slice(*span)])]
            matched_spans_1 = []
            matched_spans_2 = []
            if available_spans_1:
                # if ngram can be matched to available spans in both sequences
                for span in available_spans_1:
                    # It's possible that these newly matched spans may be overlapping with one another, so
                    # check that token positions still available:
                    if all(token_is_available_1[slice(*span)]):
                        matched_spans_1.append(span)
                        token_is_available_1[slice(*span)] = [False] * (span[1] - span[0])
                for span in spans_2:
                    # It's possible that these newly matched spans may be overlapping with one another, so
                    # check that token positions still available:
                    # if all(token_is_available_2[slice(*span)]):
                    matched_spans_2.append(span)
                        # token_is_available_2[slice(*span)] = [False] * (span[1] - span[0])
            for span1 in matched_spans_1:
                alignment[span1] = matched_spans_2

        return alignment


def _split(data, sizes):
    it = iter(data)
    return [[next(it) for _ in range(size)] for size in sizes]


def _iter_len(it):
    return sum(1 for _ in it)

    # TODO set up batching
    # To get top K axis and value per row: https://stackoverflow.com/questions/42832711/using-np-argpartition-to-index-values-in-a-multidimensional-array


def _ngrams(tokens, n):
    for i in range(len(tokens) - n + 1):
        yield tokens[i:i + n]


if __name__ == "__main__":
    import spacy

    nlp = spacy.load("en_core_web_sm")

    # source = "His injuries are not believed to be life threatening ."
    source = "(CNN)Singer-songwriter David Crosby hit a jogger with his car Sunday evening, a spokesman said. The accident happened in Santa Ynez, California, near where Crosby lives. Crosby was driving at approximately 50 mph when he struck the jogger, according to California Highway Patrol Spokesman Don Clotworthy. The posted speed limit was 55. The jogger suffered multiple fractures, and was airlifted to a hospital in Santa Barbara, Clotworthy said. His injuries are not believed to be life threatening. \"Mr. Crosby was cooperative with authorities and he was not impaired or intoxicated in any way. Mr. Crosby did not see the jogger because of the sun,\" said Clotworthy. According to the spokesman, the jogger and Crosby were on the same side of the road. Pedestrians are supposed to be on the left side of the road walking toward traffic, Clotworthy said. Joggers are considered pedestrians. Crosby is known for weaving multilayered harmonies over sweet melodies. He belongs to the celebrated rock group Crosby, Stills & Nash. \"David Crosby is obviously very upset that he accidentally hit anyone. And, based off of initial reports, he is relieved that the injuries to the gentleman were not life threatening,\" said Michael Jensen, a Crosby spokesman. \"He wishes the jogger a very speedy recovery.\""
    targets = [
        "Singer-songwriter David Crosby says he's \"relieved\" and \"appreciative\" that a jogger suffered non-life-threatening injuries after being hit by his car Sunday evening. A California Highway Patrol spokesman tells CNN that Crosby, who was driving around 50mph at the time, \"was cooperative with authorities and he was not impaired or intoxicated in any way. Mr. Crosby did not see the jogger because of the sun.\" The spokesman says the jogger was on the left side of the road near Crosby's home in Santa Ynez, and pedestrians are supposed to be on the left side of the road when walking toward traffic. \"David Crosby is obviously very upset that he accidentally hit anyone. And, based off of initial reports, he is relieved that the injuries to the gentleman were not life-threatening,\" a Crosby rep tells CNN. \"He wishes the jogger a very speedy recovery.\""
    ]
    # source = "He's extremely relieved."
    # targets = ["He is relieved. He is extremely relieved."]
    aligners = [
        # BertscoreAligner(0.0, 3),
        # StaticEmbeddingAligner(0.1, 3),
    ]
    source_doc = nlp(source)
    target_docs = list(map(nlp, targets))
    # token_maps = bertscore_sim(source_doc, target_docs, 0.1)
    for aligner in aligners:
        print(aligner.__class__)
        alignments = aligner.align(source_doc, target_docs)
        for target_doc, alignment in zip(target_docs, alignments):
            for target_idx, source_matches in alignment.items():
                for source_idx, score in source_matches:
                    print(f"{score:.3f} {target_doc[target_idx]} {source_doc[source_idx]}")

    aligner = NGramAligner(10)
    alignments = aligner.align(source_doc, target_docs)
    for target_doc, alignment in zip(target_docs, alignments):
        for target_span, source_spans in sorted(alignment.items()):
            print("target:", target_doc[slice(*target_span)], target_span)
            for source_span in source_spans:
                print("\tsource:", source_doc[slice(*source_span)])
