import heapq
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from operator import itemgetter
from typing import List, Dict, Tuple
from typing import Sequence
from abc import ABC

import numpy as np
import torch
from bert_score import BERTScorer
from nltk import PorterStemmer
from spacy.tokens import Doc, Span
from toolz import itertoolz
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PaddingStrategy


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(
        self,
        sents: List[Span]
    ):
        pass


class ContextualEmbedding(EmbeddingModel):

    def __init__(self, model, tokenizer_name, max_length, batch_size=32):
        self.model = model
        self.tokenizer = SpacyHuggingfaceTokenizer(tokenizer_name, max_length)
        self._device = model.device
        self.batch_size = batch_size

    def embed(
        self,
        sents: List[Span]
    ):
        spacy_embs_list = []
        for start_idx in range(0, len(sents), self.batch_size):
            batch = sents[start_idx: start_idx + self.batch_size]
            encoded_input, special_tokens_masks, token_alignments = self.tokenizer.batch_encode(batch)
            encoded_input = {k: v.to(self._device) for k, v in encoded_input.items()}
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                embeddings = model_output[0].cpu()
            for embs, mask, token_alignment \
                in zip(embeddings, special_tokens_masks, token_alignments):
                mask = torch.tensor(mask)
                embs = embs[mask == 0]  # Filter embeddings at special token positions
                spacy_embs = []
                for hf_idxs in token_alignment:
                    if hf_idxs is None:
                        pooled_embs = torch.zeros_like(embs[0])
                    else:
                        pooled_embs = embs[hf_idxs].mean(dim=0)  # Pool embeddings that map to the same spacy token
                    spacy_embs.append(pooled_embs.numpy())
                spacy_embs = np.stack(spacy_embs)
                spacy_embs = spacy_embs / np.linalg.norm(spacy_embs, axis=-1, keepdims=True)  # Normalize
                spacy_embs_list.append(spacy_embs)
        for embs, sent in zip(spacy_embs_list, sents):
            assert len(embs) == len(sent)
        return spacy_embs_list


class StaticEmbedding(EmbeddingModel):

    def embed(
        self,
        sents: List[Span]
    ):
        return [
            np.stack([t.vector / (t.vector_norm or 1) for t in sent])
            for sent in sents
        ]


class Aligner(ABC):
    @abstractmethod
    def align(
        self,
        source: Doc,
        targets: Sequence[Doc]
    ) -> List[Dict]:
        """Compute alignment from summary tokens to doc tokens
        Args:
            source: Source spaCy document
            targets: Target spaCy documents
        Returns: List of alignments, one for each target document"""
        pass


class EmbeddingAligner(Aligner):

    def __init__(
        self,
        embedding: EmbeddingModel,
        threshold: float,
        top_k: int,
        baseline_val=0
    ):
        self.threshold = threshold
        self.top_k = top_k
        self.embedding = embedding
        self.baseline_val = baseline_val

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
        if len(source) == 0:
            return [{} for _ in targets]
        all_sents = list(source.sents) + list(itertools.chain.from_iterable(target.sents for target in targets))
        chunk_sizes = [_iter_len(source.sents)] + \
                      [_iter_len(target.sents) for target in targets]
        all_sents_token_embeddings = self.embedding.embed(all_sents)
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
            for score, target_idx, source_idx in self._emb_sim_sparse(
                target_token_embeddings,
                source_token_embeddings,
            ):
                alignment[target_idx].append((source_idx, score))
            # TODO used argpartition to get nlargest
            for j in list(alignment):
                alignment[j] = heapq.nlargest(self.top_k, alignment[j], itemgetter(1))
            alignments.append(alignment)
        return alignments

    def _emb_sim_sparse(self, embs_1, embs_2):
        sim = embs_1 @ embs_2.T
        sim = (sim - self.baseline_val) / (1 - self.baseline_val)
        keep = sim > self.threshold
        keep_idxs_1, keep_idxs_2 = np.where(keep)
        keep_scores = sim[keep]
        return list(zip(keep_scores, keep_idxs_1, keep_idxs_2))


class BertscoreAligner(EmbeddingAligner):
    def __init__(
        self,
        threshold,
        top_k
    ):
        scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        model = scorer._model
        embedding = ContextualEmbedding(model, "roberta-large", 510)
        baseline_val = scorer.baseline_vals[2].item()

        super(BertscoreAligner, self).__init__(
            embedding, threshold, top_k, baseline_val
        )


class StaticEmbeddingAligner(EmbeddingAligner):
    def __init__(
        self,
        threshold,
        top_k
    ):
        embedding = StaticEmbedding()
        super(StaticEmbeddingAligner, self).__init__(
            embedding, threshold, top_k
        )


class NGramAligner(Aligner):

    def __init__(self):
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
        for sent in doc.sents:
            for n in range(1, len(list(sent))):
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
        token_is_available_1 = [True] * max_span_end_1  #
        matched_keys = list(set(ngram_spans_1.keys()) & set(ngram_spans_2.keys()))  # Matched normalized ngrams betwee
        matched_keys.sort(key=len, reverse=True)  # Process n-grams from longest to shortest

        alignment = defaultdict(list)  # Map from each matched span in text 1 to list of aligned spans in text 2
        for key in matched_keys:
            spans_1 = ngram_spans_1[key]
            spans_2 = ngram_spans_2[key]
            available_spans_1 = [span for span in spans_1 if all(token_is_available_1[slice(*span)])]
            matched_spans_1 = []
            if available_spans_1 and spans_2:
                # if ngram can be matched to available spans in both sequences
                for span in available_spans_1:
                    # It's possible that these newly matched spans may be overlapping with one another, so
                    # check that token positions still available (only one span allowed ber token in text 1):
                    if all(token_is_available_1[slice(*span)]):
                        matched_spans_1.append(span)
                        token_is_available_1[slice(*span)] = [False] * (span[1] - span[0])
            for span1 in matched_spans_1:
                alignment[span1] = spans_2

        return alignment


class SpacyHuggingfaceTokenizer:
    def __init__(
        self,
        model_name,
        max_length
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.max_length = max_length

    def batch_encode(
        self,
        sents: List[Span]
    ):
        token_alignments = []
        token_ids_list = []

        # Tokenize each sentence and special tokens.
        for sent in sents:
            hf_tokens, token_alignment = self.tokenize(sent)
            token_alignments.append(token_alignment)
            token_ids = self.tokenizer.convert_tokens_to_ids(hf_tokens)
            encoding = self.tokenizer.prepare_for_model(
                token_ids,
                add_special_tokens=True,
                padding=False,
            )
            token_ids_list.append(encoding['input_ids'])

        # Add padding
        max_length = max(map(len, token_ids_list))
        attention_mask = []
        input_ids = []
        special_tokens_masks = []
        for token_ids in token_ids_list:
            encoding = self.tokenizer.prepare_for_model(
                token_ids,
                padding=PaddingStrategy.MAX_LENGTH,
                max_length=max_length,
                add_special_tokens=False
            )
            input_ids.append(encoding['input_ids'])
            attention_mask.append(encoding['attention_mask'])
            special_tokens_masks.append(
                self.tokenizer.get_special_tokens_mask(
                    encoding['input_ids'],
                    already_has_special_tokens=True
                )
            )

        encoded = {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask)
        }
        return encoded, special_tokens_masks, token_alignments

    def tokenize(
        self,
        sent
    ):
        """Convert spacy sentence to huggingface tokens and compute the alignment"""
        hf_tokens = []
        token_alignment = []
        for i, token in enumerate(sent):
            # "Tokenize" each word individually, so as to track the alignment between spaCy/HF tokens
            # Prefix all tokens with a space except the first one in the sentence
            if i == 0:
                token_text = token.text
            else:
                token_text = ' ' + token.text
            start_hf_idx = len(hf_tokens)
            word_tokens = self.tokenizer.tokenize(token_text)
            end_hf_idx = len(hf_tokens) + len(word_tokens)
            if end_hf_idx < self.max_length:
                hf_tokens.extend(word_tokens)
                hf_idxs = list(range(start_hf_idx, end_hf_idx))
            else:
                hf_idxs = None
            token_alignment.append(hf_idxs)
        return hf_tokens, token_alignment


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
