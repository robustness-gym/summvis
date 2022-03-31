import argparse
import json
import operator
import os
import re
from pathlib import Path

import spacy
import spacy.lang.en
import streamlit as st
from meerkat import DataPanel
from spacy.tokens import Doc

from align import NGramAligner, BertscoreAligner, StaticEmbeddingAligner
from components import MainView
from utils import clean_text

MIN_SEMANTIC_SIM_THRESHOLD = 0.1
MAX_SEMANTIC_SIM_TOP_K = 10

Doc.set_extension("name", default=None, force=True)
Doc.set_extension("column", default=None, force=True)


class Instance():
    def __init__(self, id_, document, reference, preds, data=None):
        self.id = id_
        self.document = document
        self.reference = reference
        self.preds = preds
        self.data = data


@st.cache(allow_output_mutation=True)
def load_from_index(filename, index):
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line.strip())


def _nlp_key(x: spacy.Language):
    return str(x.path)


@st.cache(allow_output_mutation=True, hash_funcs={spacy.lang.en.English: _nlp_key})
def load_dataset(path: str, nlp: spacy.Language):
    if path.endswith('.jsonl'):
        return DataPanel.from_jsonl(path)
    try:
        return DataPanel.read(path, nlp=nlp)
    except NotADirectoryError:
        return DataPanel.from_jsonl(path)


@st.cache(allow_output_mutation=True)
def get_nlp():
    try:
        nlp = spacy.load("en_core_web_lg")
    except:
        nlp = spacy.load("en_core_web_sm")
        is_lg = False
    else:
        is_lg = True
    nlp.add_pipe('sentencizer', before="parser")
    return nlp, is_lg


def retrieve(dataset, index, filename=None):
    if index >= len(dataset):
        st.error(f"Index {index} exceeds dataset length.")

    eval_dataset = None
    if filename:
        # TODO Handle this through dedicated fields
        if "cnn_dailymail" in filename:
            eval_dataset = "cnndm"
        elif "xsum" in filename:
            eval_dataset = "xsum"

    data = dataset[index]
    id_ = data.get('id', '')

    try:
        document = data['spacy:document']
    except KeyError:
        if not is_lg:
            st.error("'en_core_web_lg model' is required unless loading from cached file."
                     "To install: 'python -m spacy download en_core_web_lg'")
        try:
            text = data['document']
        except KeyError:
            text = data['article']
        if not text:
            st.error("Document is blank")
            return
        document = nlp(text if args.no_clean else clean_text(text))
    document._.name = "Document"
    document._.column = "document"

    try:
        reference = data['spacy:summary:reference']

    except KeyError:
        if not is_lg:
            st.error("'en_core_web_lg model' is required unless loading from cached file."
                     "To install: 'python -m spacy download en_core_web_lg'")
        try:
            text = data['summary'] if 'summary' in data else data['summary:reference']
        except KeyError:
            text = data.get('highlights')
        if text:
            reference = nlp(text if args.no_clean else clean_text(text))
        else:
            reference = None
    if reference is not None:
        reference._.name = "Reference"
        reference._.column = "summary:reference"

    model_names = set()
    for k in data:
        m = re.match('(preprocessed_)?summary:(?P<model>.*)', k)
        if m:
            model_name = m.group('model')
            if model_name != 'reference':
                model_names.add(model_name)

    preds = []
    for model_name in model_names:
        try:
            pred = data[f"spacy:summary:{model_name}"]
        except KeyError:
            if not is_lg:
                st.error("'en_core_web_lg model' is required unless loading from cached file."
                         "To install: 'python -m spacy download en_core_web_lg'")
            text = data[f"summary:{model_name}"]
            pred = nlp(text if args.no_clean else clean_text(text))

        parts = model_name.split("-")
        primary_sort = 0
        if len(parts) == 2:
            model, train_dataset = parts
            if train_dataset == eval_dataset:
                formatted_model_name = model.upper()
            else:
                formatted_model_name = f"{model.upper()} ({train_dataset.upper()}-trained)"
                if train_dataset in ["xsum", "cnndm"]:
                    primary_sort = 1
                else:
                    primary_sort = 2
        else:
            formatted_model_name = model_name.upper()
        pred._.name = formatted_model_name
        pred._.column = f"summary:{model_name}"
        preds.append(
            ((primary_sort, formatted_model_name), pred)
        )

    preds = [pred for _, pred in sorted(preds)]

    return Instance(
        id_=id_,
        document=document,
        reference=reference,
        preds=preds,
        data=data,
    )


def filter_alignment(alignment, threshold, top_k):
    filtered_alignment = {}
    for k, v in alignment.items():
        filtered_matches = [(match_idx, score) for match_idx, score in v if score >= threshold]
        if filtered_matches:
            filtered_alignment[k] = sorted(filtered_matches, key=operator.itemgetter(1), reverse=True)[:top_k]
    return filtered_alignment


def select_comparison(example):
    all_summaries = []

    if example.reference:
        all_summaries.append(example.reference)
    if example.preds:
        all_summaries.extend(example.preds)

    from_documents = [example.document]
    if example.reference:
        from_documents.append(example.reference)
    document_names = [document._.name for document in from_documents]
    select_document_name = sidebar_placeholder_from.selectbox(
        label="Comparison FROM:",
        options=document_names
    )
    document_index = document_names.index(select_document_name)
    selected_document = from_documents[document_index]

    remaining_summaries = [summary for summary in all_summaries if
                           summary._.name != selected_document._.name]
    remaining_summary_names = [summary._.name for summary in remaining_summaries]

    selected_summary_names = sidebar_placeholder_to.multiselect(
        'Comparison TO:',
        remaining_summary_names,
        remaining_summary_names
    )
    selected_summaries = []
    for summary_name in selected_summary_names:
        summary_index = remaining_summary_names.index(summary_name)
        selected_summaries.append(remaining_summaries[summary_index])
    return selected_document, selected_summaries


def show_main(example):
    # Get user input

    semantic_sim_type = st.sidebar.radio(
        "Semantic similarity type:",
        ["Contextual embedding", "Static embedding"]
    )
    semantic_sim_threshold = st.sidebar.slider(
        "Semantic similarity threshold:",
        min_value=MIN_SEMANTIC_SIM_THRESHOLD,
        max_value=1.0,
        step=0.1,
        value=0.2,
    )
    semantic_sim_top_k = st.sidebar.slider(
        "Semantic similarity top-k:",
        min_value=1,
        max_value=MAX_SEMANTIC_SIM_TOP_K,
        step=1,
        value=10,
    )

    document, summaries = select_comparison(example)
    layout = st.sidebar.radio("Layout:", ["Vertical", "Horizontal"]).lower()
    scroll = True
    gray_out_stopwords = st.sidebar.checkbox(label="Gray out stopwords", value=True)

    # Gather data
    try:
        lexical_alignments = [
            example.data[f'{NGramAligner.__name__}:spacy:{document._.column}:spacy:{summary._.column}']
            for summary in summaries
        ]
    except KeyError:
        lexical_alignments = NGramAligner().align(document, summaries)

    if semantic_sim_type == "Static embedding":
        try:
            semantic_alignments = [
                example.data[f'{StaticEmbeddingAligner.__name__}:spacy:{document._.column}:spacy:{summary._.column}']
                for summary in summaries
            ]
        except KeyError:
            semantic_alignments = StaticEmbeddingAligner(
                semantic_sim_threshold,
                semantic_sim_top_k).align(
                document,
                summaries
            )
    else:
        try:
            semantic_alignments = [
                example.data[f'{BertscoreAligner.__name__}:spacy:{document._.column}:spacy:{summary._.column}']
                for summary in summaries
            ]
        except KeyError:
            semantic_alignments = BertscoreAligner(semantic_sim_threshold,
                                                   semantic_sim_top_k).align(document,
                                                                             summaries)

    MainView(
        document,
        summaries,
        semantic_alignments,
        lexical_alignments,
        layout,
        scroll,
        gray_out_stopwords,
    ).show(height=720)


if __name__ == "__main__":

    st.set_page_config(layout="wide")

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data')
    parser.add_argument('--no_clean', action='store_true', default=False,
                        help="Do not clean text (remove extraneous spaces, newlines).")
    args = parser.parse_args()

    nlp, is_lg = get_nlp()

    path = Path(args.path)
    path_dir = path.parent
    all_files = set(map(os.path.basename, path_dir.glob('*')))
    files = sorted([
        fname for fname in all_files if not (fname.endswith(".py") or fname.startswith("."))
    ])
    if path.is_file:
        try:
            file_index = files.index(path.name)
        except:
            raise FileNotFoundError(f"File not found: {path.name}")
    else:
        file_index = 0
    col1, col2 = st.beta_columns((3, 1))
    filename = col1.selectbox(label="File:", options=files, index=file_index)
    dataset = load_dataset(str(path_dir / filename), nlp=nlp)

    dataset_size = len(dataset)
    query = col2.number_input(f"Index (Size: {dataset_size}):", value=0, min_value=0, max_value=dataset_size - 1)

    sidebar_placeholder_from = st.sidebar.empty()
    sidebar_placeholder_to = st.sidebar.empty()

    if query is not None:
        example = retrieve(dataset, query, filename)
        if example:
            show_main(example)
