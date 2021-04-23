import argparse
import json
import operator
import os
import re
from pathlib import Path

import spacy
import streamlit as st
import streamlit.components.v1 as components
from htbuilder import styles, div
from robustnessgym import Dataset, Identifier
from robustnessgym import Spacy
from spacy.tokens import Doc

from align import NGramAligner, BertscoreAligner, StaticEmbeddingAligner
from components import main_view
from preprocessing import NGramAlignerCap, StaticEmbeddingAlignerCap, \
    BertscoreAlignerCap
from preprocessing import _spacy_decode, _spacy_encode
from utils import preprocess_text

MIN_SEMANTIC_SIM_THRESHOLD = 0.1
MAX_SEMANTIC_SIM_TOP_K = 10

Doc.set_extension("name", default=None, force=True)
Doc.set_extension("column", default=None, force=True)


class Instance():
    def __init__(self, id_, document, reference, preds, index=None, data=None):
        self.id = id_
        self.document = document
        self.reference = reference
        self.preds = preds
        self.index = index
        self.data = data


@st.cache(allow_output_mutation=True)
def load_from_index(filename, index):
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line.strip())


@st.cache(allow_output_mutation=True)
def load_dataset(path: str):
    if path.endswith('.jsonl'):
        return Dataset.from_jsonl(path)
    try:
        return Dataset.load_from_disk(path)
    except NotADirectoryError:
        return Dataset.from_jsonl(path)


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
        document = rg_spacy.decode(
            data[rg_spacy.identifier(columns=['preprocessed_document'])]
        )
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
        document = nlp(preprocess_text(text))
    document._.name = "Document"
    document._.column = "document"

    try:
        reference = rg_spacy.decode(
            data[rg_spacy.identifier(columns=['preprocessed_summary:reference'])]
        )
    except KeyError:
        if not is_lg:
            st.error("'en_core_web_lg model' is required unless loading from cached file."
                     "To install: 'python -m spacy download en_core_web_lg'")
        try:
            text = data['summary'] if 'summary' in data else data['summary:reference']
        except KeyError:
            text = data['highlights']
        reference = nlp(preprocess_text(text))
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
            pred = rg_spacy.decode(
                data[rg_spacy.identifier(columns=[f"preprocessed_summary:{model_name}"])]
            )
        except KeyError:
            if not is_lg:
                st.error("'en_core_web_lg model' is required unless loading from cached file."
                         "To install: 'python -m spacy download en_core_web_lg'")
            pred = nlp(preprocess_text(data[f"summary:{model_name}"]))

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
        index=data['index'] if 'index' in data else None,
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
    all_summaries = [example.reference] + example.preds

    from_documents = [example.document, example.reference]
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


def show_html(*elements, width=None, height=None, **kwargs):
    out = div(style=styles(
        **kwargs
    ))(elements)
    html = str(out)
    st.components.v1.html(html, width=width, height=height, scrolling=True)


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
    # if layout == "horizontal":
    #     scroll = st.sidebar.checkbox(label="Scroll sections", value=True)
    # else:
    scroll = True
    gray_stopwords = st.sidebar.checkbox(label="Gray out stopwords", value=True)

    # Gather data
    try:
        lexical_alignments = [
            NGramAlignerCap.decode(
                example.data[
                    Identifier(NGramAlignerCap.__name__)(
                        columns=[
                            f'preprocessed_{document._.column}',
                            f'preprocessed_{summary._.column}',
                        ]
                    )
                ])[0]
            for summary in summaries
        ]
        lexical_alignments = [
            {k: [(pair[0], int(pair[1])) for pair in v]
             for k, v in d.items()}
            for d in lexical_alignments
        ]
    except KeyError:
        lexical_alignments = NGramAligner().align(document, summaries)

    if semantic_sim_type == "Static embedding":
        try:
            semantic_alignments = [
                StaticEmbeddingAlignerCap.decode(
                    example.data[
                        Identifier(StaticEmbeddingAlignerCap.__name__)(
                            threshold=MIN_SEMANTIC_SIM_THRESHOLD,
                            top_k=MAX_SEMANTIC_SIM_TOP_K,
                            columns=[
                                f'preprocessed_{document._.column}',
                                f'preprocessed_{summary._.column}',
                            ]
                        )
                    ])[0]
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
            semantic_alignments = [
                filter_alignment(alignment, semantic_sim_threshold, semantic_sim_top_k)
                for alignment in semantic_alignments
            ]
    else:
        try:
            semantic_alignments = [
                BertscoreAlignerCap.decode(
                    example.data[
                        Identifier(BertscoreAlignerCap.__name__)(
                            threshold=MIN_SEMANTIC_SIM_THRESHOLD,
                            top_k=MAX_SEMANTIC_SIM_TOP_K,
                            columns=[
                                f'preprocessed_{document._.column}',
                                f'preprocessed_{summary._.column}',
                            ]
                        )
                    ])[0]
                for summary in summaries
            ]
        except KeyError:
            semantic_alignments = BertscoreAligner(semantic_sim_threshold,
                                                   semantic_sim_top_k).align(document,
                                                                             summaries)
        else:
            semantic_alignments = [
                filter_alignment(alignment, semantic_sim_threshold, semantic_sim_top_k)
                for alignment in semantic_alignments
            ]

    show_html(
        *main_view(
            document,
            summaries,
            semantic_alignments,
            lexical_alignments,
            layout,
            scroll,
            gray_stopwords
        ),
        height=850
    )


if __name__ == "__main__":

    st.set_page_config(layout="wide")

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data')
    parser.add_argument('--file', type=str)
    args = parser.parse_args()

    nlp, is_lg = get_nlp()

    Spacy.encode = _spacy_encode
    Spacy.decode = _spacy_decode
    rg_spacy = Spacy(nlp=nlp)

    path = Path(args.path)
    all_files = set(map(os.path.basename, path.glob('*')))
    files = sorted([
        fname for fname in all_files if not (fname.endswith(".py") or fname.startswith("."))
    ])
    if args.file:
        try:
            file_index = files.index(args.input)
        except:
            raise FileNotFoundError(f"File not found: {args.input}")
    else:
        file_index = 0
        col1, col2 = st.beta_columns((3, 1))
    filename = col1.selectbox(label="File:", options=files, index=file_index)
    dataset = load_dataset(str(path / filename))

    dataset_size = len(dataset)
    query = col2.number_input(f"Index (Size: {dataset_size}):", value=0, min_value=0, max_value=dataset_size - 1)

    sidebar_placeholder_from = st.sidebar.empty()
    sidebar_placeholder_to = st.sidebar.empty()

    if query is not None:
        example = retrieve(dataset, query, filename)
        if example:
            show_main(example)
