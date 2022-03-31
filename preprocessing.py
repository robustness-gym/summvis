import logging
from argparse import ArgumentParser
from typing import List

from meerkat import DataPanel, SpacyColumn
from meerkat.logging.utils import set_logging_level
from spacy import load

from align import BertscoreAligner, NGramAligner, StaticEmbeddingAligner, Aligner
from utils import clean_text

set_logging_level('critical')
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)


def _run_aligners(
    dataset: DataPanel,
    aligners: List[Aligner],
    doc_column: str,
    reference_column: str,
    summary_columns: List[str] = None,
):
    if not summary_columns:
        summary_columns = []

    to_columns = []
    if reference_column is not None:
        to_columns.append(reference_column)
    to_columns.extend(summary_columns)

    for aligner in aligners:

        # Run the aligner on (document, summary) pairs
        dataset = dataset.update(
            lambda x: {
                f'{type(aligner).__name__}:{doc_column}:{to_columns}':
                    aligner.align(
                        x[doc_column],
                        [x[col] for col in to_columns],
                    ),
            },
        )

        if reference_column is not None and len(summary_columns):
            # Run the aligner on (reference, summary) pairs
            dataset = dataset.update(
                lambda x: {
                    f'{type(aligner).__name__}:{reference_column}:{summary_columns}': aligner.align(
                        x[reference_column],
                        [x[col] for col in summary_columns],
                    ),
                },
            )

        if len(to_columns) > 1:
            # Instead of having one column for (document, summary) comparisons, split
            # off into (1 + |summary_columns|) total columns, one for each comparison

            # Retrieve the (document, summary) column
            doc_summary_column = dataset[f'{type(aligner).__name__}:{doc_column}:{to_columns}']

            for i, col in enumerate(to_columns):
                # Add as a new column after encoding with the aligner's `encode` method
                dataset.add_column(
                    f'{type(aligner).__name__}:{doc_column}:{col}',
                    [row[i] for row in doc_summary_column],
                )

            # Remove the (document, summary) column
            dataset.remove_column(f'{type(aligner).__name__}:{doc_column}:{to_columns}')

        if reference_column is not None and len(summary_columns) > 1:
            # Instead of having one column for (reference, summary) comparisons, split
            # off into (|summary_columns|) total columns, one for each comparison

            # Retrieve the (reference, summary) column
            reference_summary_column = dataset[f'{type(aligner).__name__}:{reference_column}:{summary_columns}']

            for i, col in enumerate(summary_columns):
                # Add as a new column
                dataset.add_column(
                    f'{type(aligner).__name__}:{reference_column}:{col}',
                    [row[i] for row in reference_summary_column],
                )

            # Remove the (reference, summary) column
            dataset.remove_column(f'{type(aligner).__name__}:{reference_column}:{summary_columns}')

    return dataset


def load_nlp():
    try:
        return load('en_core_web_lg')
    except OSError:
        raise OSError("'en_core_web_lg model' is required unless loading from cached file."
                      "To install: 'python -m spacy download en_core_web_lg'")


def run_workflow(
    jsonl_path: str,
    doc_column: str = None,
    reference_column: str = None,
    summary_columns: List[str] = None,
    bert_aligner_threshold: float = 0.5,
    bert_aligner_top_k: int = 3,
    embedding_aligner_threshold: float = 0.5,
    embedding_aligner_top_k: int = 3,
    processed_dataset_path: str = None,
    n_samples: int = None
):
    if not jsonl_path:
        raise ValueError("'jsonl_path' is required")

    if not processed_dataset_path:
        raise ValueError("Please specify a path to save the dataset.")

    # Load the dataset
    dataset = DataPanel.from_jsonl(jsonl_path)

    if doc_column is None:
        # Assume `doc_column` is called "document"
        doc_column = 'document'
        assert doc_column in dataset.columns, \
            f"`doc_column={doc_column}` is not a column in datapanel."
        print("Assuming `doc_column` is called 'document'.")

    if reference_column is None:
        # Assume `reference_column` is called "summary:reference"
        reference_column = 'summary:reference'
        print("Assuming `reference_column` is called 'summary:reference'.")
        if reference_column not in dataset.columns:
            print("No reference summary loaded")
            reference_column = None

    if summary_columns is None or len(summary_columns) == 0:
        # Assume `summary_columns` are prefixed by "summary:"
        summary_columns = []
        for col in dataset.columns:
            if col.startswith("summary:") and col != "summary:reference":
                summary_columns.append(col)
        print(f"Reading summary columns from datapanel. Found {summary_columns}.")

    if len(summary_columns) == 0 and reference_column is None:
        raise ValueError("At least one summary is required")

    # Restrict to the first `n_samples`
    if n_samples:
        print(f"Restricting to {n_samples} samples.")
        dataset = dataset.head(n_samples)

    print("size of dataset:", len(dataset))

    # Combine the text columns into one list
    text_columns = [doc_column] + ([reference_column] if reference_column else []) + summary_columns

    # Preprocessing all the text columns
    print("Preprocessing text columns")
    dataset = dataset.update(
        lambda x: {
            f'preprocessed_{k}': x[k] if args.no_clean else clean_text(x[k])
            for k in text_columns
        }
    )

    # Run the Spacy pipeline on all preprocessed text columns
    nlp = load_nlp()

    nlp.add_pipe('sentencizer', before="parser")

    print("Running spacy processing")
    for col in text_columns:
        dataset.add_column(f'spacy:{col}', SpacyColumn.from_docs(nlp.pipe(dataset[f'preprocessed_{col}'])))

    # Run the 3 align pipelines
    bert_aligner = BertscoreAligner(
        threshold=bert_aligner_threshold,
        top_k=bert_aligner_top_k,
    )

    embedding_aligner = StaticEmbeddingAligner(
        threshold=embedding_aligner_threshold,
        top_k=embedding_aligner_top_k,
    )

    ngram_aligner = NGramAligner()

    dataset = _run_aligners(
        dataset=dataset,
        aligners=[bert_aligner, embedding_aligner, ngram_aligner],
        doc_column=f'spacy:{doc_column}',
        reference_column=f'spacy:{reference_column}' if reference_column else None,
        summary_columns=[f'spacy:{col}' for col in summary_columns],
    )

    # Save the dataset
    dataset.write(processed_dataset_path)

    return dataset


def standardize_dataset(
    dataset_name: str,
    dataset_version: str,
    dataset_split: str,
    save_jsonl_path: str,
    doc_column: str = None,
    reference_column: str = None,
    n_samples: int = None

):
    """Load a dataset from Huggingface and dump it to disk."""

    if args.dataset is None or \
        args.split is None or \
        args.save_jsonl_path is None:
        raise ValueError('Missing command line argument')

    # Load the dataset from Huggingface
    dataset = get_dataset(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        dataset_split=dataset_split
    )
    if n_samples:
        dataset = dataset[:n_samples]

    if doc_column is None:
        if reference_column is not None:
            raise ValueError("You must specify `doc_column` if you specify `reference_column`")
        try:
            doc_column, reference_column = {
                'cnn_dailymail': ('article', 'highlights'),
                'xsum': ('document', 'summary')
            }[dataset_name]
        except:
            raise NotImplementedError(
                "Please specify `doc_column`."
            )

    # Rename the columns
    if doc_column != 'document':
        dataset.add_column('document', dataset[doc_column])
        dataset.remove_column(doc_column)
    dataset.add_column('summary:reference', dataset[reference_column])
    dataset.remove_column(reference_column)

    # Save the dataset back to disk
    dataset.to_jsonl(save_jsonl_path)
    return dataset


def get_dataset(
    dataset_name: str = None,
    dataset_version: str = None,
    dataset_split: str = 'test',
    dataset_jsonl: str = None,
):
    """Load a dataset."""
    assert (dataset_name is not None) != (dataset_jsonl is not None), \
        "Specify one of `dataset_name` or `dataset_jsonl`."

    # Load the dataset
    if dataset_name is not None:
        return get_hf_dataset(dataset_name, dataset_version, dataset_split)

    return DataPanel.from_jsonl(json_path=dataset_jsonl)


def get_hf_dataset(name: str, version: str = None, split: str = 'test'):
    """Get dataset from Huggingface."""
    if version:
        return DataPanel.from_huggingface(name, version, split=split)
    return DataPanel.from_huggingface(name, split=split)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['cnn_dailymail', 'xsum'],
                        help="Huggingface dataset name.")
    parser.add_argument('--version', type=str,
                        help="Huggingface dataset version.")
    parser.add_argument('--split', type=str, default='test',
                        help="Huggingface dataset split.")
    parser.add_argument('--dataset_jsonl', type=str,
                        help="Path to a jsonl file for the dataset.")
    parser.add_argument('--save_jsonl_path', type=str,
                        help="Path to save the processed jsonl dataset.")
    parser.add_argument('--doc_column', type=str,
                        help="Name of the document column in the dataset.")
    parser.add_argument('--reference_column', type=str,
                        help="Name of the reference summary column in the dataset.")
    parser.add_argument('--summary_columns', nargs='+', default=[],
                        help="Name of other summary columns in/added to the dataset.")

    parser.add_argument('--bert_aligner_threshold', type=float, default=0.1,
                        help="Minimum threshold for BERT alignment.")
    parser.add_argument('--bert_aligner_top_k', type=int, default=10,
                        help="Top-k for BERT alignment.")
    parser.add_argument('--embedding_aligner_threshold', type=float, default=0.1,
                        help="Minimum threshold for embedding alignment.")
    parser.add_argument('--embedding_aligner_top_k', type=int, default=10,
                        help="Top-k for embedding alignment.")
    parser.add_argument('--processed_dataset_path', type=str,
                        help="Path to store the final processed dataset.")
    parser.add_argument('--n_samples', type=int,
                        help="Number of dataset samples to process.")

    parser.add_argument('--workflow', action='store_true', default=False,
                        help="Whether to run the preprocessing workflow.")
    parser.add_argument('--standardize', action='store_true', default=False,
                        help="Whether to standardize the dataset and save to jsonl.")
    parser.add_argument('--no_clean', action='store_true', default=False,
                        help="Do not clean text (remove extraneous spaces, newlines).")
    args = parser.parse_args()

    if args.standardize:
        # Dump a Huggingface dataset to standardized jsonl format
        standardize_dataset(
            dataset_name=args.dataset,
            dataset_version=args.version,
            dataset_split=args.split,
            save_jsonl_path=args.save_jsonl_path,
            doc_column=args.doc_column,
            reference_column=args.reference_column,
            n_samples=args.n_samples
        )

    if args.workflow:
        # Run the processing workflow
        run_workflow(
            jsonl_path=args.dataset_jsonl,
            doc_column=args.doc_column,
            reference_column=args.reference_column,
            summary_columns=args.summary_columns,
            bert_aligner_threshold=args.bert_aligner_threshold,
            bert_aligner_top_k=args.bert_aligner_top_k,
            embedding_aligner_threshold=args.embedding_aligner_threshold,
            embedding_aligner_top_k=args.embedding_aligner_top_k,
            processed_dataset_path=args.processed_dataset_path,
            n_samples=args.n_samples
        )
