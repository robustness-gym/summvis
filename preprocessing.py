import logging
import os
from argparse import ArgumentParser
from types import SimpleNamespace
from typing import List
from spacy import load

from meerkat import DataPanel, SpacyColumn
from meerkat.logging.utils import set_logging_level

from align import BertscoreAligner, NGramAligner, StaticEmbeddingAligner
from utils import clean_text

set_logging_level('critical')
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)


def _run_aligners(
        dataset: DataPanel,
        aligners: List[BertscoreAligner, NGramAligner, StaticEmbeddingAligner],
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
        dataset.update(
            lambda x: {
                f'{aligner.__class__.__name__}:{doc_column}:{to_columns}': aligner.align(
                    x[f'spacy:{doc_column}'], 
                    [x[f'spacy:{col}'] for col in to_columns],
                ),
            },
        )

        if reference_column is not None and len(summary_columns):
            # Run the aligner on (reference, summary) pairs
            dataset.update(
                lambda x: {
                    f'{aligner.__class__.__name__}:{reference_column}:{summary_columns}': aligner.align(
                        x[f'spacy:{reference_column}'], 
                        [x[f'spacy:{col}'] for col in summary_columns],
                    ),
                },
            )

        if len(to_columns) > 1:
            # Instead of having one column for (document, summary) comparisons, split
            # off into (1 + |summary_columns|) total columns, one for each comparison

            # Retrieve the (document, summary) column
            doc_summary_column = dataset[f'{aligner.__class__.__name__}:{doc_column}:{to_columns}']

            for i, col in enumerate(to_columns):
                # Add as a new column after encoding with the aligner's `encode` method
                dataset.add_column(
                    column=f'{aligner.__class__.__name__}:{doc_column}:{col}',
                    values=[[row[i]] for row in doc_summary_column],
                )

            # Remove the (document, summary) column
            dataset.remove_column(f'{aligner.__class__.__name__}:{doc_column}:{to_columns}')

        if reference_column is not None and len(summary_columns) > 1:
            # Instead of having one column for (reference, summary) comparisons, split
            # off into (|summary_columns|) total columns, one for each comparison

            # Retrieve the (reference, summary) column
            reference_summary_column = dataset[f'{aligner.__class__.__name__}:{reference_column}:{summary_columns}']

            for i, col in enumerate(summary_columns):
                # Add as a new column
                dataset.add_column(
                    column=f'{aligner.__class__.__name__}:{reference_column}:{col}',
                    values=[[row[i]] for row in reference_summary_column],
                )

            # Remove the (reference, summary) column
            dataset.remove_column(f'{aligner.__class__.__name__}:{reference_column}:{summary_columns}')

    return dataset


def deanonymize_dataset(
        dataset_path: str,
        standardized_dataset: DataPanel,
        processed_dataset_path: str = None,
        n_samples: int = None,
):
    """Take an anonymized dataset and add back the original dataset columns."""
    assert processed_dataset_path is not None, \
        "Please specify a path to save the dataset."

    # Load the dataset
    dataset = DataPanel.read(dataset_path)

    if n_samples:
        dataset.head(n_samples)
        standardized_dataset.head(n_samples)

    text_columns = []

    # Add columns from the standardized dataset
    dataset.add_column('document', standardized_dataset['document'])
    text_columns.append('document')

    if 'summary:reference' in standardized_dataset.column_names:
        dataset.add_column('summary:reference', standardized_dataset['summary:reference'])
        text_columns.append('summary:reference')

    # Preprocessing all the text columns
    dataset = dataset.update(
        lambda x:
            {
                f'preprocessed_{k}': x[k] if args.no_clean else clean_text(x[k])
                for k in text_columns
            }
    )

    # Run the Spacy pipeline on all preprocessed text columns
    try:
        nlp = load('en_core_web_lg')
    except OSError:
        nlp = load('en_core_web_sm')

    nlp.add_pipe('sentencizer', before="parser")

    for col in text_columns:
        dataset.add_column(f'spacy:{col}', SpacyColumn.from_docs(nlp.pipe(dataset['preprocessed_{col}'])))

    # Directly save to disk
    dataset.write(processed_dataset_path)

    return dataset


def run_workflow(
        jsonl_path: str = None,
        dataset: DataPanel = None,
        doc_column: str = None,
        reference_column: str = None,
        summary_columns: List[str] = None,
        bert_aligner_threshold: float = 0.5,
        bert_aligner_top_k: int = 3,
        embedding_aligner_threshold: float = 0.5,
        embedding_aligner_top_k: int = 3,
        processed_dataset_path: str = None,
        n_samples: int = None,
        anonymize: bool = False,
):
    assert (jsonl_path is None) != (dataset is None), \
        "One of `jsonl_path` and `dataset` must be specified."
    assert processed_dataset_path is not None, \
        "Please specify a path to save the dataset."

    # Load the dataset
    if jsonl_path is not None:
        dataset = DataPanel.from_jsonl(jsonl_path)

    if doc_column is None:
        # Assume `doc_column` is called "document"
        doc_column = 'document'
        assert doc_column in dataset.column_names, \
            f"`doc_column={doc_column}` is not a column in datapanel."
        print("Assuming `doc_column` is called 'document'.")

    if reference_column is None:
        # Assume `reference_column` is called "summary:reference"
        reference_column = 'summary:reference'
        print("Assuming `reference_column` is called 'summary:reference'.")
        if reference_column not in dataset.column_names:
            print("No reference summary loaded")
            reference_column = None

    if summary_columns is None or len(summary_columns) == 0:
        # Assume `summary_columns` are prefixed by "summary:"
        summary_columns = []
        for col in dataset.column_names:
            if col.startswith("summary:") and col != "summary:reference":
                summary_columns.append(col)
        print(f"Reading summary columns from datapanel. Found {summary_columns}.")

    if len(summary_columns) == 0 and reference_column is None:
        raise ValueError("At least one summary is required")

    # Restrict to the first `n_samples`
    if n_samples:
        dataset.head(n_samples)

    # Combine the text columns into one list
    text_columns = [doc_column] + ([reference_column] if reference_column else []) + summary_columns

    # Preprocessing all the text columns
    dataset = dataset.update(
        lambda x: {
            f'preprocessed_{k}': x[k] if args.no_clean else clean_text(x[k])
            for k in text_columns
        }
    )

    # Run the Spacy pipeline on all preprocessed text columns
    try:
        nlp = load('en_core_web_lg')
    except OSError:
        nlp = None

    if nlp is None:
        raise OSError('Missing spaCy model "en_core_web_lg". Please run "python -m spacy download en_core_web_lg"')

    nlp.add_pipe('sentencizer', before="parser")

    for col in text_columns:
        dataset.add_column(f'spacy:{col}', SpacyColumn.from_docs(nlp.pipe(dataset['preprocessed_{col}'])))


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
        doc_column=f'preprocessed_{doc_column}',
        reference_column=f'preprocessed_{reference_column}' if reference_column else None,
        summary_columns=[f'preprocessed_{col}' for col in summary_columns],
    )

    # Save the dataset
    if anonymize:
        # Remove certain columns to anonymize and save to disk
        for col in [doc_column, reference_column]:
            if col is not None:
                dataset.remove_column(col)
                dataset.remove_column(f'preprocessed_{col}')
                dataset.remove_column(f'spacy:{col}')
        dataset.write(f'{processed_dataset_path}.anonymized')
    else:
        # Directly save to disk
        dataset.write(processed_dataset_path)

    return dataset


def parse_prediction_jsonl_name(prediction_jsonl: str):
    """Parse the name of the prediction_jsonl to extract useful information."""
    # Analyze the name of the prediction_jsonl
    filename = prediction_jsonl.split("/")[-1]

    # Check that the filename ends with `.results.anonymized`
    if filename.endswith(".results.anonymized"):
        # Fmt: <model>-<training dataset>.<eval dataset>.<eval split>.results.anonymized

        # Split using a period
        model_train_dataset, eval_dataset, eval_split = filename.split(".")[:-2]
        model, train_dataset = model_train_dataset.split("-")

        return SimpleNamespace(
            model_train_dataset=model_train_dataset,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_split=eval_split,
        )

    raise NotImplementedError(
        "Prediction files must be named "
        "<model>-<training dataset>.<eval dataset>.<eval split>.results.anonymized. "
        f"Please rename the prediction file {filename} and run again."
    )


def join_predictions(
        dataset_jsonl: str = None,
        prediction_jsonls: str = None,
        save_jsonl_path: str = None,
):
    """Join predictions with a dataset."""
    assert prediction_jsonls is not None, "Must have prediction jsonl files."

    print(
        "> Warning: please inspect the prediction .jsonl file to make sure that "
        "predictions are aligned with the examples in the dataset. "
        "Use `get_dataset` to inspect the dataset."
    )

    # Load the dataset
    dataset = get_dataset(dataset_jsonl=dataset_jsonl)

    # Parse names of all prediction files to get metadata
    metadata = [
        parse_prediction_jsonl_name(prediction_jsonl)
        for prediction_jsonl in prediction_jsonls
    ]

    # Load the predictions
    predictions = [
        DataPanel.from_jsonl(json_path=prediction_jsonl)
        for prediction_jsonl in prediction_jsonls
    ]

    # Predictions for a model
    for i, prediction_data in enumerate(predictions):
        # Get metadata for i_th prediction file
        metadata_i = metadata[i]

        # Construct a prefix for columns added to the dataset for this prediction file
        prefix = metadata_i.model_train_dataset

        # Add the predictions column to the dataset
        for col in prediction_data.column_names:
            # Don't add the indexing information since the dataset has it already
            if col not in {'index', 'ix', 'id'}:
                # `add_column` will automatically ensure that column lengths match
                if col == 'decoded':  # rename decoded to summary
                    dataset.add_column(f'summary:{prefix}', prediction_data[col])
                else:
                    dataset.add_column(f'{prefix}:{col}', prediction_data[col])

    # Save the dataset back to disk
    if save_jsonl_path:
        dataset.to_jsonl(save_jsonl_path)
    else:
        print("Dataset with predictions was not saved since `save_jsonl_path` "
              "was not specified.")

    return dataset


def standardize_dataset(
        dataset_name: str = None,
        dataset_version: str = None,
        dataset_split: str = 'test',
        dataset_jsonl: str = None,
        doc_column: str = None,
        reference_column: str = None,
        save_jsonl_path: str = None,
        no_save: bool = False,
):
    """Load a dataset from Huggingface and dump it to disk."""
    # Load the dataset from Huggingface
    dataset = get_dataset(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        dataset_split=dataset_split,
        dataset_jsonl=dataset_jsonl,
    )

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
    if save_jsonl_path:
        dataset.to_jsonl(save_jsonl_path)

    elif (save_jsonl_path is None) and not no_save:
        # Auto-create a path to save the standardized dataset
        os.makedirs('preprocessing', exist_ok=True)
        if not dataset_jsonl:
            dataset.to_jsonl(
                f'preprocessing/'
                f'standardized_{dataset_name}_{dataset_version}_{dataset_split}.jsonl'
            )
        else:
            dataset.to_jsonl(
                f'preprocessing/'
                f'standardized_{dataset_jsonl.split("/")[-1]}'
            )

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
    parser.add_argument('--dataset_mk', type=str,
                        help="Path to a dataset stored in the Meerkat format. "
                             "All processed datasets are stored in this format.")
    parser.add_argument('--prediction_jsonls', nargs='+', default=[],
                        help="Path to one or more jsonl files for the predictions.")
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
    parser.add_argument('--join_predictions', action='store_true', default=False,
                        help="Whether to add predictions to the dataset and save to "
                             "jsonl.")
    parser.add_argument('--try_it', action='store_true', default=False,
                        help="`Try it` mode is faster and runs processing on 10 "
                             "examples.")
    parser.add_argument('--deanonymize', action='store_true', default=False,
                        help="Deanonymize the dataset provided by summvis.")
    parser.add_argument('--anonymize', action='store_true', default=False,
                        help="Anonymize by removing document and reference summary "
                             "columns of the original dataset.")
    parser.add_argument('--no_clean', action='store_true', default=False,
                        help="Do not clean text (remove extraneous spaces, newlines).")
    args = parser.parse_args()

    if args.standardize:
        # Dump a dataset to jsonl on disk after standardizing it
        standardize_dataset(
            dataset_name=args.dataset,
            dataset_version=args.version,
            dataset_split=args.split,
            dataset_jsonl=args.dataset_jsonl,
            doc_column=args.doc_column,
            reference_column=args.reference_column,
            save_jsonl_path=args.save_jsonl_path,
        )

    if args.join_predictions:
        # Join the predictions with the dataset
        dataset = join_predictions(
            dataset_jsonl=args.dataset_jsonl,
            prediction_jsonls=args.prediction_jsonls,
            save_jsonl_path=args.save_jsonl_path,
        )

    if args.workflow:
        # Run the processing workflow
        dataset = None
        # Check if `args.dataset_mk` was passed in
        if args.dataset_mk:
            # Load the dataset directly
            dataset = DataPanel.read(args.dataset_mk)

        run_workflow(
            jsonl_path=args.dataset_jsonl,
            dataset=dataset,
            doc_column=args.doc_column,
            reference_column=args.reference_column,
            summary_columns=args.summary_columns,
            bert_aligner_threshold=args.bert_aligner_threshold,
            bert_aligner_top_k=args.bert_aligner_top_k,
            embedding_aligner_threshold=args.embedding_aligner_threshold,
            embedding_aligner_top_k=args.embedding_aligner_top_k,
            processed_dataset_path=args.processed_dataset_path,
            n_samples=args.n_samples if not args.try_it else 10,
            anonymize=args.anonymize,
        )

    if args.deanonymize:
        # Deanonymize an anonymized dataset
        # Check if `args.dataset_mk` was passed in
        assert args.dataset_mk is not None, \
            "Must specify `dataset_mk` path to be deanonymized."
        assert args.dataset_mk.endswith('anonymized'), \
            "`dataset_mk` must end in 'anonymized'."
        assert (args.dataset is None) != (args.dataset_jsonl is None), \
            "`dataset_mk` points to an anonymized dataset that will be " \
            "deanonymized. Please pass in relevant arguments: either " \
            "`dataset`, `version` and `split` OR `dataset_jsonl`."

        # Load the standardized dataset
        standardized_dataset = standardize_dataset(
            dataset_name=args.dataset,
            dataset_version=args.version,
            dataset_split=args.split,
            dataset_jsonl=args.dataset_jsonl,
            doc_column=args.doc_column,
            reference_column=args.reference_column,
            no_save=True,
        )
        # Use it to deanonymize
        dataset = deanonymize_dataset(
            dataset_path=args.dataset_mk,
            standardized_dataset=standardized_dataset,
            processed_dataset_path=args.processed_dataset_path,
            n_samples=args.n_samples if not args.try_it else 10,
        )
