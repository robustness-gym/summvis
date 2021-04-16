# SummVis
SummVis is an interactive visualization tool for analyzing abstractive summarization model outputs and datasets.

<p align="center">
  <img src="https://raw.githubusercontent.com/robustness-gym/summvis/master/images/main-vis.jpg" width="1000" alt="Figure"/>
</p>


## Installation
Please use `python>=3.8` since some dependencies require that for installation.
```zsh
git clone https://github.com/robustness-gym/summvis.git
cd summvis
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## Quickstart
Follow the steps below to start using SummVis immediately.

### 1. Download and extract data
Download our pre-cached dataset that contains predictions for state-of-the-art models such as PEGASUS and BART on 
1000 examples taken from the CNN / Daily Mail validation set.
```shell
mkdir data
mkdir preprocessing
curl https://storage.googleapis.com/sfr-summvis-data-research/cnn_dailymail_1000.validation.anonymized.zip --output preprocessing/cnn_dailymail_1000.validation.anonymized.zip
unzip preprocessing/cnn_dailymail_1000.validation.anonymized.zip -d preprocessing/
``` 

### 2. Deanonymize data
Next, we'll need to add the original examples from the CNN / Daily Mail dataset to deanonymize the data (this information 
is omitted for copyright reasons). The `preprocessing.py` script can be used for this with the `--deanonymize` flag.

#### Deanonymize 10 examples (`try_it` mode):
```shell
python preprocessing.py \
--deanonymize \
--dataset_rg preprocessing/cnn_dailymail_1000.validation.anonymized \
--dataset cnn_dailymail \
--version 3.0.0 \
--split validation \
--processed_dataset_path data/try:cnn_dailymail_1000.validation \
--try_it
```
This takes around 10 seconds on a MacBook Pro. 

### 3. Run SummVis
Finally, we're ready to run the Streamlit app. Once the app loads, make sure it's pointing to the right `File` at the top
of the interface.
```shell
streamlit run summvis.py
```

## General instructions for running with pre-loaded datasets

### 1. Download one of the pre-loaded datasets:

##### CNN / Daily Mail (1000 examples from validation set): https://storage.googleapis.com/sfr-summvis-data-research/cnn_dailymail_1000.validation.anonymized.zip
##### CNN / Daily Mail (full validation set): https://storage.googleapis.com/sfr-summvis-data-research/cnn_dailymail.validation.anonymized.zip
##### XSum (1000 examples from validation set): https://storage.googleapis.com/sfr-summvis-data-research/xsum_1000.validation.anonymized.zip
##### XSum (full validation set): https://storage.googleapis.com/sfr-summvis-data-research/xsum.validation.anonymized.zip

We recommend that you choose the smallest dataset that fits your need in order to minimize download / preprocessing time.

#### Example: Download and unzip CNN / Daily Mail
```shell
mkdir data
mkdir preprocessing
curl https://storage.googleapis.com/sfr-summvis-data-research/cnn_dailymail_1000.validation.anonymized.zip --output preprocessing/cnn_dailymail_1000.validation.anonymized.zip
unzip preprocessing/cnn_dailymail_1000.validation.anonymized.zip -d preprocessing/
``` 

### 2. Deanonymize *n* examples:

Set the `--n_samples` argument and name the `--processed_dataset_path` output file accordingly.

#### Example: Deanonymize 100 examples from CNN / Daily Mail:
```shell
python preprocessing.py \
--deanonymize \
--dataset_rg preprocessing/cnn_dailymail_1000.validation.anonymized \
--dataset cnn_dailymail \
--version 3.0.0 \
--split validation \
--processed_dataset_path data/100:cnn_dailymail_1000.validation \
--n_samples 100
```

#### Example: Deanonymize all pre-loaded examples from CNN / Daily Mail (1000 examples dataset):
```shell
python preprocessing.py \
--deanonymize \
--dataset_rg preprocessing/cnn_dailymail_1000.validation.anonymized \
--dataset cnn_dailymail \
--version 3.0.0 \
--split validation \
--processed_dataset_path data/full:cnn_dailymail_1000.validation \
--n_samples 1000
```

#### Example: Deanonymize all pre-loaded examples from CNN / Daily Mail (full dataset):
```shell
python preprocessing.py \
--deanonymize \
--dataset_rg preprocessing/cnn_dailymail.validation.anonymized \
--dataset cnn_dailymail \
--version 3.0.0 \
--split validation \
--processed_dataset_path data/full:cnn_dailymail.validation
```

#### Example: Deanonymize all pre-loaded examples from XSum (1000 examples dataset):
```shell
python preprocessing.py \
--deanonymize \
--dataset_rg preprocessing/xsum_1000.validation.anonymized \
--dataset xsum \
--split validation \
--processed_dataset_path data/full:xsum_1000.validation \
--n_samples 1000
```

### 3. Run SummVis
Once the app loads, make sure it's pointing to the right `File` at the top
of the interface.
```shell
streamlit run summvis.py
```

Alternately, if you need to point SummVis to a folder where your data is stored.
```shell
streamlit run summvis.py -- --path your/path/to/data
```
Note that the additional `--` is not a mistake, and is required to pass command-line arguments in streamlit.


## Get your data into SummVis: end-to-end preprocessing
You can also perform preprocessing end-to-end to load any summarization dataset or model predictions into SummVis. 
Instructions for this are provided below. 

### 1. Standardize and save dataset to disk.
Loads in a dataset from HF, or any dataset that you have and stores it in a 
standardized format with columns for `document` and `summary:reference`.  

#### Example: Save CNN / Daily Mail validation split to disk as a jsonl file.
```shell
python preprocessing.py \
--standardize \
--dataset cnn_dailymail \
--version 3.0.0 \
--split validation \
--save_jsonl_path preprocessing/cnn_dailymail.validation.jsonl
```

#### Example: Load custom `my_dataset.jsonl`, standardize, and save.
```shell
python preprocessing.py \
--standardize \
--dataset_jsonl path/to/my_dataset.jsonl \
--doc_column name_of_document_column \
--reference_column name_of_reference_summary_column \
--save_jsonl_path preprocessing/my_dataset.jsonl
```

### 2. Add predictions to the saved dataset.
Takes a saved dataset that has already been standardized and adds predictions to it 
from prediction jsonl files. Cached predictions for several models available here:
 https://storage.googleapis.com/sfr-summvis-data-research/predictions.zip
 
You may also generate your own predictions using this [this script](generation.py).

#### Example: Add 6 prediction files for PEGASUS and BART to the dataset.
```shell
python preprocessing.py \
--join_predictions \
--dataset_jsonl preprocessing/cnn_dailymail.validation.jsonl \
--prediction_jsonls \
predictions/bart-cnndm.cnndm.validation.results.anonymized \
predictions/bart-xsum.cnndm.validation.results.anonymized \
predictions/pegasus-cnndm.cnndm.validation.results.anonymized \
predictions/pegasus-multinews.cnndm.validation.results.anonymized \
predictions/pegasus-newsroom.cnndm.validation.results.anonymized \
predictions/pegasus-xsum.cnndm.validation.results.anonymized \
--save_jsonl_path preprocessing/cnn_dailymail.validation.jsonl
```

### 3. Run the preprocessing workflow and save the dataset.
Takes a saved dataset that has been standardized, and predictions already added. 
Applies all the preprocessing steps to it (running `spaCy`, lexical and semantic aligners), 
and stores the processed dataset back to disk.

#### Example: Autorun with default settings on a few examples to try it.
```shell
python preprocessing.py \
--workflow \
--dataset_jsonl preprocessing/cnn_dailymail.validation.jsonl \
--processed_dataset_path data/cnn_dailymail.validation \
--try_it
```

#### Example: Autorun with default settings on all examples.
```shell
python preprocessing.py \
--workflow \
--dataset_jsonl preprocessing/cnn_dailymail.validation.jsonl \
--processed_dataset_path data/cnn_dailymail
```

## Citation

When referencing this repository, please cite [this paper](https://arxiv.org/abs/2104.07605):

```
@misc{vig2021summvis,
      title={SummVis: Interactive Visual Analysis of Models, Data, and Evaluation for Text Summarization}, 
      author={Jesse Vig and Wojciech Kryscinski and Karan Goel and Nazneen Fatema Rajani},
      year={2021},
      eprint={2104.07605},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2104.07605}
}
```

## Acknowledgements

We thank [Michael Correll](http://correll.io) for his valuable feedback.


