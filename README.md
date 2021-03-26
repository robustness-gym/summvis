# SummVis
SummVis is an interactive visualization and analysis tool for inspecting summarization datasets and model generated 
summaries.

## Installation
Please use `python>=3.8` since some dependencies require that for installation.
```shell
git clone https://github.com/robustness-gym/summvis.git
cd summvis
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

## Quickstart
Follow the steps below to start using SummVis immediately.

#### Download and extract data
Download our pre-cached dataset that contains predictions for state-of-the-art models such as Pegasus and BART on 
1000 examples taken from the CNN-Dailymail validation set.
```shell
mkdir data
mkdir preprocessing
curl https://storage.googleapis.com/sfr-summvis-data-research/cnn_dailymail_1000.anonymized.zip --output preprocessing/cnn_dailymail_1000.anonymized.zip
unzip preprocessing/cnn_dailymail_1000.anonymized.zip -d preprocessing/
``` 

#### Deanonymize data
Next, we'll need to add the original examples from the CNN-Dailymail dataset to deanonymize the data (this information 
is omitted for copyright reasons). The `preprocessing.py` script can be used for this with the `--deanonymize` flag.

##### Option 1: Deanonymize all 1000 examples in the provided CNN-Dailymail data.
Takes around `2` minutes on a MacBook Pro.

```shell
python preprocessing.py \
--deanonymize \
--dataset_rg preprocessing/cnn_dailymail_1000.anonymized \
--dataset cnn_dailymail \
--version 3.0.0 \
--split validation \
--processed_dataset_path data/full:cnn_dailymail_1000
```

##### Option 2 (`try_it`): Deanonymize a sample of the provided CNN-Dailymail data.
Takes around `20` seconds on a MacBook Pro.

```shell
python preprocessing.py \
--deanonymize \
--dataset_rg preprocessing/cnn_dailymail_1000.anonymized \
--dataset cnn_dailymail \
--version 3.0.0 \
--split validation \
--processed_dataset_path data/try:cnn_dailymail_1000 \
--try_it
```

#### Option 3 (`n_samples`): Deanonymize a fixed number of examples (e.g. 50) in the provided CNN-Dailymail data.
```shell
python preprocessing.py \
--deanonymize \
--dataset_rg preprocessing/cnn_dailymail_1000.anonymized \
--dataset cnn_dailymail \
--version 3.0.0 \
--split validation \
--processed_dataset_path data/50:cnn_dailymail_1000 \
--n_samples 50
```

## Running SummVis
Finally, we're ready to run the Streamlit app. Once the app loads, make sure it's pointing to the right `File` at the top
of the interface.
```shell
streamlit run summvis.py
```
Alternately, if you need to point SummVis to a folder where your data is stored.
```shell
streamlit run summvis.py -- --path your/path/to/data
```
Note that the additional `--` is not a mistake, and is required to pass command-line arguments in streamlit.

## End-to-end Preprocessing
You can also perform preprocessing end-to-end to load any summarization dataset or model predictions into SummVis. 
Instructions for this are provided below. 

### 1. Standardize and save dataset to disk.
Loads in a dataset from HF, or any dataset that you have and stores it in a 
standardized format with columns for `document` and `summary:reference`.  

#### Example: Save CNN-Dailymail validation split to disk as a jsonl file.
```shell
python preprocessing.py \
--standardize \
--dataset cnn_dailymail \
--version 3.0.0 \
--split validation \
--save_jsonl_path preprocessing/cnn_dailymail_v3_validation.jsonl
```

#### Example: Load custom `my_dataset.jsonl`, standardize and save.
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
from prediction jsonl files. 

#### Example: Add 6 prediction files for Pegasus and BART to the dataset.
```shell
python preprocessing.py \
--join_predictions \
--dataset_jsonl preprocessing/cnn_dailymail_v3_validation.jsonl \
--prediction_jsonls \
anonymized/bart-cnndm.cnndm.validation.results.anonymized \
anonymized/bart-xsum.cnndm.validation.results.anonymized \
anonymized/pegasus-cnndm.cnndm.validation.results.anonymized \
anonymized/pegasus-multinews.cnndm.validation.results.anonymized \
anonymized/pegasus-newsroom.cnndm.validation.results.anonymized \
anonymized/pegasus-xsum.cnndm.validation.results.anonymized \
--save_jsonl_path preprocessing/cnn_dailymail_v3_validation.jsonl
```

### 3. Run the preprocessing workflow and save the dataset.
Takes a saved dataset that has been standardized, and predictions already added. 
Applies all the preprocessing steps to it, and stores the processed dataset back to 
disk.

#### Example: Autorun with default settings on a few examples to try it.
```shell
python preprocessing.py \
--workflow \
--dataset_jsonl preprocessing/cnn_dailymail_v3_validation.jsonl \
--processed_dataset_path data/cnn_dailymail_v3 \
--try_it
```

#### Example: Autorun with default settings on all examples.
```shell
python preprocessing.py \
--workflow \
--dataset_jsonl preprocessing/cnn_dailymail_v3_validation.jsonl \
--processed_dataset_path data/cnn_dailymail_v3
```


