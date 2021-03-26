## Files
* cnn_dailymail.validation.augmented (Entire CNN/DM validation set)
* cnn_dailymail.validation.extractability.0_1 (Examples with extractive confidence of 0-1%)
* cnn_dailymail.validation.extractability.0_10 (Examples with extractive confidence of 0-10%)
* cnn_dailymail.validation.extractability.90_100 (Examples with extractive confidence of 90-100%)
* xsum.validation.augmented (Entire xsum validation set)

## Installation
Please use python>=3.8 since some dependencies require that for installation.
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

## Execution
```
streamlit run summvis.py
```

## Direct Loading

### Deanonymize a provided dataset and save.
Takes in a dataset provided by us, and deanonymizes it by adding in dataset information.
This is a necessary step before using it with the tool. 

#### Example: Deanonymize the entire provided CNN-Dailymail data.
```bash 
python preprocessing.py \
--deanonymize \
--dataset_rg preprocessing/cnn_dailymail_v3.anonymized \
--dataset cnn_dailymail \
--version 3.0.0 \
--split validation \
--processed_dataset_path preprocessing/cnn_dailymail_v3 \
```

#### Example: Deanonymize a sample of the provided CNN-Dailymail data.
```bash
python preprocessing.py \
--deanonymize \
--dataset_rg preprocessing/cnn_dailymail_v3.anonymized \
--dataset cnn_dailymail \
--version 3.0.0 \
--split validation \
--processed_dataset_path preprocessing/cnn_dailymail_v3 \
--try_it \
```

#### Example: Deanonymize a few examples of the provided CNN-Dailymail data.
```bash 
python preprocessing.py \
--deanonymize \
--dataset_rg preprocessing/cnn_dailymail_v3.anonymized \
--dataset cnn_dailymail \
--version 3.0.0 \
--split validation \
--processed_dataset_path preprocessing/cnn_dailymail_v3 \
--n_samples 2
```

## End-to-end Preprocessing

### 1. Standardize and save dataset to disk.
Loads in a dataset from HF, or any dataset that you have and stores it in a 
standardized format with columns for `document` and `summary:reference`.  

#### Example: Save CNN-Dailymail validation split to disk as a jsonl file.
```bash 
python preprocessing.py \
--standardize \
--dataset cnn_dailymail \
--version 3.0.0 \
--split validation \
--save_jsonl_path preprocessing/cnn_dailymail_v3_validation.jsonl \
```

#### Example: Save custom `my_dataset.jsonl`.
```bash 
python preprocessing.py \
--standardize \
--dataset_jsonl path/to/my_dataset.jsonl \
--doc_column name_of_document_column \
--reference_column name_of_reference_summary_column \
--save_jsonl_path preprocessing/my_dataset.jsonl \
```

### 2. Add predictions to the saved dataset.
Takes a saved dataset that has already been standardized and adds predictions to it 
from prediction jsonl files. 

#### Example: Add 6 prediction files for Pegasus and BART to the dataset.
```bash
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
```bash 
python preprocessing.py \
--workflow \
--dataset_jsonl preprocessing/cnn_dailymail_v3_validation.jsonl \
--processed_dataset_path preprocessing/cnn_dailymail_v3 \
--try_it
```

#### Example: Autorun with default settings on all examples.
```bash 
python preprocessing.py \
--workflow \
--dataset_jsonl preprocessing/cnn_dailymail_v3_validation.jsonl \
--processed_dataset_path preprocessing/cnn_dailymail_v3
```


