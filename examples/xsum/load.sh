#!/bin/bash

# Make sure to first run: python -m spacy download en_core_web_lg

# Dump one example from XSum validation split to .jsonl format. This may take several minutes if loading for first time.
python ../../preprocessing.py \
--standardize \
--dataset xsum \
--split validation \
--save_jsonl_path xsum.jsonl \
--n_samples 1 &&

# Generate predictions from 4 models
python ../../generation.py --model pegasus-cnndm --data_path xsum.jsonl &&
python ../../generation.py --model pegasus-xsum --data_path xsum.jsonl &&
python ../../generation.py --model bart-cnndm --data_path xsum.jsonl &&
python ../../generation.py --model bart-xsum --data_path xsum.jsonl &&

# Join predictions with original dataset
python ../../join.py \
  --data_path xsum.jsonl \
  --generation_paths \
      pegasus-xsum.xsum.predictions \
      pegasus-xsum.xsum.predictions \
      bart-xsum.xsum.predictions \
      bart-xsum.xsum.predictions \
  --output_path xsum-decoded.jsonl

# Cache results
python ../../preprocessing.py \
--workflow \
--dataset_jsonl xsum-decoded.jsonl \
--processed_dataset_path xsum.cache
