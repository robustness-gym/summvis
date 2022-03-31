#!/bin/bash

# Generate predictions from 4 models for wikinews.jsonl file
python ../../generation.py --model pegasus-cnndm --data_path wikinews.jsonl &&
python ../../generation.py --model pegasus-xsum --data_path wikinews.jsonl &&
python ../../generation.py --model bart-cnndm --data_path wikinews.jsonl &&
python ../../generation.py --model bart-xsum --data_path wikinews.jsonl &&

# Join predictions with original dataset
python ../../join.py \
  --data_path wikinews.jsonl \
  --generation_paths \
      pegasus-cnndm.wikinews.predictions \
      pegasus-xsum.wikinews.predictions \
      bart-cnndm.wikinews.predictions \
      bart-xsum.wikinews.predictions \
  --output_path wikinews-decoded.jsonl &&

# Cache results
python ../../preprocessing.py \
--workflow \
--dataset_jsonl wikinews-decoded.jsonl \
--processed_dataset_path wikinews.cache
