"""
Script for joining dataset of documents/reference summaries with generated summaries (likely from generate.py).

Usage with custom datasets in JSONL format:
python join.py --data_path <path to data in jsonl format> --generation_paths <paths to generated predictions>  --output_path <path to output file>

Optionally specify --model_names to override default model names.

"""
# !/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

BATCH_SIZE = 8


class JSONDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super(JSONDataset, self).__init__()

        with open(data_path) as fd:
            self.data = [json.loads(line) for line in fd]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--generation_paths', type=str, nargs="+", required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model_names', type=str, nargs="+")
    args = parser.parse_args()

    if args.model_names and len(args.generation_paths) != len(args.model_names):
        raise ValueError('Length of args.generation_paths must equal length of args.model_names')

    if args.model_names:
        model_names = args.model_names
    else:
        model_names = [Path(p).name.split(".")[0] for p in args.generation_paths]

    args.dataset = os.path.splitext(os.path.basename(args.data_path))[0]
    args.split = 'user'

    # Load data

    dataset = JSONDataset(args.data_path)

    # Join files and write out single jsonl dataset

    generation_files = [open(fname) for fname in args.generation_paths]

    with open(args.output_path, 'w') as outp:
        for row in tqdm(zip(dataset, *generation_files)):
            # Process each original data record in parallel with generation(s) of the model(s)
            result = {}
            data = row[0]
            generations = row[1:]
            result['summary:reference'] = data['summary:reference']
            result['document'] = data['document']
            for model_name, gen in zip(model_names, generations):
                result[f'summary:{model_name}'] = gen
            outp.write(
                json.dumps(result) + '\n'
            )

    for file in generation_files:
        file.close()
