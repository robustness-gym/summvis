"""
Script for decoding summarization models available through Huggingface Transformers.

Usage with Huggingface Datasets:
python generation.py --model <model name> --dataset <dataset name> --split <data split>

Usage with custom datasets in JSONL format:
python generation.py --model <model name> --data_path <path to data in jsonl format>

"""
# !/usr/bin/env python
# coding: utf-8

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BATCH_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BART_CNNDM_CHECKPOINT = 'facebook/bart-large-cnn'
BART_XSUM_CHECKPOINT = 'facebook/bart-large-xsum'
PEGASUS_CNNDM_CHECKPOINT = 'google/pegasus-cnn_dailymail'
PEGASUS_XSUM_CHECKPOINT = 'google/pegasus-xsum'
PEGASUS_NEWSROOM_CHECKPOINT = 'google/pegasus-newsroom'
PEGASUS_MULTINEWS_CHECKPOINT = 'google/pegasus-multi_news'

MODEL_CHECKPOINTS = {
    'bart-xsum': BART_XSUM_CHECKPOINT,
    'bart-cnndm': BART_CNNDM_CHECKPOINT,
    'pegasus-xsum': PEGASUS_XSUM_CHECKPOINT,
    'pegasus-cnndm': PEGASUS_CNNDM_CHECKPOINT,
    'pegasus-newsroom': PEGASUS_NEWSROOM_CHECKPOINT,
    'pegasus-multinews': PEGASUS_MULTINEWS_CHECKPOINT
}


class JSONDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super(JSONDataset, self).__init__()

        with open(data_path) as fd:
            self.data = [json.loads(line) for line in fd]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def postprocess_data(decoded):
    """
    Remove generation artifacts and postprocess outputs

    :param decoded: model outputs
    """
    return [x.replace('<n>', ' ') for x in decoded]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=str, required=True,
                        choices=['bart-xsum', 'bart-cnndm', 'pegasus-xsum', 'pegasus-cnndm', 'pegasus-newsroom',
                                 'pegasus-multinews'])
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()

    # Load models & data
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINTS[args.model]).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINTS[args.model])

    dataset = JSONDataset(args.data_path)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    # Write out dataset
    model_name = args.model.replace("/", "-")
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    filename = f'{model_name}.{dataset_name}.predictions'
    fd_out = open(filename, 'w')

    model.eval()
    with torch.no_grad():
        for raw_data in tqdm(dataloader):
            batch = tokenizer(raw_data["document"], return_tensors="pt", truncation=True, padding="longest").to(DEVICE)
            summaries = model.generate(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
            decoded = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for example in postprocess_data(decoded):
                fd_out.write(example + '\n')
