"""
Script for decoding summarization models available through Huggingface Transformers.

Usage with Huggingface Datasets:
python generation.py --model <model name> --data_path <path to data in jsonl format>

Usage with custom datasets in JSONL format:
python generation.py --model <model name> --dataset <dataset name> --split <data split>
"""
#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os

import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

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


def preprocess_data(raw_data, dataset):
    """
    Unify format of Huggingface Datastes

    :param raw_data: loaded data
    :param dataset: name of dataset
    """
    if dataset == 'xsum':
        raw_data['article'] = raw_data['document']
        raw_data['target'] = raw_data['summary']
        del raw_data['document']
        del raw_data['summary']
    elif dataset == 'cnndm':
        raw_data['target'] = raw_data['highlights']
        del raw_data['highlights']
    elif dataset == 'gigaword':
        raw_data['article'] = raw_data['document']
        raw_data['target'] = raw_data['summary']
        del raw_data['document']
        del raw_data['summary']

    return raw_data


def postprocess_data(raw_data, decoded):
    """
    Remove generation artifacts and postprocess outputs

    :param raw_data: loaded data
    :param decoded: model outputs
    """
    raw_data['target'] = [x.replace('\n', ' ') for x in raw_data['target']]
    raw_data['decoded'] = [x.replace('<n>', ' ') for x in decoded]

    return [dict(zip(raw_data, t)) for t in zip(*raw_data.values())]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=str, required=True, choices=['bart-xsum', 'bart-cnndm', 'pegasus-xsum', 'pegasus-cnndm', 'pegasus-newsroom', 'pegasus-multinews'])
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--dataset', type=str, choices=['xsum', 'cnndm', 'gigaword'])
    parser.add_argument('--split', type=str, choices=['train', 'validation', 'test'])
    args = parser.parse_args()

    if args.dataset and not args.split:
        raise RuntimeError('If `dataset` flag is specified `split` must also be provided.')

    if args.data_path:
        args.dataset = os.path.splitext(os.path.basename(args.data_path))[0]
        args.split = 'user'

    # Load models & data
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINTS[args.model]).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINTS[args.model])

    if not args.data_path:
        if args.dataset == 'cnndm':
            dataset = load_dataset('cnn_dailymail', '3.0.0', split=args.split)
        elif args.dataset =='xsum':
            dataset = load_dataset('xsum', split=args.split)
        elif args.dataset =='gigaword':
            dataset = load_dataset('gigaword', split=args.split)
    else:
        dataset = JSONDataset(args.data_path)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    # Run validation
    filename = '%s.%s.%s.results' % (args.model.replace("/", "-"), args.dataset, args.split)
    fd_out = open(filename, 'w')

    results = []
    model.eval()
    with torch.no_grad():
        for raw_data in tqdm(dataloader):
            raw_data = preprocess_data(raw_data, args.dataset)
            batch = tokenizer(raw_data["article"], return_tensors="pt", truncation=True, padding="longest").to(DEVICE)
            summaries = model.generate(input_ids=batch.input_ids, attention_mask=batch.attention_mask)

            decoded = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            result = postprocess_data(raw_data, decoded)
            results.extend(result)

            for example in result:
                fd_out.write(json.dumps(example) + '\n')
