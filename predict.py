import csv
import os
import pickle
from pathlib import Path
from bisect import bisect_left, bisect_right
import logging

import numpy as np
from transformers import BertTokenizer
from bert_deid.model.transformer import Transformer
from bert_deid.processors import DeidProcessor
from bert_deid.label import LabelCollection, LABEL_SETS, LABEL_MEMBERSHIP
from tqdm import tqdm
import torch
from lxml import etree

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default='/enc_data/deid-gs/i2b2_2014/test',
        type=str,
        required=True,
        help="The input data dir.",
    )
    parser.add_argument(
        "--model_dir",
        default='/enc_data/models/bert-i2b2-2014',
        type=str,
        help="Path to the model.",
    )
    parser.add_argument(
        "--model_type", default='bert', type=str, help="Type of model"
    )
    parser.add_argument(
        "--task",
        default='i2b2_2014',
        type=str,
        choices=LABEL_SETS,
        help=f"Type of dataset: {', '.join(LABEL_SETS)}.",
    )
    parser.add_argument(
        "--output",
        default='preds.pkl',
        type=str,
        help="Output file",
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        type=str,
        help="Output folder for CSV stand-off annotations.",
    )
    args = parser.parse_args()

    # load in the training args from the model
    if not os.path.exists(args.model_dir):
        raise ValueError(f'Model directory does not exist: {model_dir}')

    training_args = torch.load(
        os.path.join(args.model_dir, 'training_args.bin')
    )

    # args.patterns = []
    # if args.feature is not None:
    #     for f in args.feature:
    #         f = f.lower()
    #         args.patterns.append(f)

    # if 'all' in args.patterns:
    #     args.patterns = PATTERN_NAMES
    # prepare the label set - gives us mapping from ID to label
    # label_set = LabelCollection(
    #     args.task_type, transform=args.label_transform
    # )
    label_set = LabelCollection(args.task)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load in a trained model
    transformer = Transformer(
        args.model_dir
    )

    label_to_id = transformer.label_set.label_to_id
    data_path = Path(args.data_dir)

    if args.output_folder is not None:
        output_folder = Path(args.output_folder)
        if not output_folder.exists():
            output_folder.mkdir(parents=True)
        output_header = [
            'document_id', 'annotation_id', 'start', 'stop', 'entity',
            'entity_type', 'comment'
        ]
    else:
        output_folder = None

    files = os.listdir(data_path / 'test/xml-test')
    files = [f for f in files if f.endswith('.xml')]
    data = []

    preds = None
    lengths = []
    offsets = []
    labels = []
    num_subwords = 0
    num_tokens = 0

    for f in tqdm(files, total=len(files), desc='Files'):
        text = etree.parse(f"{data_path}/test/xml-test/{f}").find('TEXT').text
        ex_preds = transformer.predict(text)
        if preds is None:
            preds = ex_preds
        else:
            preds = np.append(preds, ex_preds, axis=0)
        if output_folder is not None:
            # output the data to this folder as .pred files
            with open(output_folder / f'{f[:-4]}.pred', 'w') as fp:
                csvwriter = csv.writer(fp)
                # header
                csvwriter.writerow(output_header)
                for i in range(len(ex_preds)):
                    start, length = int(ex_preds[i][2]), int(ex_preds[i][3])
                    stop = start + length
                    entity = text[start:stop]
                    # if args.model_type == 'bert_crf' or args.model_type == 'bert_extra_feature_crf':
                    #     # print(ex_preds[i, :])
                    #     # assert (len(ex_preds[i, :]) == 1)
                    #     # BertCRF gives one predicted tag id: (batch_size, max_seq_len, 1)
                    #     entity_type = transformer.label_set.id_to_label[int(
                    #         ex_preds[i, :][1]
                    #     )]
                    # else:
                    #     entity_type = transformer.label_set.id_to_label[
                    #         np.argmax(ex_preds[i, :])]
                    entity_type = ex_preds[i, 1]
                    # do not save object entity types
                    if entity_type == 'O':
                        continue
                    row = [
                        f[:-4],
                        str(i + 1), start, stop, entity, entity_type, None
                    ]
                    csvwriter.writerow(row)
        # lengths.append(ex_lengths)
        # offsets.append(ex_offsets)
        """
        # load in gold standard
        gs_fn = data_path / 'ann' / f'{f[:-4]}.gs'

        # load the gold standard labels into the label set
        label_set.from_csv(gs_fn)
        label_set.sort_labels()

        # initialize all predictions as objects
        label_tokens = ['O'] * len(ex_offsets)

        for i, g in enumerate(label_set.labels):
            # any tokens which overlap with 
            idxStart = bisect_left(ex_offsets, g.start)
            idxStop = bisect_right(ex_offsets, g.start + g.length)
            label_tokens[idxStart:idxStop] = [g.entity_type] * (idxStop - idxStart)

        label_tokens = [label_to_id[l.upper()] for l in label_tokens]
        labels.extend(label_tokens)
        """

    print(
        f'Number of subwords: {num_subwords} out of number of tokens {num_tokens}'
    )
    # with open(args.output, 'wb') as fp:
    #     pickle.dump([files, preds, labels, lengths, offsets], fp)
