import os, sys

sys.path.append(os.path.abspath('..'))

from argparse import ArgumentParser
from pathlib import Path

import conllu
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

from bert_repro.probe.data import ProbeDataset
from bert_repro.probe.trainer import ProbeTrainer
from bert_repro.probe.model import TwoWordProbe


def get_args():
    """Returns arguments
    """
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        '--data_fp',
        default='./data/ud-treebanks-v2.7/UD_English-EWT/en_ewt-ud-train.conllu'
    ),
    arg_parser.add_argument('--output_dir', type=Path, required=True)
    arg_parser.add_argument('--bert_model', default='bert-base-uncased')
    arg_parser.add_argument('--embedding_size', default=768, type=int)
    arg_parser.add_argument('--max_probe_rank', default=128, type=int)
    arg_parser.add_argument('--n_epochs', default=20, type=int)

    return arg_parser.parse_args()


def main(args):
    train_data = ProbeDataset(args.data_fp)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = TFBertModel.from_pretrained(args.bert_model)

    probe_model = TwoWordProbe(args.embedding_size, args.max_probe_rank)

    trainer = ProbeTrainer(args.n_epochs)

    tf_data = tf.data.Dataset.from_generator(
        lambda: train_data.build_pairs_distance_generator(bert, tokenizer),
        output_types=(tf.float32, tf.float32))

    trained_model = trainer.train(probe_model, tf_data)

    args.output_dir.mkdir(exist_ok=True)
    trained_model.save(str(args.output_dir))


if __name__ == '__main__':
    args = get_args()
    main(args)
