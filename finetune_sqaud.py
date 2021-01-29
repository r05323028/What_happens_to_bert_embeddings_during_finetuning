from argparse import ArgumentParser
from typing import Dict

import tensorflow as tf
from transformers import (BertTokenizer, TFBertForQuestionAnswering, TFTrainer,
                          TFTrainingArguments,
                          squad_convert_examples_to_features)
import tensorflow_datasets as tfds
from transformers.data.processors.squad import SquadV1Processor

BERT_MODEL = 'bert-base-uncased'


def get_args():
    '''Return arguments
    '''
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data_dir', default='./data')
    arg_parser.add_argument("--max_seq_length", default=128, type=int)
    arg_parser.add_argument('--doc_stride', default=128, type=int)
    arg_parser.add_argument('--max_query_length', default=64, type=int)
    arg_parser.add_argument('--output_dir', default='./models/squad')
    arg_parser.add_argument('--log_dir', default='./logs')
    arg_parser.add_argument('--n_epochs', default=3, type=int)

    return arg_parser.parse_args()


@tf.function
def get_required_features(question: Dict, answer: Dict):
    '''Selecting features for Question-Answering model

    Args:
        question (Dict)
        answer (Dict)

    Return:
        question (Dict)
        answer (Dict)
    '''
    answer_ = {
        k: v
        for k, v in answer.items()
        if k in ['start_positions', 'end_positions']
    }

    return question, answer_


def main(args):
    training_args = TFTrainingArguments(
        num_train_epochs=args.n_epochs,
        output_dir=args.output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=args.log_dir,
        do_train=True,
    )

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    with training_args.strategy.scope():
        bert = TFBertForQuestionAnswering.from_pretrained(BERT_MODEL)

    tfds_examples = tfds.load("squad", data_dir=args.data_dir)
    train_examples = SquadV1Processor().get_examples_from_dataset(
        tfds_examples, evaluate=False)
    eval_examples = SquadV1Processor().get_examples_from_dataset(tfds_examples,
                                                                 evaluate=True)

    train_dataset = squad_convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=128,
        doc_stride=128,
        max_query_length=64,
        is_training=True,
        return_dataset="tf",
    )

    train_dataset = train_dataset.apply(
        tf.data.experimental.assert_cardinality(len(train_examples)))

    eval_dataset = squad_convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=128,
        doc_stride=128,
        max_query_length=64,
        is_training=False,
        return_dataset="tf",
    )

    eval_dataset = eval_dataset.apply(
        tf.data.experimental.assert_cardinality(len(eval_examples)))

    train_dataset = train_dataset.map(get_required_features)
    eval_dataset = eval_dataset.map(get_required_features)

    trainer = TFTrainer(model=bert,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset)

    if training_args.do_train:
        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    args = get_args()
    main(args)