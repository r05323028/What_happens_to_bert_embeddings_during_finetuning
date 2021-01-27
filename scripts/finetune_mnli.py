from argparse import ArgumentParser

import tensorflow as tf
from transformers import (TFBertForSequenceClassification, BertTokenizer,
                          TFTrainer, TFTrainingArguments,
                          glue_convert_examples_to_features,
                          glue_tasks_num_labels, glue_output_modes, BertConfig)

import tensorflow_datasets as tfds

BERT_MODEL = 'bert-base-uncased'


def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data_dir', default='./data')
    arg_parser.add_argument("--max_seq_length", default=128, type=int)
    arg_parser.add_argument('--doc_stride', default=128, type=int)
    arg_parser.add_argument('--max_query_length', default=64, type=int)
    arg_parser.add_argument('--output_dir', default='./models/squad')
    arg_parser.add_argument('--log_dir', default='./logs')
    arg_parser.add_argument('--n_epochs', default=3, type=int)

    return arg_parser.parse_args()


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

    n_labels = glue_tasks_num_labels['mnli']

    config = BertConfig.from_pretrained(BERT_MODEL, num_labels=n_labels)

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    with training_args.strategy.scope():
        bert = TFBertForSequenceClassification.from_pretrained(BERT_MODEL,
                                                               config=config)

    train_ds, info = tfds.load('glue/mnli',
                               split='train',
                               with_info=True,
                               data_dir="./data")

    eval_ds, info = tfds.load('glue/mnli',
                              split='validation_matched',
                              with_info=True,
                              data_dir="./data")

    train_ds = glue_convert_examples_to_features(train_ds, tokenizer, 128,
                                                 'mnli')
    eval_ds = glue_convert_examples_to_features(eval_ds, tokenizer, 128,
                                                'mnli')
    train_ds = train_ds.apply(
        tf.data.experimental.assert_cardinality(
            info.splits['train'].num_examples))
    eval_ds = eval_ds.apply(
        tf.data.experimental.assert_cardinality(
            info.splits['validation_matched'].num_examples))

    trainer = TFTrainer(
        model=bert,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model('./models/mnli')
        tokenizer.save_pretrained('./models/mnli')


if __name__ == '__main__':
    args = get_args()
    main(args)