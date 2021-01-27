import re
from typing import Dict

import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer


@tf.function
def get_text(example: Dict[str, tf.Tensor]) -> tf.Tensor:
    text = example['text']
    text = tf.py_function(func=lambda x: re.findall(
        r'(?<=_START_PARAGRAPH_\n).+',
        x.numpy().decode('utf-8'))[0].replace('_NEWLINE_', ''),
                          inp=[text],
                          Tout=tf.string)
    return text


class BertComparator:
    def __init__(self, bert_base: TFBertModel, bert_finetune: TFBertModel,
                 tokenizer: BertTokenizer):
        self._bert_base = bert_base
        self._bert_finetune = bert_finetune
        self._tokenizer = tokenizer

    def get_layer_cos_sim_in_data(self,
                                  data: tf.data.Dataset,
                                  n_samples: int = 5000) -> np.ndarray:
        '''Get layer cosine similarities in dataset

        Args:
            data (tf.data.Dataset)
            n_samples (int)

        Returns:
            cosine_similarities (np.ndarray): cos similarity of layers.
        '''
        data = data.map(get_text)

        cos_sims = []
        cos_metric = tf.metrics.CosineSimilarity()

        for text in data.take(n_samples):
            layer_cos_sims = []
            text = text.numpy().decode('utf-8')
            input_ids = self._tokenizer.encode(text,
                                               return_tensors='tf',
                                               max_length=512,
                                               truncation=True)
            base_outputs = self._bert_base(input_ids)
            finetune_outputs = self._bert_finetune(input_ids)

            for layer_idx in range(len(base_outputs.hidden_states)):
                base_hs, finetune_hs = base_outputs.hidden_states[
                    layer_idx], finetune_outputs.hidden_states[layer_idx]
                layer_cos_sims.append(cos_metric(base_hs, finetune_hs))
                cos_metric.reset_states()

            cos_sims.append(layer_cos_sims)

        return np.mean(cos_sims, axis=0)

    def plot_rsa(self):
        '''Draws Rrepresentational Similarity Analysis.
        '''
