from typing import Iterator

import conllu
import numpy as np
import networkx as nx
import tensorflow as tf


def label_matrices_padding(label_matrices, max_length):
    '''Pads label matrices to max length

    Args:
        label_matrics (List[np.ndarray])
        max_length (int)

    Returns:
        padded_matrices (np.ndarray)
    '''
    ret = []

    for sent_labels in label_matrices:
        sent_length = sent_labels.shape[-1]
        pad_length = max_length - sent_length
        ret.append(tf.pad(sent_labels, [[0, pad_length], [0, pad_length]]))

    return np.array(ret)


class ProbeDataset:
    def __init__(self, ud_path: str):
        self._ud_path = ud_path
        self._graph = nx.Graph()

    def traverse_tree(self, root):
        self._graph.add_node(root.token['id'] - 1)

        for child in root.children:
            self._graph.add_node(child.token['id'] - 1)
            self._graph.add_edge(root.token['id'] - 1, child.token['id'] - 1)
            self.traverse_tree(child)

    def data_generator(self, sentences) -> Iterator:

        for sentence in sentences:
            self.traverse_tree(sentence.to_tree())
            dim = len(sentence)
            label_matrix = np.zeros(shape=(dim, dim))

            for start, targets in nx.all_pairs_shortest_path_length(
                    self._graph):
                try:
                    for target, distance in targets.items():
                        label_matrix[start, target] = distance

                except:
                    print(
                        f'Catch invalid shape: {label_matrix.shape, start, target}'
                    )
                    continue

            yield (sentence.tokens, label_matrix)
            self._graph.clear()

    def build_pairs_distance_generator(self,
                                       bert,
                                       tokenizer,
                                       batch_size: int = 32,
                                       layer_index: int = -1,
                                       return_length: bool = True) -> Iterator:
        '''Builds probe model pairs distance generator
        '''
        with open(self._ud_path, 'r', encoding='utf-8') as file:
            sentences = conllu.parse(file.read())

        counter = 0
        sents, label_matrices = [], []

        for sent in self.data_generator(sentences):
            tokens, label_matrix = sent

            if counter >= batch_size:
                inputs = [[token['form'] for token in sent] for sent in sents]
                inputs = tokenizer(inputs,
                                   padding=True,
                                   is_split_into_words=True,
                                   return_tensors='tf',
                                   return_length=True,
                                   add_special_tokens=False)
                lengths = inputs['length']
                max_length = tf.reduce_max(inputs['length']).numpy()
                label_matrices = label_matrices_padding(
                    label_matrices, max_length)
                del inputs['length']  # delete key for feeding model correctly
                outputs = bert(**inputs)
                embeddings = outputs.hidden_states[layer_index]

                if return_length:
                    yield embeddings, label_matrices, lengths
                else:
                    yield embeddings, label_matrices
                counter = 0
                sents, label_matrices = [], []

            else:
                counter += 1
                sents.append(tokens)
                label_matrices.append(label_matrix)
