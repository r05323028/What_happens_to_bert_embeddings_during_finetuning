import numpy as np
import tensorflow as tf

from tqdm import tqdm


def build_label_mask(length, max_length):
    outputs = []

    for l in length:
        mask = tf.ones((l, l))
        outputs.append(tf.pad(mask,
                              [[0, max_length - l], [0, max_length - l]]))

    return outputs


class ProbeTrainer:
    '''Probe Model Trainer
    '''
    def __init__(self, n_epochs: int = 20, learning_rate: float = 0.00003):
        self._n_epochs = n_epochs
        self._lr = learning_rate

    def train(self, probe_model, train_dataset, dev_dataset=None):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self._lr)

        for _ in range(self._n_epochs):
            pbar = tqdm(train_dataset)
            for batch in pbar:
                features, label_matrix, length = batch
                max_length = label_matrix.shape[1]
                label_mask = build_label_mask(length, max_length)

                with tf.GradientTape() as tape:
                    distances = probe_model(features, training=True)
                    loss = tf.reduce_sum(
                        tf.abs(distances * label_mask - label_matrix *
                               label_mask)) / tf.square(float(max_length))

                pbar.set_description(f'Loss: {loss}')

                grads = tape.gradient(loss, probe_model.trainable_variables)

                optimizer.apply_gradients(
                    zip(grads, probe_model.trainable_variables))

        return probe_model