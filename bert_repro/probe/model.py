import tensorflow as tf


class TwoWordProbe(tf.keras.Model):
    '''Pair words positive semi-definited linear transformation model. 
    
    Computes euclidean distance after projection by a matrix.

    ref: https://github.com/john-hewitt/structural-probes/blob/master/structural-probes/probe.py
    '''
    def __init__(self, model_dim, probe_rank):
        super(TwoWordProbe, self).__init__()
        self.B = tf.Variable(initial_value=tf.random.uniform(
            (model_dim, probe_rank), -0.05, 0.05, dtype=tf.float32))

    def call(self, batch):
        '''Feed forword for computing pairs distances after projection

        Args:
            batch (tf.Tensor): batch with size (batch, max_seq_len, vec_dim)

        Returns:
            euclidean_distances (tf.Tensor): L2 distances with shape (batch, max_seq_len, max_seq_len)
        '''
        transformed = tf.matmul(batch, self.B)
        _, seq_len, _ = tf.shape(transformed)

        transformed = tf.expand_dims(transformed, axis=2)
        transformed = tf.tile(transformed, multiples=(1, 1, seq_len, 1))
        transposed = tf.transpose(transformed, perm=(0, 2, 1, 3))
        diffs = transformed - transposed
        squared_diffs = tf.square(diffs)
        euclidean_distances = tf.math.reduce_sum(squared_diffs, axis=-1)

        return euclidean_distances


class OneWordProbe(tf.keras.Model):
    '''Computes squared L2 norm of words after projection by a matrix.

    ref: https://github.com/john-hewitt/structural-probes/blob/master/structural-probes/probe.py
    '''
    def __init__(self, model_dim, probe_rank):
        super(OneWordProbe, self).__init__()
        self.B = tf.Variable(initial_value=tf.random.uniform(
            (model_dim, probe_rank), -0.05, 0.05, dtype=tf.float32))

    def call(self, batch):
        '''Computes all n depths after projection

        Args:
            batch (tf.Tensor): batch with size (batch, max_seq_len, vec_dim)

        Returns:
            L2 norms (tf.Tensor): L2 norms with size (batch, max_seq_len, max_seq_len)
        '''
        transformed = tf.matmul(batch, self.B)
        batch_len, seq_len, rank = tf.shape(transformed)
        norms = tf.matmul(tf.reshape(batch_len * seq_len, 1, rank),
                          tf.reshape(batch_len * seq_len, rank, 1))
        norms = norms.reshape(batch_len, seq_len)

        return norms
