import tensorflow as tf
from tensorflow.keras import layers, models

from util import ID_TO_CLASS


class MyBasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students) START
        # ...
        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START
        # ...
        ### TODO(Students) END

        return output

    def call(self, inputs, pos_inputs, training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        ### TODO(Students) START
        # ...
        ### TODO(Students) END

        return {'logits': logits}


class MyAdvancedModel(models.Model):

    def __init__(self):
        super(MyAdvancedModel, self).__init__()
        ### TODO(Students) START
        # ...
        ### TODO(Students END

    def call(self, inputs, pos_inputs, training):
        raise NotImplementedError
        ### TODO(Students) START
        # ...
        ### TODO(Students END
