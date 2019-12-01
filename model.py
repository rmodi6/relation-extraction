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

        gru = layers.GRU(units=hidden_size, return_sequences=True)
        self.bigru = layers.Bidirectional(gru, merge_mode='concat')

        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START

        M = tf.tanh(rnn_outputs)
        alpha = tf.nn.softmax(tf.matmul(M, self.omegas))
        r = tf.matmul(rnn_outputs, alpha, transpose_a=True)
        output = tf.tanh(r)

        ### TODO(Students) END

        return output

    def call(self, inputs, pos_inputs, training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        ### TODO(Students) START

        batch_size, sequence_length, embed_size = word_embed.shape
        input_embed = tf.concat([word_embed, pos_embed], axis=-1)
        h = self.bigru(input_embed, training=training)
        output = self.attn(h)
        logits = self.decoder(tf.reshape(output, [batch_size, h.shape[-1]]))

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
