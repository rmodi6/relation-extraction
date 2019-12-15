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

        gru = layers.GRU(units=hidden_size, return_sequences=True)  # GRU layer
        self.bigru = layers.Bidirectional(gru, merge_mode='concat')  # Bidirectional layer to make the GRU bidirectional

        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START

        M = tf.tanh(rnn_outputs)  # Apply tanh non-linearity on rnn_outputs
        # Element-wise multiplication of last dimension of M with first dimension of omegas and
        # apply softmax over the product on the second dimension
        alpha = tf.nn.softmax(tf.tensordot(M, self.omegas, axes=[-1, 0]), axis=1)
        # Element-wise multiplication of rnn outputs with alpha and reduce sum on the second dimension
        r = tf.reduce_sum(tf.multiply(rnn_outputs, alpha), axis=1)
        output = tf.tanh(r)  # Apply tanh non-linearity on the resultant r

        ### TODO(Students) END

        return output

    def call(self, inputs, pos_inputs, training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        ### TODO(Students) START

        # Get the different dimensions of the inputs
        batch_size, sequence_length, embed_size = word_embed.shape
        # Concatenate the word embeddings and pos embeddings
        input_embed = tf.concat([word_embed, pos_embed], axis=-1)
        # input_embed = word_embed  # Word only experiment
        # Create sequence mask for padded values in the input
        mask = tf.cast(inputs != 0, tf.float32)
        # Feed the input embeddings to the bidirectional gru layer with sequence mask
        h = self.bigru(input_embed, training=training, mask=mask)
        # Apply attention on the hidden outputs of bigru
        output = self.attn(h)
        # Feed the attended hidden outputs to the final output layer [dense layer]
        logits = self.decoder(output)

        ### TODO(Students) END

        return {'logits': logits}


class MyAdvancedModel(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyAdvancedModel, self).__init__()
        ### TODO(Students) START

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        self.bilstm = layers.Bidirectional(layers.LSTM(units=hidden_size, return_sequences=True), merge_mode='concat')
        self.cnn_layers = []
        for kernel_size in [2, 3, 4]:
            cnn = layers.Convolution1D(hidden_size, input_shape=(None, 2 * hidden_size), kernel_size=kernel_size,
                                     activation='tanh')
            self.cnn_layers.append(cnn)
        self.max_pooling = layers.GlobalMaxPooling1D()
        self.dropout_layer = layers.Dropout(rate=0.5)

        ### TODO(Students) END

    def call(self, inputs, pos_inputs, training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        ### TODO(Students) START

        batch_size, sequence_length, embed_size = word_embed.shape
        input_embed = tf.concat([word_embed, pos_embed], axis=-1)
        mask = tf.cast(inputs != 0, tf.float32)

        input_embed = self.bilstm(input_embed, mask=mask)

        hidden_outputs = []
        for cnn_layer in self.cnn_layers:
            hidden_output = cnn_layer(input_embed)
            hidden_output = self.max_pooling(hidden_output)
            hidden_outputs.append(hidden_output)

        h = tf.concat(hidden_outputs, axis=-1)
        if training:
            h = self.dropout_layer(h)
        logits = self.decoder(h)

        ### TODO(Students) END

        return {'logits': logits}
