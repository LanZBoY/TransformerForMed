import tensorflow as tf
from keras import layers

class Linear(layers.Layer):
    
    def __init__(self, unit = 32, input_dim = 32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value= w_init(shape = (input_dim, unit)), dtype="float32",
            trainable=True
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape = (unit, ), dtype="float32"),
            trainable=True
        )
        print(self.w.shape)
        print(self.b.shape)

    def call(self, inputs):
        x = tf.matmul(inputs, self.w) + self.b
        return x