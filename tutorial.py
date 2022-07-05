import tensorflow as tf
from keras import layers

class Linear(layers.Layer):
    
    def __init__(self, unit = 32, input_dim = 32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()