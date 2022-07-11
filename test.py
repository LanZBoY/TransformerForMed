import tensorflow as tf
from tutorial import Linear

x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)