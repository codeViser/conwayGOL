import tensorflow as tf
from matplotlib import pyplot as plt

shape = (50, 50)
initial_board = tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32)

with tf.Session() as session:
    X = session.run(initial_board)

fig = plt.figure()
plot = plt.imshow(X, cmap='Greys',  interpolation='nearest')
plt.show()
