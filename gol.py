import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import matplotlib.animation as animation


shape = (50, 50)
initial_board = tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32)

with tf.Session() as session:
    X = session.run(initial_board)

fig = plt.figure()
plot = plt.imshow(X, cmap='Greys',  interpolation='nearest')
#plt.show()




def update_board(X):
    # Check out the details at: https://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/
    # Compute number of neighbours,
    N = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X
    # Apply rules of the game
    X = (N == 3) | (X & (N == 2))
    return X


board = tf.placeholder(tf.int32, shape=shape, name='board')
board_update = tf.py_func(update_board, [board], [tf.int32])



# X=[]

with tf.Session() as session:
    global X
    initial_board_values = session.run(initial_board)
    X = session.run(board_update, feed_dict={board: initial_board_values})[0]



def game_of_life(*args):
	global X
	with tf.Session() as session:
		Y = session.run(board_update, feed_dict={board: X})[0]
		Z = session.run(board_update, feed_dict={board: Y})[0]
		if np.array_equal(X,Z):
			initial_board_values = session.run(initial_board)
			X = session.run(board_update, feed_dict={board: initial_board_values})[0]
			Y=X

		X=Y
		plot.set_array(Y)
	return plot,



ani = animation.FuncAnimation(fig, game_of_life, interval=1, blit=True)
plt.show()