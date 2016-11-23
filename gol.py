import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.patches as patches
import time


img = Image.open('testImage4.png').convert('L')
data = np.asarray(img.getdata()).reshape(img.size)
shape = data.shape
data = data < 128;
#custInp=np.loadtxt(open("customInput2.csv","rb"),delimiter=",",skiprows=0)
# initial_board = tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32)
initial_board = tf.constant(data, dtype=tf.int32)

with tf.Session() as session:
	print(shape)
	print(session.run(initial_board))
	X = session.run(initial_board)

fig = plt.figure()
plot = plt.imshow(X, cmap='PuBu',  interpolation='nearest')
#plt.show()


start_time = time.time()

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


sizeNet = np.prod(data.shape)
def game_of_life(*args):
	global X
	with tf.Session() as session:
		Y = session.run(board_update, feed_dict={board: X})[0]
		Z = session.run(board_update, feed_dict={board: Y})[0]
		diff =  np.count_nonzero(X==Z)
		if (diff > 0.98*sizeNet):
			print("--- %s SCORE ---" % (time.time() - start_time))
			time.sleep(30)
			exit(0)

		if np.array_equal(X,Z):
			print("--- %s SCORE ---" % (time.time() - start_time))
			time.sleep(30)
			exit(0)


			# initial_board_values = session.run(initial_board)
			# X = session.run(board_update, feed_dict={board: initial_board_values})[0]
			# Y=X

		X=Y
		print("--- %s SCORE ---" % (time.time() - start_time))
		plot.set_array(Y)
	return plot,



ani = animation.FuncAnimation(fig, game_of_life, interval=1, blit=True)
plt.show()
