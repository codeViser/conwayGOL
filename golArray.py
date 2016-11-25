import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.patches as patches
import time


#INITIALIZE BOARD BY IMAGE
# img = Image.open('testImageX.png').convert('L')
# data = np.asarray(img.getdata()).reshape(img.size)
# shape = data.shape
# data = data < 128;


#INITIALIZE BOARD BY CUSTOM MATRIX
# data=[[0,0,1,0,0],[0,1,0,1,0],[0,1,1,1,0],[0,1,0,1,0],[0,1,0,1,0]]
# shape = (5,5)
data=np.zeros((30,25))
data[12][8]=data[12][9]=data[12][10]=data[12][12]=1
data[13][8]=1
data[14][11]=data[14][12]=1
data[15][9]=data[15][10]=data[15][12]=1
data[16][8]=data[16][10]=data[16][12]=1
shape = data.shape



#INITIALIZE BOARD RANDOMLY
# initial_board = tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32)


#PROCESSING
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


sizeNet = np.prod(shape)

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

			#Random Init When Repeated
			# initial_board_values = session.run(initial_board)
			# X = session.run(board_update, feed_dict={board: initial_board_values})[0]
			# Y=X

		X=Y
		print("--- %s SCORE ---" % (time.time() - start_time))
		plot.set_array(Y)
	return plot,



ani = animation.FuncAnimation(fig, game_of_life, interval=100, blit=True)
plt.show()
