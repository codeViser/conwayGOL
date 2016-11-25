import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

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
plt.show()
