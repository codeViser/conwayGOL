import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

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
plt.show()
