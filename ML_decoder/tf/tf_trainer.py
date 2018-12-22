# Yet to write
import tensorflow as tf
from utils import *
from models import *

window = 100
batch_size = 512
learning_rate = 1e-3
epochs = 5

def split(x, y, shuffle=True, f=0.3):
	if shuffle:
		indices = np.arange(len(x))
		np.random.shuffle(indices)

		x = x[indices]
		y = y[indices]

	l = int(len(x)*0.3)

	train_x = x[:-l]
	train_y = y[:-l]

	test_x = x[-l:]
	test_y = y[-l:]

	return train_x, train_y, test_x, test_y

def main():
	X, Y = gen_and_process(p=0.4, SNR=1.0, N=100000, window=window)
	X = X[:, :, 0:1]
	Y = Y[:, -1:]

	# Input Placeholders
	x = tf.placeholder(tf.float32, [None, window, 1], name='InputData')
	y = tf.placeholder(tf.float32, [None, 1], name='LabelData')

	log, pred = fc(x)

	# Loss Function
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=log))

	# Optimisation Operation
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	values = tf.round(pred)
	correct_pred = tf.equal(values, y)
	acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# Initializing Operation
	init = tf.initialize_all_variables()

	sess = tf.Session()

	sess.run(init)

	x_train, y_train, x_test, y_test = split(X, Y, shuffle=True, f=split)

	for epoch in np.arange(1, epochs+1):
		acc_list = []
		loss_list = []
		for batch_x, batch_y in iterate_minibatches(x_train, y_train, batchsize=batch_size, shuffle=True):
			_, accuracy, tr_loss = sess.run([optimizer, acc, loss], feed_dict={x: batch_x, y:batch_y})
			acc_list.append(accuracy)
			loss_list.append(tr_loss)
		print("Epoch ", epoch, " Training Accuracy ", np.mean(acc_list), " Training Loss ", np.mean(loss_list))

		val_acc, val_loss = sess.run([acc, loss], feed_dict={x: x_test, y: y_test})
		print("Epoch ", epoch, " Validation Accuracy ", val_acc, " Validation Loss ", val_loss)

if __name__ == "__main__":
	main()



