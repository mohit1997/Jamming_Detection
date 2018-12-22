# Yet to write
import tensorflow as tf
from utils import *
from models import *

window = 100
batch_size = 512
learning_rate = 1e-3
epochs = 20

def split(x, y, a, shuffle=True, f=0.3):
	if shuffle:
		indices = np.arange(len(x))
		np.random.shuffle(indices)

		x = x[indices]
		y = y[indices]
		a = a[indices]

	l = int(len(x)*0.3)

	train_x = x[:-l]
	train_y = y[:-l]
	train_a = a[:-l]

	test_x = x[-l:]
	test_y = y[-l:]
	test_a = a[-l:]

	return train_x, train_y, train_a, test_x, test_y, test_a

def main():
	X, Y, A = gen_and_process(p=0.5, SNR=1.0, N=100000, window=window)
	Y = Y[:, -1:]

	# Input Placeholders
	x = tf.placeholder(tf.float32, [None, window, 2], name='InputData')
	y = tf.placeholder(tf.float32, [None, 1], name='LabelData')
	a = tf.placeholder(tf.float32, [None, 1], name='LabelAttack')

	log, pred, att = fc_att(x)

	# Loss Function
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=log)) + \
		tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=a, logits=att))

	# Optimisation Operation
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	values = tf.round(pred)
	correct_pred = tf.equal(values, y)
	acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	attack_prob = tf.nn.sigmoid(att)
	correct_attack_pred = tf.equal(tf.round(attack_prob), a)
	det_acc = tf.reduce_mean(tf.cast(correct_attack_pred, tf.float32))

	# Initializing Operation
	init = tf.initialize_all_variables()

	sess = tf.Session()

	sess.run(init)

	x_train, y_train, a_train, x_test, y_test, a_test = split(X, Y, A, shuffle=True, f=split)

	for epoch in np.arange(1, epochs+1):
		acc_list = []
		loss_list = []
		det_list = []
		for batch_x, batch_y, batch_a in iterate_minibatches(x_train, y_train, a_train, batchsize=batch_size, shuffle=True):
			_, accuracy, tr_loss, det_accuracy = sess.run([optimizer, acc, loss, det_acc], feed_dict={x: batch_x, y:batch_y, a: batch_a})
			acc_list.append(accuracy)
			loss_list.append(tr_loss)
			det_list.append(det_accuracy)
		print("Epoch ", epoch, " Training Accuracy ", np.mean(acc_list), " Attack Detection Accuracy ", np.mean(det_list), " Training Loss ", np.mean(loss_list))

		val_acc, val_loss, val_det_acc = sess.run([acc, loss, det_acc], feed_dict={x: x_test, y: y_test, a: a_test})
		print("Epoch ", epoch, " Validation Accuracy ", val_acc, " Attack Detection Accuracy ", val_det_acc,  " Validation Loss ", val_loss)

if __name__ == "__main__":
	main()



