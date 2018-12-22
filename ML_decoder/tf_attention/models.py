import tensorflow as tf

def fc_att(x):
	inp = x

	shp = x.get_shape().as_list()
	window = shp[1]
	channels = shp[2]

	flat = tf.contrib.layers.flatten(inp)
	y1, y2 = tf.unstack(x, num=channels, axis=2)

	h1 = tf.layers.dense(flat, 128, activation=tf.nn.relu)
	h2 = tf.layers.dense(h1, 64, activation=tf.nn.relu)
	att = tf.layers.dense(h2, 1)

	y2_activated = y2 * tf.nn.sigmoid(att)

	activated_inp = tf.concat([y1, y2_activated], axis=1)

	hidden1 = tf.layers.dense(activated_inp, 256, activation=tf.nn.relu)
	hidden2 = tf.layers.dense(hidden1, 64, activation=tf.nn.relu)

	logits = tf.layers.dense(hidden2, 1)
	predictions = tf.nn.sigmoid(logits)

	return logits, predictions, att


def fc(x):
	inp = x

	shp = x.get_shape().as_list()
	window = shp[1]
	channels = shp[2]

	flat = tf.contrib.layers.flatten(inp)

	hidden1 = tf.layers.dense(flat, 128, activation=tf.nn.relu)
	hidden2 = tf.layers.dense(hidden1, 32, activation=tf.nn.relu)

	logits = tf.layers.dense(hidden2, 1)
	predictions = tf.nn.sigmoid(logits)

	return logits, predictions
