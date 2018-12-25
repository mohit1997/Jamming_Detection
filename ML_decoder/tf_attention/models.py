import tensorflow as tf

def gaussian_noise_layer(input_layer, std, is_train):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    output = tf.cond(is_train, lambda: input_layer + noise, lambda: input_layer)
    return output


def fc_att(x, is_train):
	inp = gaussian_noise_layer(x, std=0.5, is_train=is_train)

	shp = x.get_shape().as_list()
	window = shp[1]
	channels = shp[2]

	flat = tf.contrib.layers.flatten(inp)
	y1, y2 = tf.unstack(inp, num=channels, axis=2)

	h1 = tf.layers.dense(flat, 256, activation=tf.nn.relu)
	h1 = gaussian_noise_layer(h1, std=0.2, is_train=is_train)
	h2 = tf.layers.dense(h1, 128, activation=tf.nn.relu)
	h2 = gaussian_noise_layer(h2, std=0.2, is_train=is_train)
	att = tf.layers.dense(h2, 1)

	y2_activated = y2 * tf.nn.sigmoid(att)

	activated_inp = tf.concat([y1, y2_activated], axis=1)

	hidden1 = tf.layers.dense(activated_inp, 256, activation=tf.nn.relu)
	hidden1 = gaussian_noise_layer(hidden1, std=0.2, is_train=is_train)
	hidden2 = tf.layers.dense(hidden1, 128, activation=tf.nn.relu)
	hidden2 = gaussian_noise_layer(hidden2, std=0.2, is_train=is_train)
	logits = tf.layers.dense(hidden2, shp[1])
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
