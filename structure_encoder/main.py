"""

This file is part of the pattern grouping project.
Copyright (c) 2017 - Zhaoliang Lun / UMass-Amherst
This is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this software.  If not, see <http://www.gnu.org/licenses/>.

"""

import tensorflow as tf

import time
import os

import data
import patnet as pn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('train', False,
							"""Flag for training routine.""")
tf.app.flags.DEFINE_boolean('test', True,
							"""Flag for testing routine.""")
tf.app.flags.DEFINE_boolean('real_data', False,
							"""Flag for testing on real data.""")
tf.app.flags.DEFINE_boolean('visualize_feature', False,
							"""Flag for visualizing feature maps.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
							"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 800,
							"""Size of images to be learned.""")
tf.app.flags.DEFINE_integer('feature_size', 8,
							"""Size of representation features.""")
tf.app.flags.DEFINE_integer('num_triplets', 50,
							"""Number of triplets for training.""")
tf.app.flags.DEFINE_float('max_epochs', 1000.0,
							"""Maximum epochs for optimization.""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.7,
							"""Upper-bound fraction of GPU memory usage.""")
tf.app.flags.DEFINE_string('data_dir', '/',
							"""Directory containing training/testing data.""")
tf.app.flags.DEFINE_string('train_dir', './trainlog/',
							"""Directory where to write training logs.""")
tf.app.flags.DEFINE_string('test_dir', '/',
							"""Directory where to write testing logs.""")

def main(argv=None):

	if not os.path.exists(FLAGS.train_dir):
		os.makedirs(FLAGS.train_dir)

	if not os.path.exists(FLAGS.test_dir):
		os.makedirs(FLAGS.test_dir)

	print('start running...')
	start_time = time.time()

	############################################ build graph ############################################

	patnet = pn.PatNet(FLAGS)

	if int(FLAGS.train) + int(FLAGS.test) != 1:
		print('please specify \'train\' or \'test\'')
		return

	if FLAGS.train:
		train_names, train_images, train_triplets, num_train_patterns = data.load_train_data(FLAGS)
		valid_names, valid_images, valid_triplets, num_valid_patterns = data.load_validate_data(FLAGS)

		with tf.variable_scope("patnet") as scope:
			patnet.build_network(\
				names=train_names,
				images=train_images,
				triplets=train_triplets,
				is_training=True)
			scope.reuse_variables() # sharing weights
			patnet.build_network(\
				names=valid_names,
				images=valid_images,
				triplets=valid_triplets,
				is_validation=True)
	elif FLAGS.test:
		test_names, test_images, test_triplets, num_test_patterns = data.load_test_data(FLAGS)

		with tf.variable_scope("patnet") as scope:
			patnet.build_network(\
				names=test_names,
				images=test_images,
				triplets=test_triplets,
				is_testing=True)


	############################################ compute graph ############################################

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction)

	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
		log_device_placement=False,
		allow_soft_placement=True)) as sess:

		if FLAGS.train:
			patnet.train(sess, num_train_patterns, num_valid_patterns)
		elif FLAGS.test:
			patnet.test(sess, num_test_patterns)

		sess.close()

	duration = time.time() - start_time
	print('total running time: %.1f\n' % duration)


if __name__ == '__main__':
	tf.app.run()
