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
import numpy as np
import scipy.io

import os

import image

def load_data(config, pattern_list, shuffle=True, batch_size=-1):
	"""
		input:
			config               tf.app.flags        command line arguments
			pattern_list         list of string      input pattern name list
			shuffle              bool                whether input list should be shuffled
		output:
			name_batch           n x string          pattern names
			image_batch          n x H x W x 1       pattern images
			triplet_batch        n x T x 3 x 2       triplets of patch coordinates (# triplets x {A,B,C} x {h,w})
			num_patterns         int                 number of loaded patterns
	"""

	print('Loading data...')

	if batch_size==-1:
		batch_size = config.batch_size

	# build input queue

	pattern_list_queue = tf.train.input_producer(pattern_list, shuffle=shuffle)
	pattern_name = pattern_list_queue.dequeue()
	
	# decode pattern image

	#image_file = config.data_dir+'image/'+pattern_name+'.png'
	image_file = config.data_dir+'region/'+pattern_name+'.png'
	image_tensor = tf.image.decode_png(tf.read_file(image_file), channels=1, dtype=tf.uint8)
	image_tensor = image.normalize_image(tf.slice(image_tensor, [0,0,0], [config.image_size, config.image_size, -1])) # just do a useless slicing to establish size

	# decode pattern triplets

	if not config.real_data:
		triplet_file = config.data_dir+'triplet-region/'+pattern_name+'.bin'
		triplet_data = tf.read_file(triplet_file)
		triplet_tensor = tf.reshape(tf.decode_raw(triplet_data, tf.int16), [-1, 3, 2])
		triplet_tensor = tf.cast(triplet_tensor, dtype=tf.int32)
		triplet_tensor = tf.slice(tf.random_shuffle(triplet_tensor), [0,0,0], [config.num_triplets,-1,-1])
	else:
		triplet_tensor = tf.zeros([1, 3, 2], dtype=tf.int32)

	# create prefetching tensor

	num_patterns = len(pattern_list)
	min_queue_examples = max(1, int(num_patterns* 0.01))

	tensor_data = [pattern_name, image_tensor, triplet_tensor]

	if shuffle:
		num_preprocess_threads = 12
		batch_data = tf.train.shuffle_batch(
			tensor_data,
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples + 3 * batch_size,
			min_after_dequeue=min_queue_examples)
	else:
		num_preprocess_threads = 1
		batch_data = tf.train.batch(
			tensor_data,
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples)

	name_batch = batch_data[0]
	image_batch = batch_data[1]
	triplet_batch =  batch_data[2]

	print('name: ', name_batch)
	print('image: ', image_batch)
	print('triplet: ', triplet_batch)

	return name_batch, image_batch, triplet_batch, num_patterns

def load_train_data(config, batch_size=-1):

	print("Loading training data...")

	pattern_list_file = open(os.path.join(config.data_dir, 'train-list.txt'), 'r')
	pattern_list = pattern_list_file.read().splitlines()
	pattern_list_file.close()

	return load_data(config, pattern_list, shuffle=True, batch_size=batch_size)

def load_test_data(config, batch_size=-1):

	print("Loading testing data...")

	pattern_list_file = open(os.path.join(config.data_dir, 'list.txt'), 'r')
	pattern_list = pattern_list_file.read().splitlines()
	pattern_list_file.close()

	return load_data(config, pattern_list, shuffle=False, batch_size=batch_size)

def load_validate_data(config, batch_size=-1):

	print("Loading validation data...")

	pattern_list_file = open(os.path.join(config.data_dir, 'validate-list.txt'), 'r')
	pattern_list = pattern_list_file.read().splitlines()
	pattern_list_file.close()

	return load_data(config, pattern_list, shuffle=False, batch_size=batch_size)

def write_bin_data(file_name, data):

	path = os.path.dirname(file_name)
	if not os.path.exists(path):
		os.makedirs(path)
	data.tofile(file_name)

def extract_element_feature(image_feature, image_name, data_dir):
	"""
		input:
			image_feature   : H x W x C   :  learned pixel-wise features
			image_name      : string      :  image name
			data_dir        : string      :  data root directory
		output:
			element_feature : m x C       :  aggregated feature for each element
	"""

	element_name = os.path.join(data_dir, 'element', image_name+'.mat')
	element = scipy.io.loadmat(element_name)['eleIDMatrix'] # H x W
	num_elements = element.max()
	num_channels = image_feature.shape[2]
	flatten_feature = image_feature.reshape((-1,num_channels)) # (H*W) x C
	flatten_element = element.reshape(-1) # (H*W)
	element_feature = np.zeros((num_elements, num_channels), dtype=np.float32) # m x C
	for element_id in range(num_elements):
		element_feature[element_id,:] = flatten_feature[(flatten_element==element_id+1).nonzero(),:].sum(axis=1) # 1 x C
	return element_feature
