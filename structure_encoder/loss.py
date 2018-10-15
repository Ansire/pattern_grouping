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

import image

def compute_contrastive_loss(features, triplets):
	"""
		input:
			features    : n x H x W x C    pattern representation features
			triplets    : n x T x 3 x 2    triplets ( T triplets x {A, B, C} x {h, w} )
			patch_size  : int              patch size in pixels
		output:
			pos_loss    : scalar           positive loss
			neg_loss    : scalar           negative loss
			disc_loss   : scalar           discrete loss
	"""

	## tf.gather_nd seems buggy for backprop in tf1.0
	# batch_indices = tf.tile(tf.reshape(tf.range(features.get_shape()[0]), [-1,1,1,1]), [1,triplets.get_shape()[1].value,3,1]) # n x T x 3 x 1
	# gather_indices = tf.concat([batch_indices, triplets], 3) # n x T x 3 x 3
	# triplet_features = tf.gather_nd(features, gather_indices) # n x T x 3 x C

	## use tf.gather instead
	size_h = features.get_shape()[1].value
	size_w = features.get_shape()[2].value
	size_c = features.get_shape()[3].value
	feature_stacked = tf.reshape(features, [-1, size_c]) # (n*H*W) x C
	index_batch = tf.tile(tf.reshape(tf.range(features.get_shape()[0].value), [-1,1,1]), [1,triplets.get_shape()[1].value,3]) # n x T x 3
	index_height, index_width = tf.unstack(triplets, axis=3) # n x T x 3
	index_stacked = index_batch * (size_h*size_w) + index_height * size_w + index_width # n x T x 3
	triplet_features = tf.gather(feature_stacked, index_stacked) # n x T x 3 x C

	margin = 1.0 # unit margin
	norm_ord = 2 # L-2 distance
	split_features = tf.unstack(triplet_features, axis=2) # [ n x T x C ] * 3
	pos_norm = tf.norm(split_features[0]-split_features[1], ord=norm_ord, axis=2) # n x T
	neg_norm = tf.norm(split_features[0]-split_features[2], ord=norm_ord, axis=2) # n x T

	pos_loss = tf.reduce_mean(tf.square(pos_norm))
	neg_loss = tf.reduce_mean(tf.square(tf.maximum(margin - neg_norm, 0.0)))
	disc_loss = tf.reduce_sum(tf.cast(tf.greater(pos_norm, neg_norm), tf.float32))

	return pos_loss, neg_loss, disc_loss

def compute_total_variation_loss(features):
	"""
		input:
			features    : n x H x W x C    pattern representation features
		output:
			loss        : scalar           loss value
	"""

	shape = features.get_shape().as_list()
	diff_v = features[:,1:,:,:] - features[:,:shape[1]-1,:,:] # n x (H-1) x W x C
	diff_h = features[:,:,1:,:] - features[:,:,:shape[2]-1,:] # n x H x (W-1) x C
	loss = tf.reduce_mean(tf.norm(diff_v, axis=3)) + tf.reduce_mean(tf.norm(diff_h, axis=3))

	return loss