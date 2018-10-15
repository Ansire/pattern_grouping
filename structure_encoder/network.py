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

import tensorflow.contrib.layers as tf_layers

import layer

def represent(images, num_output_channels):
	"""
		input:
			images               : n   x H x W x 1   input images ( 800 x 800 x 1 )
			num_output_channels  : int               number of output image channels
		output:
			results              : n   x H x W x C   output representation ( 800 x 800 x C )
	"""

	###### encoding ######

	nc = num_output_channels

	e0 = tf_layers.conv2d(images, num_outputs= 16, kernel_size=16, stride=1, scope='e0', normalizer_fn=None) # 800 x 800 x  16
	e1 = tf_layers.conv2d(    e0, num_outputs= 32, kernel_size=8,  stride=1, scope='e1')                     # 800 x 800 x  32
	e1 = tf_layers.max_pool2d(e1, kernel_size=2, stride=2, padding='same')                                   # 400 x 400 x  32
	e2 = tf_layers.conv2d(    e1, num_outputs= 64, kernel_size=8,  stride=1, scope='e2')                     # 400 x 400 x  64
	e2 = tf_layers.max_pool2d(e2, kernel_size=2, stride=2, padding='same')                                   # 200 x 200 x  64
	e3 = tf_layers.conv2d(    e2, num_outputs=128, kernel_size=4,  stride=2, scope='e3')                     # 100 x 100 x 128
	e4 = tf_layers.conv2d(    e3, num_outputs=128, kernel_size=4,  stride=2, scope='e4')                     #  50 x  50 x 128
	e5 = tf_layers.conv2d(    e4, num_outputs=128, kernel_size=4,  stride=2, scope='e5')                     #  25 x  25 x 128
	e6 = tf_layers.conv2d(    e5, num_outputs=256, kernel_size=4,  stride=2, scope='e6')                     #  13 x  13 x 256

	###### decoding ######

	d5 =      layer.unconv_layer(                    e6, num_outputs=128, kernel_size=4,  image_size=    e5.get_shape()[1].value, scope='d5')   #  25 x  25 x 128
	d4 =      layer.unconv_layer(tf.concat([d5, e5], 3), num_outputs=128, kernel_size=4,  image_size=    e4.get_shape()[1].value, scope='d4')   #  50 x  50 x 128
	d3 =      layer.unconv_layer(tf.concat([d4, e4], 3), num_outputs=128, kernel_size=4,  image_size=    e3.get_shape()[1].value, scope='d3')   # 100 x 100 x 128
	d2 =      layer.unconv_layer(tf.concat([d3, e3], 3), num_outputs= 64, kernel_size=4,  image_size=    e2.get_shape()[1].value, scope='d2')   # 200 x 200 x  64
	d1 =      layer.unconv_layer(tf.concat([d2, e2], 3), num_outputs= 32, kernel_size=8,  image_size=    e1.get_shape()[1].value, scope='d1')   # 400 x 400 x  32
	d0 =      layer.unconv_layer(tf.concat([d1, e1], 3), num_outputs= 16, kernel_size=8,  image_size=    e0.get_shape()[1].value, scope='d0')   # 800 x 800 x  16
	results = layer.unconv_layer(tf.concat([d0, e0], 3), num_outputs= nc, kernel_size=16, image_size=images.get_shape()[1].value, scope='re', normalizer_fn=None, activation_fn=None)

	return results
