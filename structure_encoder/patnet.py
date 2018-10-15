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

import tensorflow.contrib.framework as tf_framework

import time
import os
import math

import data
import image
import network
import layer
import loss

class PatNet(object):

	def __init__(self, config):
		self.config = config

	def build_network(self, names, images, triplets, is_training=False, is_validation=False, is_testing=False):
		"""
			input:
				names         :  n x String             image names
				images        :  n x H x W x 1          pattern images
				triplets      :  n x T x 3 x 2          triplets ( T triplets x {A, B, C} x {h, w} )
				is_training   :  boolean                whether it is in training routine
				is_validation :  boolean                whether it is handling validation data set
				is_testing    :  boolean                whether it is in testing routine
		"""

		print('Building network...')

		self.names = names

		# scope names

		bn_scope = 'bn'
		train_summary_name = 'train_summary'
		valid_summary_name = 'valid_summary'

		# network

		with tf_framework.arg_scope(layer.unet_scopes(bn_scope, is_training)):
			features = network.represent(images, self.config.feature_size) # n x H x W x C

		# extract features

		if self.config.test:
			self.image_features = features
			#batch_size = self.config.batch_size
			#batch_images = tf.unstack(tf.squeeze(images, axis=3), axis=0) # [H x W] * n
			#batch_features = tf.unstack(features, axis=0) # [H x W x C] * n
			#self.masked_features = [None] * batch_size
			#for batch_id in range(batch_size):
			#	mask_indices = tf.where(tf.greater(batch_images[batch_id], 0.0)) # m x 2
			#	self.masked_features[batch_id] = tf.gather_nd(batch_features[batch_id], mask_indices) # m x C

			if self.config.visualize_feature:
				size_h = features.get_shape()[1].value;
				size_w = features.get_shape()[2].value;
				reshaped = tf.transpose(features, perm=[0,3,1,2]);
				reshaped = tf.reshape(reshaped, [-1, size_h, size_w, 1]);
				unpacked = tf.unstack(reshaped)
				size_n = len(unpacked)
				encoded = [None] * size_n
				for k in range(size_n):
					normalized = unpacked[k];
					tmax = tf.reduce_max(normalized);
					tmin = tf.reduce_min(normalized);
					normalized = (normalized-tmin)/(tmax-tmin);
					encoded[k] = tf.image.encode_png(tf.image.convert_image_dtype(normalized, dtype=tf.uint8))
				self.visual_features = encoded;
			return
		
		# loss

		alpha = 0.1
		pos_loss, neg_loss, disc_loss = loss.compute_contrastive_loss(features, triplets)
		tv_loss = loss.compute_total_variation_loss(features)
		self.loss = pos_loss + neg_loss + tv_loss * alpha
		self.losses = tf.stack([self.loss, pos_loss, neg_loss, tv_loss, disc_loss])

		# validation

		if is_validation:
			self.valid_summary_loss = tf.placeholder(tf.float32, shape=self.losses.get_shape())
			valid_total_loss, valid_pos_loss, valid_neg_loss, valid_tv_loss, valid_disc_loss = tf.unstack(self.valid_summary_loss)
			tf.summary.scalar('validation total loss', valid_total_loss, collections=[valid_summary_name])
			tf.summary.scalar('validation positive loss', valid_pos_loss, collections=[valid_summary_name])
			tf.summary.scalar('validation negative loss', valid_neg_loss, collections=[valid_summary_name])
			tf.summary.scalar('validation variation loss', valid_tv_loss, collections=[valid_summary_name])
			tf.summary.scalar('validation discrete loss', valid_disc_loss, collections=[valid_summary_name])
			self.valid_summary_op = tf.summary.merge_all(valid_summary_name)
			return # all stuffs below are irrelevant to validation pass

		# statistics

		all_vars = tf.trainable_variables()
		print('Num all vars: %d' % len(all_vars))
		num_params = 0
		# print('All vars:')
		for var in all_vars:
			num_params += np.prod(var.get_shape().as_list())
			# print(var.name, var.get_shape().as_list())
		print('Num all params: %d' % num_params)

		# optimization

		init_learning_rate = 0.0001
		adam_beta1 = 0.9
		adam_beta2 = 0.999
		opt_step = tf.Variable(0, trainable=False)
		learning_rate = tf.train.exponential_decay(init_learning_rate, global_step=opt_step, decay_steps=10000, decay_rate=0.96, staircase=True)
		
		opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=adam_beta1, beta2=adam_beta2, name='ADAM')
		# opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name='SGD')

		grads = opt.compute_gradients(self.loss, var_list=all_vars, colocate_gradients_with_ops=True)
		# grads = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in grads] # gradient clipping
		self.grad_placeholder = [(tf.placeholder(tf.float32, shape=grad[1].get_shape()), grad[1]) for grad in grads if grad[0] is not None]
		self.grad_list = [grad[0] for grad in grads if grad[0] is not None]
		self.update_op = opt.apply_gradients(self.grad_placeholder, global_step=opt_step)

		# summary

		tf.summary.scalar('total loss', self.loss, collections=[train_summary_name])
		tf.summary.scalar('positive loss', pos_loss, collections=[train_summary_name])
		tf.summary.scalar('negative loss', neg_loss, collections=[train_summary_name])
		tf.summary.scalar('variation loss', tv_loss, collections=[train_summary_name])
		tf.summary.scalar('discrete loss', disc_loss, collections=[train_summary_name])
		self.train_summary_op = tf.summary.merge_all(train_summary_name)

		# batch normalization

		bn_collection = tf.get_collection(bn_scope)
		self.bn_op = tf.group(*bn_collection)

	def train(self, sess, num_train_patterns, num_valid_patterns):

		print('Training...')

		ckpt = tf.train.get_checkpoint_state(self.config.train_dir)
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		if ckpt and ckpt.model_checkpoint_path:
			#self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=10.0, max_to_keep=2)
			self.saver = tf.train.Saver(max_to_keep=2)
			self.saver.restore(sess, ckpt.model_checkpoint_path)
			self.step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
		else:
			#self.saver = tf.train.Saver(tf.global_variables(), keep_checkpoint_every_n_hours=10.0, max_to_keep=2)
			self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
			self.step = 0

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		self.summarizer = tf.summary.FileWriter(self.config.train_dir, sess.graph)

		print_interval = 10 # steps
		update_interval = 10 # steps
		summary_interval = 200 # steps
		validate_interval = 200 # steps
		checkpoint_interval = 1000 # steps

		print('Start iterating...')

		start_time = time.time()

		batch_grads = None

		while True:

			# compute epochs

			epochs = 1.0*(self.step+1)*self.config.batch_size/num_train_patterns
			do_print = ((self.step+1) % print_interval == 0)
			do_update = ((self.step+1) % update_interval == 0)
			do_summary = ((self.step+1) % summary_interval == 0)
			do_validate = ((self.step+1) % validate_interval == 0)
			do_checkpoint = ((self.step+1) % checkpoint_interval == 0)

			# training networks

			step_grads,_,step_losses = sess.run([self.grad_list, self.bn_op, self.losses])
			step_grads = [np.nan_to_num(grad) for grad in step_grads] # handle nan
			batch_grads = self.cumulate_gradients(batch_grads, step_grads)
			step_losses = step_losses

			# update gradients

			if do_update:
				grad_dict = {}
				for k in range(len(self.grad_placeholder)):
					grad_dict[self.grad_placeholder[k][0]] = batch_grads[k] / update_interval
				sess.run(self.update_op, feed_dict=grad_dict)
				batch_grads = None

			# validation

			if do_validate:
				self.validate_loss(sess, num_valid_patterns)

			# log

			if do_summary:
				summary_str = sess.run(self.train_summary_op)
				self.summarizer.add_summary(summary_str, self.step)

			if do_checkpoint:
				self.saver.save(sess, os.path.join(self.config.train_dir,'model.ckpt'), global_step=self.step+1)
			
			if do_print:
				now_time = time.time()
				batch_duration = now_time - start_time
				start_time = now_time
				log_str = 'Step %7d: %5.1f sec, epoch: %7.2f, loss: %7.3g %7.3g %7.3g %7.3g %7.3g\n' \
					% (self.step+1, batch_duration, epochs, step_losses[0], step_losses[1], step_losses[2], step_losses[3], step_losses[4])
				print(log_str)
				log_file_name = os.path.join(self.config.train_dir, 'log.txt')
				with open(log_file_name, 'a') as log_file:
					log_file.write(log_str)

			if epochs >= self.config.max_epochs:
				break

			self.step += 1

		coord.request_stop()
		coord.join(threads)

	def test(self, sess, num_patterns):

		print('Testing...')

		self.saver = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state(self.config.train_dir)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(sess, ckpt.model_checkpoint_path)
			self.step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
		else:
			print('Cannot find any checkpoint file')
			return

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		self.summarizer = tf.summary.FileWriter(self.config.test_dir, sess.graph)

		output_features_folder = 'features'
		output_visuals_folder = 'visuals'
		output_count = 0
		finished = False
		while not finished:
			if self.config.visualize_feature:
				names, visuals = sess.run([self.names, self.visual_features])
			else:
				names, features = sess.run([self.names, self.image_features])
			for k in range(len(names)):

				image_name = names[k].decode('utf8')
				print('Processed %d: %s' % (output_count, image_name))

				if self.config.visualize_feature:
					# visualize feature maps
					visual_path = os.path.join(self.config.test_dir, output_visuals_folder, image_name)
					if not os.path.exists(visual_path):
						os.makedirs(visual_path)
					for j in range(self.config.feature_size):
						offset = k*self.config.feature_size+j
						visual_name = os.path.join(visual_path, str(j)+'.png')
						image.write_image(visual_name, visuals[offset])
				else:
					# extract element feature
					element_feature = data.extract_element_feature(features[k], image_name, self.config.data_dir)

					# export results
					name_output = os.path.join(self.config.test_dir, output_features_folder, image_name+'.txt')
					# data.write_bin_data(name_output, element_feature)
					np.savetxt(name_output, element_feature)

				# check termination
				output_count += 1
				if output_count >= num_patterns:
					finished = True
					break

		coord.request_stop()
		coord.join(threads)

	def validate_loss(self, sess, num_patterns):

		num_processed_patterns = 0
		cum_losses = None
		while num_processed_patterns < num_patterns:
			losses = sess.run(self.losses)
			cum_losses = losses if cum_losses is None else cum_losses+losses
			num_processed_patterns += self.config.batch_size
		cum_losses /= num_processed_patterns

		print('===== validation loss: %7.3g %7.3g %7.3g %7.3g %7.3g' \
			% (cum_losses[0], cum_losses[1], cum_losses[2], cum_losses[3], cum_losses[4]))

		summary_str = sess.run(self.valid_summary_op, feed_dict={self.valid_summary_loss:cum_losses})
		self.summarizer.add_summary(summary_str, self.step)

	def cumulate_gradients(self, cum_grads, grads):
		if cum_grads is None:
			cum_grads = grads
		else:
			for k in range(len(grads)):
				cum_grads[k] += grads[k]
		return cum_grads
