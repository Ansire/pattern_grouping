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

import os
import random
import numpy as np
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.io as sio

# data_dir = './../../../data/synthetic/'
data_dir = '/media/hdd/Projects/Pattern/Pattern_Data/Data_No_Occlution/'
# img = sio.loadmat(os.path.join(data_dir, '1_4.mat'))
# img = img['labelingMatrix']
# plt.imshow(img)

num_triplets = 1000

label_dir = os.path.join(data_dir, 'label')
triplet_dir = os.path.join(data_dir, 'triplet-region')
if not os.path.exists(triplet_dir):
	os.makedirs(triplet_dir)

list_file = open(os.path.join(data_dir, 'list.txt'), 'r')
names = list_file.read().splitlines()
list_file.close()
# print(len(names))

random.seed(1403)

for name in names:
	print(name)
	# label = misc.imread(os.path.join(label_dir, name+'.png'))
	label = sio.loadmat(os.path.join(label_dir, name + '.mat'))
	label = label['labelingMatrix']
	# num_labels = np.amax(label)//2 # first half for region, second half for edge
	num_labels = np.amax(label)
	coords_list = []
	for label_id in range(num_labels):
		# edge_label_id = label_id + num_labels + 1
		# coords_list.append(np.transpose(np.nonzero(label==edge_label_id)))
		region_label_id = label_id+1
		coords_list.append(np.transpose(np.nonzero(label==region_label_id)))
		# plt.imshow(label==region_label_id)
		# plt.show()
	triplets = np.zeros((num_triplets, 3, 2), dtype=np.int16)
	for triplet_id in range(num_triplets):
		while True:
			label_AB = random.randrange(num_labels)
			num_pixels_AB = coords_list[label_AB].shape[0]
			if num_pixels_AB>0:
				break
		while True:
			label_C = random.randrange(num_labels)
			num_pixels_C = coords_list[label_C].shape[0]
			if num_pixels_C>0:
				break
		coord_A = coords_list[label_AB][random.randrange(num_pixels_AB)]
		coord_B = coords_list[label_AB][random.randrange(num_pixels_AB)]
		coord_C = coords_list[label_C][random.randrange(num_pixels_C)]
		coord_ABC = np.array([coord_A, coord_B, coord_C]).astype(np.int16)
		triplets[triplet_id] = coord_ABC
		# input('pause')
		# box_size = 20
		# fig, ax = plt.subplots(1)
		# ax.imshow(label>num_labels)
		# rect_A = patches.Rectangle(np.array([coord_A[1]-box_size,coord_A[0]-box_size]), box_size*2, box_size*2, linewidth=1, edgecolor='r', facecolor='none')
		# rect_B = patches.Rectangle(np.array([coord_B[1]-box_size,coord_B[0]-box_size]), box_size*2, box_size*2, linewidth=1, edgecolor='g', facecolor='none')
		# rect_C = patches.Rectangle(np.array([coord_C[1]-box_size,coord_C[0]-box_size]), box_size*2, box_size*2, linewidth=1, edgecolor='b', facecolor='none')
		# ax.add_patch(rect_A)
		# ax.add_patch(rect_B)
		# ax.add_patch(rect_C)
		# plt.show()
	triplets.tofile(os.path.join(triplet_dir, name+'.bin'))
	# print(triplets)
	# input('pause')