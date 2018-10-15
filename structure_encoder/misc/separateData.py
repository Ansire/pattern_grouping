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
from random import shuffle

data_dir = './../../../data/synthetic/'
test_split = 0.1
valid_split = 0.02

fin = open(os.path.join(data_dir, 'list.txt'), 'r')
names = fin.read().splitlines()
fin.close()

shuffle(names)

n = len(names)
if 'test_size' not in locals():
	test_size = int(n * test_split)
if 'valid_size' not in locals():
	valid_size = int(n * valid_split)
if 'train_size' not in locals():
	train_size = n - test_size - valid_size

train_names = names[0:train_size]
valid_names = names[train_size:train_size+valid_size]
test_names = names[train_size+valid_size:]

fout = open(os.path.join(data_dir, 'random-list.txt'), 'w')
fout.write('\n'.join(names))
fout.write('\n')
fout.close()

fout = open(os.path.join(data_dir, 'train-list.txt'), 'w')
fout.write('\n'.join(train_names))
fout.write('\n')
fout.close()

fout = open(os.path.join(data_dir, 'validate-list.txt'), 'w')
fout.write('\n'.join(valid_names))
fout.write('\n')
fout.close()

fout = open(os.path.join(data_dir, 'test-list.txt'), 'w')
fout.write('\n'.join(test_names))
fout.write('\n')
fout.close()