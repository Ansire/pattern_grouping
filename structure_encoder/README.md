# Learning to Group Discrete Graphical Patterns

## Introduction

This archive contains Python code for the **structure encoder network** part (including both training and testing) of the Pattern Grouping project.

## Environment

- [Python 3.5](https://www.python.org/)
- [NumPy 1.12.1](http://www.numpy.org/)
- [SciPy 0.19.0](https://www.scipy.org/)
- [TensorFlow 1.0.1](https://www.tensorflow.org/)

## How to Run

### Training

- run the following script in code directory

> python main.py --train --data\_dir XXX --train\_dir XXX

- The data required for training must have:
	- `$data_dir$/region/*.png`
	- `$data_dir$/triplet-region/*.bin`
	- `$data_dir$/train-list.txt`
	- `$data_dir$/validate-list.txt`

### Testing

- run the following script in code directory

> python main.py --test --real\_data --data\_dir XXX --train\_dir XXX --test\_dir XXX

- The data required for testing must have:
	- `$data_dir$/region/*.png`
	- `$data_dir$/element/*.mat`
	- `$data_dir$/test-list.txt`
	- `$train_dir$/checkpoint`
	- `$train_dir$/model.ckpt-*`