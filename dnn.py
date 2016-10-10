#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import time
import json
import cPickle as pickle
import logging
import traceback

import numpy
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
import chainer
from chainer import cuda
from chainer import Variable
from chainer import Chain
import chainer.functions as F

import yaml
from dbarchive import Base


class Argument(object):
	layer_num_expr = re.compile('\w+(?P<layer_num>\d+)')
	def __init__(self, name, chain):
		self.layer_num = self.layer_num_expr.match(name).group('layer_num')
		self.chain = chain

	@classmethod
	def get_link_name(self, layer_num):
		for child in self.chain._children:
			if re.match(child, '\w+{}'.format(self.layer_num)):
				return child
		return None


class Convolution2DArgument(Argument):
	def __init__(self, name, chain, *args, **kwargs):
		Argument.__init__(self, name, chain)
		self.args = args
		self.kwargs = kwargs

	def channel_pass(self):
		try:
			previous_layer
		except:
			pass



class Network(Base):
	_layer_expr = re.compile('\w+(?P<layer_num>\d+)')
	_function_shortcuts = {
		'conv': 'chainer.functions.Convolution2D',
		'Convolution2D': 'chainer.functions.Convolution2D',
		'Linear': 'chainer.functions.Linear',
		'linear': 'chainer.functions.Linear',
		'max_pooling': 'chainer.functions.max_pooling_2d',
		'max_pooling_2d': 'chainer.functions.max_pooling_2d'
	}
	_argument_shortcuts = {
		'chainer.functions.': ""
	}

	def __init__(self):
		self.excludes.append('xp')
		self.model_name = 'anonymous neural network'
		self.expression = None
		self.gpu = -1
		self.x_train, self.x_test = None, None
		self.y_train, self.y_test = None, None
		self.n_train = -1
		self.n_test = -1
		self.model = Chain()
		self.ordered_layer_keys = []
		self.optimizer = None
		self.train_accuracies = []
		self.train_losses = []
		self.test_accuracies = []
		self.test_losses = []

	@classmethod
	def build(cls, *args, **kwargs):
		try:
			if 'expression' in kwargs:
				return cls.build_from_dict(*args, **kwargs)
			elif 'path' in kwargs:
				base, ext = os.path.splitext(kwargs['path'])
				if ext == '.json':
					return cls.build_from_json(*args, **kwargs)
				elif ext == '.yaml':
					return cls.build_from_yaml(*args, **kwargs)
				else:
					raise Exception('invalid file format: {}'.format(ext))
			else:
				raise Exception('invalid parameters: {}, {}'.format(args, kwargs))
		except:
			logging.error(traceback.format_exc())
			return None

	@classmethod
	def build_from_yaml(cls, data, target, path):
		try:
			with open(os.path.abspath(path), 'r') as fp:
				expression = yaml.load(fp)
			return cls.build_from_dict(data, target, expression)
		except:
			logging.error(traceback.format_exc())
			return None

	@classmethod
	def build_from_json(cls, data, target, path):
		try:
			with open(os.path.abspath(path), 'r') as fp:
				expression = json.load(fp)
			return cls.build_from_dict(data, target, expression)
		except:
			logging.error(traceback.format_exc())

	@classmethod
	def build_from_dict(cls, data, target, expression):
		try:
			net = Network()
			net.expression = expression
			net.x_train, net.x_test = data
			net.y_train, net.y_test = target
			net.n_train = len(net.y_train)
			net.n_test = len(net.y_test)
			net.gpu = expression['gpu']
			layers = {}
			for k, v in expression['layers'].items():
				if 'model' in v:
					net.expression['layers'][k]['model'] = v['model'] \
						if not v['model'] in cls._function_shortcuts \
						else cls._function_shortcuts[v['model']]
				elif 'function' in v:
					net.expression['layers'][k]['function'] = v['function'] \
						if not v['function'] in cls._function_shortcuts \
						else cls._function_shortcuts[v['function']]
				layers[k] = net.expression['layers'][k]
			net.ordered_layer_keys = [None for l in layers.keys()]
			for key, value in layers.items():
				layer_num = int(Network._layer_expr.search(key).group('layer_num'))
				net.ordered_layer_keys[layer_num - 1] = key
				layer = value
				if 'model' in layer:
					net.model.add_link(key, eval(layer['model'])(**layer['kwargs']))
			net.optimizer = eval(expression['optimizer'])()
			net.optimizer.setup(net.model)
			return net
		except:
			logging.error(traceback.format_exc())
			return None

	@property
	def xp(self):
		return cuda.cupy if self.gpu >= 0 else numpy

	def forward(self, x_data, train=True, gpu=-1):
		h = x_data
		for layer_key in self.ordered_layer_keys:
			layer = self.expression['layers'][layer_key]
			if 'model' in layer:
				model = self.model.__getattribute__(layer_key)
				h = model(h)
			elif 'function' in layer:
				h = eval(layer['function'])(h, **layer['kwargs'])
			else:
				raise Exception('invalid layer type {}.'.format(layer))
		return h

	def forward_and_eval(self, x_data, y_data, train=True, gpu=-1):
		x, t = Variable(x_data), Variable(y_data)
		y = self.forward(x, gpu=gpu)
		return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

	def predict(self, x_data, gpu=-1):
		x = Variable(x_data)
		y = self.forward(x)
		sftmx = F.softmax(y)
		out_data = cuda.to_cpu(sftmx.data)
		return out_data

	def train_and_test(self, n_epoch=20, batchsize=100):
		epoch = 1
		while epoch <= n_epoch:
			logging.info('epoch {}'.format(epoch))

			perm = numpy.random.permutation(self.n_train)
			sum_train_accuracy = 0
			sum_train_loss = 0
			for i in xrange(0, self.n_train, batchsize):
				x_batch = self.xp.asarray(self.x_train[perm[i:i+batchsize]])
				y_batch = self.xp.asarray(self.y_train[perm[i:i+batchsize]])

				real_batchsize = len(x_batch)

				self.optimizer.zero_grads()
				loss, acc = self.forward_and_eval(x_batch, y_batch, train=True, gpu=self.gpu)
				loss.backward()
				self.optimizer.update()

				sum_train_loss += float(loss.data) * real_batchsize
				sum_train_accuracy += float(acc.data) * real_batchsize

			logging.info(
				'train mean loss={}, accuracy={}'.format(
					sum_train_loss / self.n_train,
					sum_train_accuracy / self.n_train
				)
			)
			self.train_accuracies.append(sum_train_accuracy / self.n_train)
			self.train_losses.append(sum_train_loss / self.n_train)

			# evalation
			sum_test_accuracy = 0
			sum_test_loss = 0
			for i in xrange(0, self.n_test, batchsize):
				x_batch = self.xp.asarray(self.x_test[i:i+batchsize])
				y_batch = self.xp.asarray(self.y_test[i:i+batchsize])

				real_batchsize = len(x_batch)

				loss, acc = self.forward_and_eval(x_batch, y_batch, train=False, gpu=self.gpu)

				sum_test_loss += float(loss.data) * real_batchsize
				sum_test_accuracy += float(acc.data) * real_batchsize

			logging.info(
				'test mean loss={}, accuracy={}'.format(
					sum_test_loss / self.n_test,
					sum_test_accuracy / self.n_test
				)
			)
			self.test_accuracies.append(sum_test_accuracy / self.n_test)
			self.test_losses.append(sum_test_loss / self.n_test)

			epoch += 1

####################################################################################################
## SCRIPT EXECUTION CODE
####################################################################################################

import click

@click.command('exec denoising autoencoder learning')
@click.option(
	'--gpu', '-g',
	type=int,
	default=-1,
	help='GPU ID (negative value indicates CPU)'
)
@click.option(
	'--epochs', '-e',
	type=int,
	default=1,
	help='training epochs'
)
@click.option(
	'--json', '-j',
	default=None,
	help='json expression of the network'
)
@click.option(
	'--yaml', '-y',
	default=None,
	help='yaml expression of the network'
)
def main(gpu, epochs, json, yaml):
	cnn_expr = {
		'name': 'CNN',
		'optimizer': 'chainer.optimizers.Adam',
		'gpu': -1,
		'layers': {
			'conv1': {
				'model': 'chainer.functions.Convolution2D',
				'kwargs': {
					'in_channels': 1,
					'out_channels': 32,
					'ksize': 2
				}
			},
			'pool2': {
				'function': 'chainer.functions.max_pooling_2d',
				'kwargs': {
					'ksize': 2,
					'stride': 2
				}
			},
			'conv3': {
				'model': 'chainer.functions.Convolution2D',
				'kwargs': {
					'in_channels': 32,
					'out_channels': 32,
					'ksize': 2,
				}
			},
			'pool4': {
				'function': 'chainer.functions.max_pooling_2d',
				'kwargs': {
					'ksize': 2
				}
			},
			'l5': {
				'model': 'chainer.functions.Linear',
				'kwargs': {
					'in_size': 1568,
					'out_size': 100
				}
			},
			'l6': {
				'model': 'chainer.functions.Linear',
				'kwargs': {
					'in_size': 100,
					'out_size': 100
				}
			},
		}
	}

	logging.basicConfig(level=logging.INFO)

	if gpu >= 0:
		cuda.check_cuda_available()
		cuda.get_device(gpu).use()


	logging.info('fetch MNIST dataset')
	mnist = fetch_mldata('MNIST original')
	mnist.data = mnist.data.astype(numpy.float32)
	mnist.data /= 255
	mnist.data = mnist.data.reshape(70000, 1, 28, 28)
	mnist.target = mnist.target.astype(numpy.int32)

	data_train, data_test, target_train, target_test = train_test_split(mnist.data, mnist.target)

	data = data_train[:10000], data_test[:10000]
	target = target_train[:10000], target_test[:10000]

	n_outputs = 10
	in_channels = 1

	start_time = time.time()


	if json:
		logging.info('test for build_from_json ...')
		cnn = Network.build(data=data, target=target, path=json)
	elif yaml:
		logging.info('test for build_from_yaml ...')
		cnn = Network.build(data=data, target=target, path=yaml)
	else:
		logging.info('test for build_from_dict ...')
		cnn = Network.build(data=data, target=target, expression=cnn_expr)

	cnn.train_and_test(n_epoch=epochs)

	end_time = time.time()

	logging.info("time = {} min".format((end_time - start_time) / 60.0))

	logging.info('saving trained cnn')
	cnn.save()


if __name__ == '__main__': main()
