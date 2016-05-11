#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import traceback
import cPickle as pickle

import sys
from copy import *
import argparse
import time
import numpy
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from chainer import cuda
from chainer import Variable
from chainer import FunctionSet
from chainer import optimizers
import chainer.computational_graph as c
import chainer.functions as F

import click


class DA(object):
	def __init__(
		self,
		rng,
		data,
		n_inputs=784,
		n_hidden=784,
		corruption_level=0.3,
		optimizer=optimizers.AdaDelta,
		gpu=-1
	):
		"""
		Denoising AutoEncoder
		data: data for train
		n_inputs: a number of units of input layer and output layer
		n_hidden: a number of units of hidden layer
		corruption_level: a ratio of masking noise
		"""

		self.model = FunctionSet(
			encoder=F.Linear(n_inputs, n_hidden),
			decoder=F.Linear(n_hidden, n_inputs)
		)

		if gpu >= 0:
			self.model.to_gpu()

		self.gpu = gpu

		self.x_train, self.x_test = data

		self.n_train = len(self.x_train)
		self.n_test = len(self.x_test)

		self.n_inputs = n_inputs
		self.n_hidden = n_hidden

		self.optimizer = optimizer()
		self.optimizer.setup(self.model)
		self.corruption_level = corruption_level
		self.rng = rng

		self.train_losses = []
		self.test_losses = []

	@property
	def xp(self):
		return cuda.cupy if self.gpu >= 0 else numpy

	def forward(self, x_data, train=True):
		y_data = x_data
		# add noise (masking noise)
		x_data = self.get_corrupted_inputs(x_data, train=train)

		x, t = Variable(x_data), Variable(y_data)
		# encode
		h = self.encode(x)
		# decode
		y = self.decode(h)
		# compute loss
		loss = F.mean_squared_error(y, t)
		return loss

	def compute_hidden(self, x_data):
		# x_data = self.xp.asarray(x_data)
		x = Variable(x_data)
		h = self.encode(x)
		# return cuda.to_cpu(h.data)
		return h.data

	def predict(self, x_data):
		x = Variable(x_data)
		# encode
		h = self.encode(x)
		# decode
		y = self.decode(h)
		return cuda.to_cpu(y.data)

	def encode(self, x):
		return F.relu(self.model.encoder(x))

	def decode(self, h):
		return F.relu(self.model.decoder(h))

	def encoder(self):
		initialW = self.model.encoder.W
		initial_bias = self.model.encoder.b

		return F.Linear(self.n_inputs,
						self.n_hidden,
						initialW=initialW,
						initial_bias=initial_bias)

	def decoder(self):
		return self.model.decoder

	def to_cpu(self):
		self.model.to_cpu()
		self.xp = np

	def to_gpu(self):
		if self.gpu < 0:
			logging.error("something wrong")
			raise
		self.model.to_gpu()
		self.xp = cuda.cupy

	# masking noise
	def get_corrupted_inputs(self, x_data, train=True):
		if train and self.corruption_level != 0.0:
			mask = self.rng.binomial(size=x_data.shape, n=1, p=1.0 - self.corruption_level)
			mask = mask.astype(numpy.float32)
			mask = self.xp.asarray(mask)
			ret = mask * x_data
			# return self.xp.asarray(ret.astype(numpy.float32))
			return ret
		else:
			return x_data


	def train_and_test(self, n_epoch=5, batchsize=100):
		for epoch in xrange(1, n_epoch+1):
			logging.info('epoch: {}'.format(epoch))

			perm = self.rng.permutation(self.n_train)
			sum_loss = 0
			for i in xrange(0, self.n_train, batchsize):
				x_batch = self.xp.asarray(self.x_train[perm[i:i+batchsize]])

				real_batchsize = len(x_batch)

				self.optimizer.zero_grads()
				loss = self.forward(x_batch)
				loss.backward()
				self.optimizer.update()

				sum_loss += float(loss.data) * real_batchsize

			logging.info(
				'train mean loss={}'.format(sum_loss / self.n_train)
			)

			# evaluation
			sum_loss = 0
			for i in xrange(0, self.n_test, batchsize):
				x_batch = self.xp.asarray(self.x_test[i:i+batchsize])

				real_batchsize = len(x_batch)

				loss = self.forward(x_batch, train=False)

				sum_loss += float(loss.data) * real_batchsize

			logging.info(
				'test mean loss={}'.format(sum_loss / self.n_test)
			)


@click.command('exec denoising autoencoder learning')
@click.option('--description', '-d', default='MNIST original')
@click.option(
	'--gpu', '-g',
	type=int,
	default=-1,
	help='GPU ID (negative value indicates CPU)'
)
@click.option(
	'--output', '-o',
	default='mlp.pkl',
	help='output filepath to store trained mlp object'
)
def main(description, gpu, output):
	logging.basicConfig(level=logging.INFO)

	logging.info('fetch MNIST dataset')
	mnist = fetch_mldata(description)
	mnist.data = mnist.data.astype(numpy.float32)
	mnist.data /= 255
	mnist.target = mnist.target.astype(numpy.int32)

	data_train, data_test, target_train, target_test = train_test_split(mnist.data, mnist.target)

	data = data_train, data_test
	target = target_train, target_test

	if gpu >= 0:
		cuda.check_cuda_available()
		cuda.get_device(gpu).use()

	# draw_digits(mnist.data[0:9])
	rng = numpy.random.RandomState(1)
	start_time = time.time()

	da = DA(rng=rng, data=data, gpu=gpu)

	perm = numpy.random.permutation(len(data[0]))
	data = mnist.data[perm[0:9]]

	da.train_and_test(n_epoch=5)

	end_time = time.time()

	logging.info("time = {} min".format((end_time - start_time) / 60.0))

	logging.info("saving trained da into {}".format(output))
	with open(output, 'wb') as fp:
		pickle.dump(da, fp)


if __name__ == '__main__': main()
