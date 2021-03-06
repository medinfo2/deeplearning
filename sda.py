#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import logging
import traceback
import cPickle as pickle

import numpy
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from chainer import cuda
from chainer import Variable
from chainer import FunctionSet
from chainer import optimizers
import chainer.functions as F

from da import DA

import click


class SDA(object):
	def __init__(
		self,
		rng,
		data,
		target,
		n_inputs=784,
		n_hidden=[784,784,784],
		n_outputs=10,
		corruption_levels=[0.1,0.2,0.3],
		gpu=-1):

		self.model = FunctionSet(
			l1=F.Linear(n_inputs, n_hidden[0]),
			l2=F.Linear(n_hidden[0], n_hidden[1]),
			l3=F.Linear(n_hidden[1], n_hidden[2]),
			l4=F.Linear(n_hidden[2], n_outputs)
		)

		if gpu >= 0:
			self.model.to_gpu()

		self.rng = rng
		self.gpu = gpu
		self.data = data
		self.target = target

		self.x_train, self.x_test = data
		self.y_train, self.y_test = target

		self.n_train = len(self.y_train)
		self.n_test = len(self.y_test)

		self.corruption_levels = corruption_levels
		self.n_inputs = n_inputs
		self.n_hidden = n_hidden
		self.n_outputs = n_outputs

		self.dae1 = None
		self.dae2 = None
		self.dae3 = None
		self.optimizer = None
		self.setup_optimizer()

		self.train_accuracies = []
		self.train_losses = []

		self.test_accuracies = []
		self.test_losses = []

	def setup_optimizer(self):
		self.optimizer = optimizers.AdaDelta()
		self.optimizer.setup(self.model)

	@property
	def xp(self):
		return cuda.cupy if self.gpu >= 0 else numpy

	def pre_train(self, n_epoch=20, batchsize=100):
		first_inputs = self.data

		# initialize first dAE
		self.dae1 = DA(self.rng,
					   data=first_inputs,
					   n_inputs=self.n_inputs,
					   n_hidden=self.n_hidden[0],
					   corruption_level=self.corruption_levels[0],
					   gpu=self.gpu)
		# train first dAE
		logging.info("--------First DA training has started!--------")
		self.dae1.train_and_test(n_epoch=n_epoch, batchsize=batchsize)
		self.dae1.to_cpu()
		# compute second iputs for second dAE
		tmp1 = self.dae1.compute_hidden(first_inputs[0])
		tmp2 = self.dae1.compute_hidden(first_inputs[1])
		if self.gpu >= 0:
			self.dae1.to_gpu()
		second_inputs = [tmp1, tmp2]

		# initialize second dAE
		self.dae2 = DA(
			self.rng,
			data=second_inputs,
			n_inputs=self.n_hidden[0],
			n_hidden=self.n_hidden[1],
			corruption_level=self.corruption_levels[1],
			gpu=self.gpu
		)
		# train second dAE
		logging.info("--------Second DA training has started!--------")
		self.dae2.train_and_test(n_epoch=n_epoch, batchsize=batchsize)
		self.dae2.to_cpu()
		# compute third inputs for third dAE
		tmp1 = self.dae2.compute_hidden(second_inputs[0])
		tmp2 = self.dae2.compute_hidden(second_inputs[1])
		if self.gpu >= 0:
			self.dae2.to_gpu()
		third_inputs = [tmp1, tmp2]

		# initialize third dAE
		self.dae3 = DA(
			self.rng,
			data=third_inputs,
			n_inputs=self.n_hidden[1],
			n_hidden=self.n_hidden[2],
			corruption_level=self.corruption_levels[2],
			gpu=self.gpu
		)
		# train third dAE
		logging.info("--------Third DA training has started!--------")
		self.dae3.train_and_test(n_epoch=n_epoch, batchsize=batchsize)

		# update model parameters
		self.model.l1 = self.dae1.encoder()
		self.model.l2 = self.dae2.encoder()
		self.model.l3 = self.dae3.encoder()

		self.setup_optimizer()

	def forward(self, x_data, y_data, train=True):
		x, t = Variable(x_data), Variable(y_data)
		h1 = F.dropout(F.relu(self.model.l1(x)), train=train)
		h2 = F.dropout(F.relu(self.model.l2(h1)), train=train)
		h3 = F.dropout(F.relu(self.model.l3(h2)), train=train)
		y = self.model.l4(h3)
		return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

	def fine_tune(self, n_epoch=20, batchsize=100):
		for epoch in xrange(1, n_epoch+1):
			logging.info('fine tuning epoch {}'.format(epoch))

			perm = self.rng.permutation(self.n_train)
			sum_accuracy = 0
			sum_loss = 0
			for i in xrange(0, self.n_train, batchsize):
				x_batch = self.xp.asarray(self.x_train[perm[i:i+batchsize]])
				y_batch = self.xp.asarray(self.y_train[perm[i:i+batchsize]])

				real_batchsize = len(x_batch)

				self.optimizer.zero_grads()
				loss, acc = self.forward(x_batch, y_batch)
				loss.backward()
				self.optimizer.update()

				sum_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
				sum_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

			logging.info(
				'fine tuning train mean loss={}, accuracy={}'.format(
					sum_loss / self.n_train,
					sum_accuracy / self.n_train
				)
			)
			self.train_accuracies.append(sum_accuracy / self.n_train)
			self.train_losses.append(sum_loss / self.n_train)

			# evaluation
			sum_accuracy = 0
			sum_loss = 0
			for i in xrange(0, self.n_test, batchsize):
				x_batch = self.xp.asarray(self.x_test[i:i+batchsize])
				y_batch = self.xp.asarray(self.y_test[i:i+batchsize])

				real_batchsize = len(x_batch)

				loss, acc = self.forward(x_batch, y_batch, train=False)

				sum_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
				sum_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

			logging.info(
				'fine tuning test mean loss={}, accuracy={}'.format(
					sum_loss / self.n_test,
					sum_accuracy / self.n_test
				)
			)
			self.test_accuracies.append(sum_accuracy / self.n_test)
			self.test_losses.append(sum_loss / self.n_test)

		return self.train_accuracies, self.test_accuracies


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
	default='sda.pkl',
	help='output filepath to store trained sda object'
)
def main(description, gpu, output):
	logging.basicConfig(level=logging.INFO)

	logging.info('fetch MNIST dataset')
	mnist = fetch_mldata(description)
	mnist.data   = mnist.data.astype(numpy.float32)
	mnist.data  /= 255
	mnist.target = mnist.target.astype(numpy.int32)

	data_train, data_test, target_train, target_test = train_test_split(mnist.data, mnist.target)

	data = data_train, data_test
	target = target_train, target_test

	rng = numpy.random.RandomState(1)

	if gpu >= 0:
		cuda.check_cuda_available()
		cuda.get_device(gpu).use()

	start_time = time.time()

	sda = SDA(
		rng=rng,
		data=data,
		target=target,
		gpu=gpu
	)
	# sda.pre_train(n_epoch=15)
	sda.fine_tune(n_epoch=5)

	end_time = time.time()

	logging.info("time = {} min".format((end_time - start_time) / 60.0))
	logging.info('saving trained sda into {}'.format(output))
	with open(output, 'wb') as fp:
		pickle.dump(sda, fp)


if __name__ == '__main__': main()
