#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from datetime import datetime
import logging
import traceback
import cPickle as pickle

import numpy
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from chainer import cuda
from chainer import Variable
from chainer import FunctionSet
from chainer import optimizers
import chainer.functions as F

import click


class MLP(object):
	def __init__(
		self,
		data,
		target,
		n_inputs=784,
		n_hidden=784,
		n_outputs=10,
		gpu=-1
	):

		self.model = FunctionSet(
			l1=F.Linear(n_inputs, n_hidden),
			l2=F.Linear(n_hidden, n_hidden),
			l3=F.Linear(n_hidden, n_outputs)
		)

		if gpu >= 0:
			self.model.to_gpu()

		self.x_train, self.x_test = data
		self.y_train, self.y_test = target

		self.n_train = len(self.y_train)
		self.n_test = len(self.y_test)

		self.gpu = gpu
		self.optimizer = optimizers.Adam()
		self.optimizer.setup(self.model)

		self.train_accuracies = []
		self.train_losses = []
		self.test_accuracies = []
		self.test_losses = []

	@property
	def xp(self):
		return cuda.cupy if self.gpu >= 0 else numpy

	def _forward(self, x_data):
		x = Variable(x_data)
		h1 = F.relu(self.model.l1(x))
		h2 = F.relu(self.model.l2(h1))
		return self.model.l3(h2)

	def forward(self, x_data, y_data, train=True):
		x, t = Variable(x_data), Variable(y_data)
		h1 = F.dropout(F.relu(self.model.l1(x)), train=train)
		h2 = F.dropout(F.relu(self.model.l2(h1)), train=train)
		y = self.model.l3(h2)
		return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

	def predict(self, x_data):
		y = self._forward(x_data)
		return numpy.argmax(y.data)

	def accuracy(self, x_data, y_data):
		t = Variable(y_data)
		y = self._forward(x_data)
		return F.accuracy(y, t).data

	def train_and_test(self, n_epoch=20, batchsize=100):
		for epoch in xrange(1, n_epoch + 1):
			logging.info('epoch {}'.format(epoch))

			perm = numpy.random.permutation(self.n_train)
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

			self.train_accuracies.append(sum_accuracy / self.n_train)
			self.train_losses.append(sum_loss / self.n_train)

			logging.info(
				'train mean loss={}, accuracy={}'.format(
					sum_loss / self.n_train,
					sum_accuracy / self.n_train
				)
			)

			# evalation
			sum_accuracy = 0
			sum_loss = 0
			for i in xrange(0, self.n_test, batchsize):
				x_batch = self.xp.asarray(self.x_test[i:i+batchsize])
				y_batch = self.xp.asarray(self.y_test[i:i+batchsize])

				real_batchsize = len(x_batch)

				loss, acc = self.forward(x_batch, y_batch, train=False)

				sum_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
				sum_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

			self.test_accuracies.append(sum_accuracy / self.n_test)
			self.test_accuracies.append(sum_loss / self.n_test)
			logging.info(
				'test mean loss={}, accuracy={}'.format(
					sum_loss / self.n_test,
					sum_accuracy / self.n_test
				)
			)


@click.command('exec multi layer perceptron learning')
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

	start_time = time.time()

	if gpu >= 0:
		cuda.check_cuda_available()
		cuda.get_device(gpu).use()
		logging.info("Using gpu device {}".format(gpu))
	else:
		logging.info("Not using gpu device")

	mlp = MLP(data=data, target=target, gpu=gpu)
	mlp.train_and_test(n_epoch=1)

	end_time = time.time()

	# logging.info("time = {} min".format((end_time - start_time) / 60.0))
	# logging.info('saving trained mlp into {}'.format(output))
	# with open(output, 'wb') as fp:
	# 	pickle.dump(mlp, fp)

	xsample = data_test[0]
	xsample.shape = 1, xsample.shape[0]
	print 'y_data:', target_test[0], 'predicted:', mlp.predict(xsample)
	print "accuracy of 0-100:", mlp.accuracy(data_test[:100], target_test[:100])

if __name__ == '__main__': main()
