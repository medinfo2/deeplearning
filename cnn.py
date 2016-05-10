# -*- coding: utf-8 -*-

import argparse
import time
import cPickle as pickle
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F

# import sys
# sys.path.append

class CNNModel(FunctionSet):
	def __init__(self, in_channels=1, n_hidden=100, n_outputs=10):
		super(CNNModel, self).__init__(
			conv1=	F.Convolution2D(in_channels, 32, 5),
			conv2=	F.Convolution2D(32, 32, 5),
			l3=		F.Linear(288, n_hidden),
			l4=		F.Linear(n_hidden, n_outputs)
		)

	def forward(self, x_data, y_data, train=True, gpu=-1):
		x, t = Variable(x_data), Variable(y_data)
		h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2)
		h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=3, stride=3)
		h = F.dropout(F.relu(self.l3(h)), train=train)
		y = self.l4(h)
		return F.softmax_cross_entropy(y, t), F.accuracy(y,t)

	def predict(self, x_data, gpu=-1):
		x = Variable(x_data)
		h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2)
		h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=3, stride=3)
		h = F.dropout(F.relu(self.l3(h)), train=train)
		y = self.l4(h)
		sftmx = F.softmax(y)
		out_data = cuda.to_cpu(sftmx.data)
		return out_data
	

class CNN:
	def __init__(self, data, target, in_channels=1,
									 n_hidden=100,
									 n_outputs=10,
									 gpu=-1):

		self.model = CNNModel(in_channels, n_hidden, n_outputs)
		self.model_name = 'cnn.model'

		if gpu >= 0:
			self.model.to_gpu()
			self.xp = cuda.cupy
		else:
			self.xp = np

		self.gpu = gpu

		self.x_train, self.x_test = data
		self.y_train, self.y_test = target

		self.n_train = len(self.y_train)
		self.n_test = len(self.y_test)

		self.optimizer = optimizers.Adam()
		self.optimizer.setup(self.model)


	def train_and_test(self, n_epoch=20, batchsize=100):
		epoch = 1
		while epoch <= n_epoch:
			print 'epoch', epoch

			perm = np.random.permutation(self.n_train)
			sum_train_accuracy = 0
			sum_train_loss = 0
			for i in xrange(0, self.n_train, batchsize):
				x_batch = self.xp.asarray(self.x_train[perm[i:i+batchsize]])
				y_batch = self.xp.asarray(self.y_train[perm[i:i+batchsize]])

				real_batchsize = len(x_batch)

				self.optimizer.zero_grads()
				loss, acc = self.model.forward(x_batch, y_batch, train=True, gpu=self.gpu)
				loss.backward()
				self.optimizer.update()

				sum_train_loss += float(loss.data) * real_batchsize
				sum_train_accuracy += float(acc.data) * real_batchsize

			print 'train mean loss={}, accuracy={}'.format(sum_train_loss/self.n_train, sum_train_accuracy/self.n_train)

			# evalation
			sum_test_accuracy = 0
			sum_test_loss = 0
			for i in xrange(0, self.n_test, batchsize):
				x_batch = self.xp.asarray(self.x_test[i:i+batchsize])
				y_batch = self.xp.asarray(self.y_test[i:i+batchsize])

				real_batchsize = len(x_batch)

				loss, acc = self.model.forward(x_batch, y_batch, train=False, gpu=self.gpu)

				sum_test_loss += float(loss.data) * real_batchsize
				sum_test_accuracy += float(acc.data) * real_batchsize

			print 'test mean loss={}, accuracy={}'.format(sum_test_loss/self.n_test, sum_test_accuracy/self.n_test)

			epoch += 1

	def dump_model(self):
		self.model.to_cpu()
		pickle.dump(self.model, open(self.model_name, 'wb'), -1)

	def load_model(self):
		self.model = pickle.load(open(self.model_name,'rb'))
		if self.gpu >= 0:
			self.model.to_gpu()
		self.optimizer.setup(self.model)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='MNIST')
	parser.add_argument('--gpu', '-g', default=-1, type=int,
						help='GPU ID (negative value indicates CPU)')
	args = parser.parse_args()

	if args.gpu >= 0:
		cuda.check_cuda_available()
		cuda.get_device(args.gpu).use()


	print 'fetch MNIST dataset'
	mnist = fetch_mldata('MNIST original')
	mnist.data   = mnist.data.astype(np.float32)
	mnist.data  /= 255
	mnist.data = mnist.data.reshape(70000,1,28,28)
	mnist.target = mnist.target.astype(np.int32)

	data_train,\
	data_test,\
	target_train,\
	target_test = train_test_split(mnist.data, mnist.target)

	data = [data_train, data_test]
	target = [target_train, target_test]

	n_outputs = 10
	in_channels = 1

	# from animeface import AnimeFaceDataset
	# print 'load AnimeFace dataset'
	# dataset = AnimeFaceDataset()
	# dataset.read_data_target()
	# data = dataset.data
	# target = dataset.target
	# n_outputs = dataset.get_n_types_target()
	# in_channels = 3


	start_time = time.time()

	cnn = CNN(data=data,
			  target=target,
			  gpu=args.gpu,
			  in_channels=in_channels,
			  n_outputs=n_outputs,
			  n_hidden=100)
	cnn.train_and_test(n_epoch=10)
	cnn.dump_model()

	# cnn.load_model()

	# while True:
	# 	_input = raw_input()
	# 	_input = _input[0:len(_input)-1]
	# 	import cv2 as cv
	# 	image = cv.imread(_input)
	# 	image = cv.resize(image, (dataset.image_size, dataset.image_size))
	# 	image = image.transpose(2,0,1)
	# 	image = image/255.
	# 	tmp = []
	# 	tmp.append(image)
	# 	data = np.array(tmp, np.float32)
	# 	target = int(dataset.get_class_id(_input))
	# 	predicted = cnn.predict(data)[0]
	# 	rank = {}
	# 	for i in xrange(len(predicted)):
	# 		rank[dataset.index2name[i]] = predicted[i]
	# 	rank = sorted(rank.items(),key=lambda x:x[1],reverse=True)
	# 	for i in range(9):
	# 		r = rank[i]
	# 		print "#" + str(i+1) + '  ' + r[0] + '  ' + str(r[1]*100) + '%'
	# 	print '#########################################'




	end_time = time.time()

	print "time = {} min".format((end_time-start_time)/60.0)









