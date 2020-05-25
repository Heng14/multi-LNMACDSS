"""
@author: Yi Cui
"""

import tensorflow as tf
import utils
import keras.backend as K
import numpy as np
import math
from keras.preprocessing.image import ImageDataGenerator
from resnet_101 import resnet101_model
from resnet_50 import resnet50_model
from test_net import testnet_model
from resnet18 import ResnetBuilder
from snet import snet

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range = 0.2,
    horizontal_flip=True)

class SurvivalModel:

	def __init__(self):
		'''
		Args
			model_builder: a function which returns user defined Keras model
		'''
		#self.model_builder = resnet101_model
		#self.model_builder = resnet50_model
		#self.model_builder = testnet_model
		#self.model_builder = ResnetBuilder.build_resnet_18
		self.model_builder = snet

	def fit(self, datasets_train, datasets_val, datasets_test, datasets_train_1, datasets_val_1, datasets_test_1, loss_func='hinge', epochs=500, lr=0.001, mode='merge', batch_size = 8):
		'''
		Train a deep survival model
		Args
			datasets_train:     training datasets, a list of (X, time, event) tuples
			datasets_val:       validation datasets, a list of (X, time, event) tuples
			loss_func:          loss function to approximate concordance index, {'hinge', 'log', 'cox'}
			epochs:             number of epochs to train
			lr:                 learning rate
			mode:               if mode=='merge', merge datasets before training
								if mode='decentralize', treat each dataset as a mini-batch
			batch_size:         only effective for 'merge' mode
		'''

		self.datasets_train = datasets_train
		self.datasets_val = datasets_val
		self.datasets_test = datasets_test
		self.datasets_train_1 = datasets_train_1
		self.datasets_val_1 = datasets_val_1
		self.datasets_test_1 = datasets_test_1
		self.loss_func = loss_func
		self.epochs = epochs
		self.lr = lr
		self.batch_size = batch_size

		## build a tensorflow graph to define loss function
		self.__build_graph()
		
		## train the model
		if mode=='merge':
		  self.__train_merge()
		elif mode=='decentralize':
		  self.__train_decentralize()



	def __build_graph(self):
		'''
		Build a tensorflow graph. Call this within self.fit()
		'''

		input_shape = self.datasets_train[0][0].shape[1:]
		#print (input_shape[0])
		#raise

		with tf.name_scope('input'):
			X = tf.placeholder(dtype=tf.float32, shape=(None, )+input_shape, name='X')
			time = tf.placeholder(dtype=tf.float32, shape=(None, ), name='time')
			event = tf.placeholder(dtype=tf.int16, shape=(None, ), name='event')

		with tf.name_scope('model'):
			#self.model = self.model_builder(input_shape[0], input_shape[0])
			#self.model = self.model_builder((11, 224, 224), 1)
			self.model = self.model_builder()

		with tf.name_scope('output'):
			score = tf.identity(self.model(X), name='score')

		with tf.name_scope('metric'):
			ci = self.__concordance_index(score, time, event)
			if self.loss_func=='hinge':
				loss = self.__hinge_loss(score, time, event)
			elif self.loss_func=='log':
				loss = self.__log_loss(score, time, event)
			elif self.loss_func=='cox':
				loss = self.__cox_loss(score, time, event)
			elif self.loss_func=='cox_yh':
				loss = self.__cox_loss_yh(score, time, event)

		with tf.name_scope('train'):

			num_epoch = tf.Variable(0, name='global_step', trainable=False)
			boundaries = [5, 25, 70]
			#learing_rates = [0.0001, 0.0005, 0.00001, 0.00005]
			learing_rates = [0.0001, 0.0001, 0.0001, 0.0001]
			learing_rate = tf.train.piecewise_constant(num_epoch, boundaries=boundaries, values=learing_rates)
			optimizer = tf.train.AdamOptimizer(learning_rate=learing_rate)

			#optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
			train_op = optimizer.minimize(loss, name='train_op')

		## save the tensors and ops so that we can use them later
		self.__X = X
		self.__time = time
		self.__event = event
		self.__score = score
		self.__ci = ci
		self.__loss = loss
		self.__train_op = train_op



	def __train_decentralize(self):
		'''
		Decentralized training mode. Each dataset is regarded as a mini-batch
		'''

		## start training
		self.__sess = tf.Session()
		self.__sess.run(tf.global_variables_initializer())
		for epoch in range(self.epochs):
			for X_batch, time_batch, event_batch in self.datasets_train:
				self.__sess.run(self.__train_op, feed_dict={self.__X: X_batch, self.__time: time_batch, self.__event: event_batch, K.learning_phase(): 1})
			if epoch%100==0:
				print('-'*20 + 'Epoch: {0}'.format(epoch) + '-'*20)
				self.__print_loss_ci()
				


	def __train_merge(self):
		'''
		Merge training datasets into a single dataset. Sample mini-batches from the merged dataset for training
		'''
		#weights_path = 'checkpoints/resnet101_weights_tf.h5'
		weights_path = 'checkpoints/resnet50_weights_tf_dim_ordering_tf_kernels.h5'


		## Merge training datasets
		X_train, time_train, event_train = utils.combine_datasets(self.datasets_train)
		X_train_1, time_train_1, event_train_1 = utils.combine_datasets(self.datasets_train_1)

		## get training datasets ___ by heng
		#X_train, time_train, event_train = utils.get_datasets(self.datasets_train)

		## To fetch mini-batches
		#next_batch, num_batches = utils.batch_factory(X_train, time_train, event_train, self.batch_size)

		## start training
		self.__sess = tf.Session()
		self.__sess_1 = tf.Session()
		K.set_session(self.__sess)
		K.set_session(self.__sess_1)
		self.__sess.run(tf.global_variables_initializer())
		self.__sess_1.run(tf.global_variables_initializer())
		#self.model.load_weights(weights_path, by_name=True)

		print (f'pre epoch train log:')
		self.__print_loss_ci_yh_avg(self.datasets_train, self.datasets_train_1)


		for epoch in range(self.epochs):

			#next_batch, num_batches = utils.batch_factory(X_train, time_train, event_train, self.batch_size)
			#for _ in range(num_batches):
			#	X_batch, time_batch, event_batch = next_batch()


			time_event = np.hstack((time_train[...,None], event_train[...,None]))			
			batches = 0
			for X_batch, Y_batch in datagen.flow(X_train, time_event, batch_size=self.batch_size):
				batches += 1
				if batches >= X_train.shape[0] // self.batch_size:
					break
				time_batch = Y_batch[:,0]
				event_batch = Y_batch[:,1]
			
				self.__sess.run(self.__train_op, feed_dict={self.__X: X_batch, self.__time: time_batch, self.__event: event_batch, K.learning_phase(): 1})
				#print (self.__sess.run([self.__train_op, self.__loss], feed_dict={self.__X: X_batch, self.__time: time_batch, self.__event: event_batch, K.learning_phase(): 1}))
			#print (self.__sess.run([self.__score, self.__loss, self.__ci], feed_dict={self.__X: X_batch, self.__time: time_batch, self.__event: event_batch, K.learning_phase(): 0}))

			time_event_1 = np.hstack((time_train_1[...,None], event_train_1[...,None]))
			batches = 0
			for X_batch_1, Y_batch_1 in datagen.flow(X_train_1, time_event_1, batch_size=self.batch_size):
				batches += 1
				if batches >= X_train_1.shape[0] // self.batch_size:
					break
				time_batch_1 = Y_batch_1[:,0]
				event_batch_1 = Y_batch_1[:,1]

				self.__sess_1.run(self.__train_op, feed_dict={self.__X: X_batch_1, self.__time: time_batch_1, self.__event: event_batch_1, K.learning_phase(): 1})


			print (f'epoch {epoch} train log:')
			self.__print_loss_ci_yh_avg(self.datasets_train, self.datasets_train_1)
			if epoch%2==0:
				print('-'*20 + 'Epoch: {0}'.format(epoch) + '-'*20)
				print (f'epoch {epoch} val log:')
				self.__print_loss_ci_yh_avg(self.datasets_val, self.datasets_val_1)
				print (f'epoch {epoch} test log:')
				self.__print_loss_ci_yh_avg(self.datasets_test, self.datasets_test_1)



	def predict(self, X_test):
		'''
		Args
			X: design matrix of shape (num_samples, ) + input_shape
		'''

		assert X_test.shape[1:]==self.datasets_train[0][0].shape[1:], 'Shapes of testing and training data must equal'
		
		return self.__sess.run(self.__score, feed_dict = {self.__X: X_test, K.learning_phase():0})



	def evaluate(self, X_test, time_test, event_test):
		'''
		Evaluate the loss and c-index of the model for the given test data
		'''
		assert X_test.shape[1:]==self.datasets_train[0][0].shape[1:], 'Shapes of testing and training data must equal'

		return self.__sess.run([self.__loss, self.__ci], feed_dict = {self.__X: X_test, self.__time: time_test, self.__event: event_test, K.learning_phase(): 1})



	def __concordance_index(self, score, time, event):
		'''
		Args
			score: 		predicted score, tf tensor of shape (None, )
			time:		true survival time, tf tensor of shape (None, )
			event:		event, tf tensor of shape (None, )
		'''

		## find index pairs (i,j) satisfying time[i]<time[j] and event[i]==1
		ix = tf.where(tf.logical_and(tf.expand_dims(time, axis=-1)<time, tf.expand_dims(tf.cast(event, tf.bool), axis=-1)), name='ix')

		## count how many score[i]<score[j]
		s1 = tf.gather(score, ix[:,0])
		s2 = tf.gather(score, ix[:,1])
		ci = tf.reduce_mean(tf.cast(s1<s2, tf.float32), name='c_index')

		return ci



	def __hinge_loss(self, score, time, event):
		'''
		Args
			score:	 	predicted score, tf tensor of shape (None, 1)
			time:		true survival time, tf tensor of shape (None, )
			event:		event, tf tensor of shape (None, )
		'''

		## find index pairs (i,j) satisfying time[i]<time[j] and event[i]==1
		ix = tf.where(tf.logical_and(tf.expand_dims(time, axis=-1)<time, tf.expand_dims(tf.cast(event, tf.bool), axis=-1)), name='ix')

		## if score[i]>score[j], incur hinge loss
		s1 = tf.gather(score, ix[:,0])
		s2 = tf.gather(score, ix[:,1])
		loss = tf.reduce_mean(tf.maximum(1+s1-s2, 0.0), name='loss')

		return loss



	def __log_loss(self, score, time, event):
		'''
		Args
			score: 	predicted survival time, tf tensor of shape (None, 1)
			time:		true survival time, tf tensor of shape (None, )
			event:		event, tf tensor of shape (None, )
		'''

		## find index pairs (i,j) satisfying time[i]<time[j] and event[i]==1
		ix = tf.where(tf.logical_and(tf.expand_dims(time, axis=-1)<time, tf.expand_dims(tf.cast(event, tf.bool), axis=-1)), name='ix')

		## if score[i]>score[j], incur log loss
		s1 = tf.gather(score, ix[:,0])
		s2 = tf.gather(score, ix[:,1])
		loss = tf.reduce_mean(tf.log(1+tf.exp(s1-s2)), name='loss')

		return loss



	def __cox_loss(self, score, time, event):
		'''
		Args
			score: 		predicted survival time, tf tensor of shape (None, 1)
			time:		true survival time, tf tensor of shape (None, )
			event:		event, tf tensor of shape (None, )
		Return
			loss:		partial likelihood of cox regression
		'''


		## cox regression computes the risk score, we want the opposite
		score = -score

		## find index i satisfying event[i]==1
		ix = tf.where(tf.cast(event, tf.bool)) # shape of ix is [None, 1]

		## sel_mat is a matrix where sel_mat[i,j]==1 where time[i]<=time[j]
		sel_mat = tf.cast(tf.gather(time, ix)<=time, tf.float32)

		## formula: \sum_i[s_i-\log(\sum_j{e^{s_j}})] where time[i]<=time[j] and event[i]==1
		p_lik = tf.gather(score, ix) - tf.log(tf.reduce_sum(sel_mat * tf.transpose(tf.exp(score)), axis=-1))
		#p_lik = tf.gather(score, ix) - tf.log(tf.reduce_sum(tf.transpose(tf.exp(score)), axis=-1))
		loss = -tf.reduce_mean(p_lik)

		return loss

	def __cox_loss_yh(self, score, time, event):
		'''
		Args
			score: 		predicted survival time, tf tensor of shape (None, 1)
			time:		true survival time, tf tensor of shape (None, )
			event:		event, tf tensor of shape (None, )
		Return
			loss:		partial likelihood of cox regression
		'''


		## cox regression computes the risk score, we want the opposite
		#score = -score
		score_max = tf.reduce_max(score)
		score_sub = tf.subtract(score,score_max)

		## find index i satisfying event[i]==1
		ix = tf.where(tf.cast(event, tf.bool)) # shape of ix is [None, 1]

		## sel_mat is a matrix where sel_mat[i,j]==1 where time[i]<=time[j]
		sel_mat = tf.cast(tf.gather(time, ix)<=time, tf.float32)

		## formula: \sum_i[s_i-\log(\sum_j{e^{s_j}})] where time[i]<=time[j] and event[i]==1
		p_lik = tf.gather(score_sub, ix) - tf.log(tf.reduce_sum(sel_mat * tf.transpose(tf.exp(score_sub)), axis=-1))
		#p_lik = tf.gather(score, ix) - tf.log(tf.reduce_sum(tf.transpose(tf.exp(score)), axis=-1))
		loss = -tf.reduce_mean(p_lik)

		return loss


	def __print_loss_ci(self):
		'''
		Helper function to print the losses and c-indices on training & validation datasets
		'''
		## losses and c-indices on traning
		loss_train = np.zeros(len(self.datasets_train))
		ci_train = np.zeros(len(self.datasets_train))
		for i, (X_batch, time_batch, event_batch) in enumerate(self.datasets_train):
			loss_train[i], ci_train[i] = self.evaluate(X_batch, time_batch, event_batch)

		## losses and c-indices on validation
		loss_val = np.zeros(len(self.datasets_val))
		ci_val = np.zeros(len(self.datasets_val))
		for i, (X_batch, time_batch, event_batch) in enumerate(self.datasets_val):
			loss_val[i], ci_val[i] = self.evaluate(X_batch, time_batch, event_batch)

		## print them
		print('loss_train={0}'.format(np.round(loss_train, 2)))
		print('loss_val={0}'.format(np.round(loss_val, 2)))
		print('ci_train={0}'.format(np.round(ci_train, 2)))
		print('ci_val={0}'.format(np.round(ci_val, 2)))
		print()


	def __print_loss_ci_yh_old(self, datasets):
		'''
		This loss and ci compute each batch and then mean. It may be wrong.
		'''
		## losses and c-indices on traning
		#loss_train = np.zeros(len(self.datasets_train))
		#ci_train = np.zeros(len(self.datasets_train))

		X, time, event = zip(*datasets)
		X = np.concatenate(X, axis=0)
		time = np.concatenate(time, axis=0)
		event = np.concatenate(event, axis=0)
		assert len(X) == len(time) == len(event), print ('X, time, event len are not equal!!!')

		loss_sum = 0
		ci_sum = 0
		count = 0
		bs = 16

		for i in range(len(X)//bs):
			X_batch, time_batch, event_batch = X[bs*i:bs*(i+1),...], time[bs*i:bs*(i+1),...], event[bs*i:bs*(i+1),...]

			loss_i, ci_i = self.evaluate(X_batch, time_batch, event_batch)
			#print ('loss_i, ci_i: ',loss_i, ci_i)

			if math.isnan(loss_i) or math.isnan(ci_i) or math.isinf(loss_i) or math.isinf(ci_i):
				print ('here')
				continue
			loss_sum += loss_i*bs
			ci_sum += ci_i*bs
			count += bs             

			#print (loss_i, ci_i, math.isnan(loss_i), math.isnan(ci_i))
			#print (X_batch.shape, time_batch.shape, event_batch.shape)
		#print (loss_sum, ci_sum, count)

		i += 1
		X_batch, time_batch, event_batch = X[bs*i:len(X),...], time[bs*i:len(time),...], event[bs*i:len(event),...]
		loss_i, ci_i = self.evaluate(X_batch, time_batch, event_batch)
		
		if not (math.isnan(loss_i) or math.isnan(ci_i) or math.isinf(loss_i) or math.isinf(ci_i)):
			loss_sum += loss_i*(len(X)-bs*i)
			ci_sum += ci_i*(len(X)-bs*i)	
			count += (len(X)-bs*i)


		## print them
		print('loss={0}'.format(round(loss_sum/count, 3)))
		print('ci={0}'.format(round(ci_sum/count, 3)))
		print()

	def __print_loss_ci_yh(self, datasets):
		'''
		This compute loss and ci on the whole dataset. It is reasonable.
		'''
		## losses and c-indices on traning
		#loss_train = np.zeros(len(self.datasets_train))
		#ci_train = np.zeros(len(self.datasets_train))

		X, time, event = zip(*datasets)
		X = np.concatenate(X, axis=0)
		time = np.concatenate(time, axis=0)
		event = np.concatenate(event, axis=0)
		assert len(X) == len(time) == len(event), print ('X, time, event len are not equal!!!')

		loss_sum = 0
		ci_sum = 0
		count = 0
		bs = 16

		for i in range(len(X)//bs):
			X_batch, time_batch, event_batch = X[bs*i:bs*(i+1),...], time[bs*i:bs*(i+1),...], event[bs*i:bs*(i+1),...]

			score_i, time_i, event_i = self.__sess.run([self.__score, self.__time, self.__event], feed_dict = {self.__X: X_batch, self.__time: time_batch, self.__event: event_batch, K.learning_phase(): 0})

			if i == 0:
				socre_all = score_i
				time_all = time_i
				event_all = event_i
			else:
				socre_all = np.concatenate((socre_all, score_i), axis=0)
				time_all = np.concatenate((time_all, time_i), axis=0)
				event_all = np.concatenate((event_all, event_i), axis=0)
				
			#loss_i, ci_i = self.evaluate(X_batch, time_batch, event_batch)
			#print ('loss_i, ci_i: ',loss_i, ci_i)

			#if math.isnan(loss_i) or math.isnan(ci_i) or math.isinf(loss_i) or math.isinf(ci_i):
			#	print ('here')
			#	continue
			#loss_sum += loss_i*bs
			#ci_sum += ci_i*bs
			#count += bs             

			#print (loss_i, ci_i, math.isnan(loss_i), math.isnan(ci_i))
			#print (X_batch.shape, time_batch.shape, event_batch.shape)
		#print (loss_sum, ci_sum, count)

		i += 1
		X_batch, time_batch, event_batch = X[bs*i:len(X),...], time[bs*i:len(time),...], event[bs*i:len(event),...]
		#loss_i, ci_i = self.evaluate(X_batch, time_batch, event_batch)

		score_i, time_i, event_i = self.__sess.run([self.__score, self.__time, self.__event], feed_dict = {self.__X: X_batch, self.__time: time_batch, self.__event: event_batch, K.learning_phase(): 0})

		socre_all = np.concatenate((socre_all, score_i), axis=0)
		time_all = np.concatenate((time_all, time_i), axis=0)
		event_all = np.concatenate((event_all, event_i), axis=0)
		#print (socre_all.shape, time_all.shape, event_all.shape)

		'''
		ci = self.__concordance_index(socre_all, time_all, event_all)
		if self.loss_func=='hinge':
			loss = self.__hinge_loss(socre_all, time_all, event_all)
		elif self.loss_func=='log':
			loss = self.__log_loss(socre_all, time_all, event_all)
		elif self.loss_func=='cox':
			loss = self.__cox_loss(socre_all, time_all, event_all)		
		'''


		loss, ci = self.__sess.run([self.__loss, self.__ci], feed_dict={self.__score: socre_all, self.__time: time_all, self.__event: event_all, K.learning_phase(): 0})

		#print (loss, ci)
		#raise

		#if not (math.isnan(loss_i) or math.isnan(ci_i) or math.isinf(loss_i) or math.isinf(ci_i)):
		#	loss_sum += loss_i*(len(X)-bs*i)
		#	ci_sum += ci_i*(len(X)-bs*i)	
		#	count += (len(X)-bs*i)


		## print them
		#print('loss={0}'.format(round(loss, 3)))
		#print('ci={0}'.format(round(ci, 3)))
		print(f'loss={loss}')
		print(f'ci={ci}')
		print()


	def __print_loss_ci_yh_avg(self, datasets, datasets_1):
		'''
		This compute average loss and ci on the cut dataset and not cut one.
		'''
		## losses and c-indices on traning
		#loss_train = np.zeros(len(self.datasets_train))
		#ci_train = np.zeros(len(self.datasets_train))

		X, time, event = zip(*datasets)
		X_1, time_1, event_1 = zip(*datasets_1)


		X = np.concatenate(X, axis=0)
		X_1 = np.concatenate(X_1, axis=0)

		time = np.concatenate(time, axis=0)
		event = np.concatenate(event, axis=0)


		assert len(X_1) == len(X) == len(time) == len(event), print ('X_1, X, time, event len are not equal!!!')

		bs = 16

		for i in range(len(X)//bs):
			X_batch, time_batch, event_batch = X[bs*i:bs*(i+1),...], time[bs*i:bs*(i+1),...], event[bs*i:bs*(i+1),...]

			score_i, time_i, event_i = self.__sess.run([self.__score, self.__time, self.__event], feed_dict = {self.__X: X_batch, self.__time: time_batch, self.__event: event_batch, K.learning_phase(): 0})

			if i == 0:
				socre_all = score_i
				time_all = time_i
				event_all = event_i
			else:
				socre_all = np.concatenate((socre_all, score_i), axis=0)
				time_all = np.concatenate((time_all, time_i), axis=0)
				event_all = np.concatenate((event_all, event_i), axis=0)
				

		i += 1
		X_batch, time_batch, event_batch = X[bs*i:len(X),...], time[bs*i:len(time),...], event[bs*i:len(event),...]
		#loss_i, ci_i = self.evaluate(X_batch, time_batch, event_batch)

		score_i, time_i, event_i = self.__sess.run([self.__score, self.__time, self.__event], feed_dict = {self.__X: X_batch, self.__time: time_batch, self.__event: event_batch, K.learning_phase(): 0})

		socre_all = np.concatenate((socre_all, score_i), axis=0)
		time_all = np.concatenate((time_all, time_i), axis=0)
		event_all = np.concatenate((event_all, event_i), axis=0)
		#print (socre_all.shape, time_all.shape, event_all.shape)


		for i in range(len(X_1)//bs):
			X_batch, time_batch, event_batch = X_1[bs*i:bs*(i+1),...], time[bs*i:bs*(i+1),...], event[bs*i:bs*(i+1),...]

			score_i, time_i, event_i = self.__sess_1.run([self.__score, self.__time, self.__event], feed_dict = {self.__X: X_batch, self.__time: time_batch, self.__event: event_batch, K.learning_phase(): 0})

			if i == 0:
				socre_all_1 = score_i
				time_all = time_i
				event_all = event_i
			else:
				socre_all_1 = np.concatenate((socre_all_1, score_i), axis=0)
				time_all = np.concatenate((time_all, time_i), axis=0)
				event_all = np.concatenate((event_all, event_i), axis=0)
				

		i += 1
		X_batch, time_batch, event_batch = X_1[bs*i:len(X),...], time[bs*i:len(time),...], event[bs*i:len(event),...]
		#loss_i, ci_i = self.evaluate(X_batch, time_batch, event_batch)

		score_i, time_i, event_i = self.__sess_1.run([self.__score, self.__time, self.__event], feed_dict = {self.__X: X_batch, self.__time: time_batch, self.__event: event_batch, K.learning_phase(): 0})

		socre_all_1 = np.concatenate((socre_all_1, score_i), axis=0)
		time_all = np.concatenate((time_all, time_i), axis=0)
		event_all = np.concatenate((event_all, event_i), axis=0)

		#print (socre_all.shape, socre_all_1.shape, time_all.shape)

		#print (socre_all[0:10])
		#print (socre_all_1[0:10])
		#print (time_all[0:10])
		score_avg = (socre_all + socre_all_1)/2


		loss, ci = self.__sess.run([self.__loss, self.__ci], feed_dict={self.__score: score_avg, self.__time: time_all, self.__event: event_all, K.learning_phase(): 0})
		loss, ci2 = self.__sess_1.run([self.__loss, self.__ci], feed_dict={self.__score: score_avg, self.__time: time_all, self.__event: event_all, K.learning_phase(): 0})
		loss, ci3 = self.__sess.run([self.__loss, self.__ci], feed_dict={self.__score: socre_all, self.__time: time_all, self.__event: event_all, K.learning_phase(): 0})
		loss, ci4 = self.__sess.run([self.__loss, self.__ci], feed_dict={self.__score: socre_all_1, self.__time: time_all, self.__event: event_all, K.learning_phase(): 0})

		#print (loss, ci)
		#raise

		#if not (math.isnan(loss_i) or math.isnan(ci_i) or math.isinf(loss_i) or math.isinf(ci_i)):
		#	loss_sum += loss_i*(len(X)-bs*i)
		#	ci_sum += ci_i*(len(X)-bs*i)	
		#	count += (len(X)-bs*i)


		## print them
		#print('loss={0}'.format(round(loss, 3)))
		#print('ci={0}'.format(round(ci, 3)))
		print(f'loss={loss}')
		print(f'ci={ci}')
		print(f'ci2={ci2}')
		print(f'ci3={ci3}')
		print(f'ci4={ci4}')
		print()
