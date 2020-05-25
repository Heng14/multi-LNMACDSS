"""
this is for classification metastasis or not
"""
import os
from skimage import io
import tensorflow as tf
from utils import *
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
#from snet_true_fusion import snet
#from snet_true import snet
from test_snet import test_snet
from test_snet1 import test_snet1
from grad_cam import *

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range = 0.2,
    horizontal_flip=True,
    vertical_flip=True)

#    shear_range=0.5,

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
		#self.model_builder = test_snet
		#self.model_builder = test_snet1

	def fit(self, datasets_train, datasets_val, datasets_test, train_name=None, val_name=None, test_name=None, loss_func='cross_entropy', epochs=500, lr=0.001, mode='merge', batch_size = 8):
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
		self.loss_func = loss_func
		self.epochs = epochs
		self.lr = lr
		self.batch_size = batch_size
		self.train_name = train_name
		self.val_name = val_name
		self.test_name = test_name

		## build a tensorflow graph to define loss function
		self.__build_graph()
		
		## train the model
		if mode=='train':
		  self.__train_merge()
		elif mode=='infer':
		  self.__infer()
		elif mode=='vis':
		  self.__vis()
		elif mode=='vis_cam':
		  self.__vis_cam()


	def __build_graph(self):
		'''
		Build a tensorflow graph. Call this within self.fit()
		'''

		input_shape = self.datasets_train[0][0].shape[1:]
		n_classes = 2
		#print (input_shape[0])
		#raise

		with tf.name_scope('input'):
			X = tf.placeholder(dtype=tf.float32, shape=(None, )+input_shape, name='X')
			Y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='Y')

		with tf.name_scope('model'):
			#self.model = self.model_builder(input_shape[0], input_shape[0])
			#self.model = self.model_builder((3, 160, 160), 2)
			self.model = self.model_builder()

		with tf.name_scope('output'):
			score = tf.identity(self.model(X), name='score')

		with tf.name_scope('metric'):
			acc, correct_prediction = self.__calc_acc(score, Y)
			if self.loss_func=='cross_entropy':
				loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=score, labels=Y))

		with tf.name_scope('train'):

			num_epoch = tf.Variable(0, name='global_step', trainable=False)
			boundaries = [5, 25, 70]
			#learing_rates = [0.0001, 0.0005, 0.00001, 0.00005]
			#learing_rates = [0.0001, 0.0001, 0.0001, 0.0001]
			learing_rates = [0.0001, 0.0001, 0.0001, 0.0001]
			learing_rate = tf.train.piecewise_constant(num_epoch, boundaries=boundaries, values=learing_rates)
			optimizer = tf.train.AdamOptimizer(learning_rate=learing_rate)
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learing_rate)
			#optimizer = tf.train.MomentumOptimizer(learning_rate=learing_rate, momentum=0.9)

			#optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
			train_op = optimizer.minimize(loss, name='train_op')

		## save the tensors and ops so that we can use them later
		self.__X = X
		self.__Y = Y
		self.__score = score
		self.__acc = acc
		self.__correct_prediction = correct_prediction
		self.__loss = loss
		self.__train_op = train_op

	def __calc_acc(self, score, Y):
		correct_prediction = tf.equal(tf.argmax(score, 1), tf.argmax(Y, 1))
		acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return acc, correct_prediction

			
	def __infer(self):
		weights_path = "checkpoints/my_fusion_model_weights_58.h5"
		self.__sess = tf.Session()
		K.set_session(self.__sess)
		self.__sess.run(tf.global_variables_initializer())
		self.model.load_weights(weights_path, by_name=True)
		self.__print_loss_ci_yh_infer(self.datasets_train, self.train_name, 'train')
		self.__print_loss_ci_yh_infer(self.datasets_val, self.val_name, 'val')
		self.__print_loss_ci_yh_infer(self.datasets_test, self.test_name, 'test')

	def __vis(self):
		weights_path = "checkpoints/my_model_weights_35.h5"
		self.__sess = tf.Session()
		K.set_session(self.__sess)
		self.__sess.run(tf.global_variables_initializer())
		self.model.load_weights(weights_path, by_name=True)
		self.__vis_layer(self.datasets_train, self.train_name, 'train')

	def __vis_cam(self):
		weights_path = "checkpoints/my_model_weights_113.h5"
		#self.__sess = tf.Session()
		#K.set_session(self.__sess)
		#self.__sess.run(tf.global_variables_initializer())
		self.model.load_weights(weights_path, by_name=True)
		self.__vis_layer_cam(self.datasets_train, self.train_name, 'train')


	def __train_merge(self):
		'''
		Merge training datasets into a single dataset. Sample mini-batches from the merged dataset for training
		'''
		#weights_path = 'checkpoints/resnet101_weights_tf.h5'
		#weights_path = 'checkpoints/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

		## Merge training datasets
		#X_train, Y_train = utils.combine_datasets(self.datasets_train)

		X, Y = zip(*self.datasets_train)
		#print (len(X))
		#print ('*'*20)
		#print (len(Y))
		X_train = np.concatenate(X, axis=0)
		Y_train = np.concatenate(Y, axis=0)
		#print (len(X_train))
		#print ('*'*20)
		#print (len(Y_train))

		assert len(X_train) == len(Y_train), 'X_train, Y_train len are not equal!!!'


		## get training datasets ___ by heng
		#X_train, time_train, event_train = utils.get_datasets(self.datasets_train)

		## To fetch mini-batches
		#next_batch, num_batches = utils.batch_factory(X_train, time_train, event_train, self.batch_size)

		## start training
		self.__sess = tf.Session()
		K.set_session(self.__sess)
		self.__sess.run(tf.global_variables_initializer())
		#self.__sess.run(tf.truncated_normal_initializer())
		#self.model.load_weights(weights_path, by_name=True)

		file_train = open(f"train_log.txt","w")
		file_val = open(f"val_log.txt","w")
		file_test = open(f"test_log.txt","w")

		print (f'pre epoch train log:')
		acc_train, loss_train = self.__print_loss_acc_yh(self.datasets_train)
		acc_val, loss_val = self.__print_loss_acc_yh(self.datasets_val)
		acc_test, loss_test = self.__print_loss_acc_yh(self.datasets_test)
		
		file_train.write(str(acc_train)+" "+str(loss_train)+"\n")
		file_val.write(str(acc_val)+" "+str(loss_val)+"\n")
		file_test.write(str(acc_test)+" "+str(loss_test)+"\n")

		acc = 0

		for epoch in range(self.epochs):

			#next_batch, num_batches = utils.batch_factory(X_train, time_train, event_train, self.batch_size)
			#for _ in range(num_batches):
			#	X_batch, time_batch, event_batch = next_batch()

			#time_event = np.hstack((time_train[...,None], event_train[...,None]))			
			batches = 0
			for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=self.batch_size):
				batches += 1
				if batches >= X_train.shape[0] // self.batch_size:
					break

				self.__sess.run(self.__train_op, feed_dict={self.__X: X_batch, self.__Y: Y_batch, K.learning_phase(): 1})
				#print (self.__sess.run([self.__train_op, self.__acc], feed_dict={self.__X: X_batch, self.__Y: Y_batch, K.learning_phase(): 1}))
			#print (self.__sess.run([self.__score, self.__loss, self.__ci], feed_dict={self.__X: X_batch, self.__time: time_batch, self.__event: event_batch, K.learning_phase(): 0}))

			print('-'*20 + 'Epoch: {0}'.format(epoch) + '-'*20)
			print (f'epoch {epoch} train log:')
			acc_train, loss_train = self.__print_loss_acc_yh(self.datasets_train)
			file_train.write(str(acc_train)+" "+str(loss_train)+"\n")

			print (f'epoch {epoch} val log:')
			acc_val, loss_val = self.__print_loss_acc_yh(self.datasets_val)
			file_val.write(str(acc_val)+" "+str(loss_val)+"\n")
			#if val_ci > ci:
			#	ci = val_ci
			#	self.model.save_weights(f'my_model_weights_{epoch}.h5')
			#	print ("model saved!")
			print (f'epoch {epoch} test log:')
			acc_test, loss_test = self.__print_loss_acc_yh(self.datasets_test)
			file_test.write(str(acc_test)+" "+str(loss_test)+"\n")

			#if ci_test > ci:
			#if acc_test > 0.62:
			#	acc = acc_test
				#self.model.save_weights(f'my_fusion_model_weights_{epoch}.h5')
			#	print ("model saved!")

			if epoch>300:
				file_train.close()
				file_val.close()
				file_test.close()
				raise


	def __print_loss_acc_yh(self, datasets):
		'''
		This compute loss and ci on the whole dataset. It is reasonable.
		'''

		X, Y = zip(*datasets)
		X = np.concatenate(X, axis=0)
		Y = np.concatenate(Y, axis=0)
		assert len(X) == len(Y), 'X, Y len are not equal!!!'

		loss_sum = 0
		acc_sum = 0
		count = 0
		bs = 16

		for i in range(len(X)//bs):
			X_batch, Y_batch = X[bs*i:bs*(i+1),...], Y[bs*i:bs*(i+1),...]

			score_i, Y_i = self.__sess.run([self.__score, self.__Y], feed_dict = {self.__X: X_batch, self.__Y: Y_batch, K.learning_phase(): 1})
			#print (score_i)
			if i == 0:
				socre_all = score_i
				Y_all = Y_i
			else:
				socre_all = np.concatenate((socre_all, score_i), axis=0)
				Y_all = np.concatenate((Y_all, Y_i), axis=0)

		i += 1
		X_batch, Y_batch = X[bs*i:len(X),...], Y[bs*i:len(Y),...]

		score_i, Y_i = self.__sess.run([self.__score, self.__Y], feed_dict = {self.__X: X_batch, self.__Y: Y_batch, K.learning_phase(): 1})

		socre_all = np.concatenate((socre_all, score_i), axis=0)
		Y_all = np.concatenate((Y_all, Y_i), axis=0)

		#print (socre_all.shape, Y_all.shape)
		#print (socre_all, Y_all)
		loss, acc = self.__sess.run([self.__loss, self.__acc], feed_dict={self.__score: socre_all, self.__Y: Y_all, K.learning_phase(): 1})

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
		print(f'acc={acc}')
		return acc, loss

	def __print_loss_ci_yh_infer(self, datasets, dataset_name, flag = 'train'):
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

		#print (len(socre_all), len(time_all), len(event_all))
		cal_ci(socre_all, time_all, event_all, dataset_name, flag)

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
		return ci



	def __vis_layer(self, datasets, dataset_name, flag = 'train'):


		X, time, event = zip(*datasets)
		X = np.concatenate(X, axis=0)
		time = np.concatenate(time, axis=0)
		event = np.concatenate(event, axis=0)
		assert len(X) == len(time) == len(event), print ('X, time, event len are not equal!!!')

		index = 12
		x_in = X[index]

		vis_out_dir = f'vis_{index}_layer'
		os.makedirs(vis_out_dir, exist_ok=True)

		print (x_in.shape)
		#print (x_in.max(), x_in.min())
		x_in = (x_in-np.min(x_in))/(np.max(x_in)-np.min(x_in))


		io.imsave(os.path.join(vis_out_dir, f'ori.jpg'),  np.squeeze(x_in))

		image_arr=np.reshape(x_in, (-1,160,160,1))
		layer = K.function([self.model.layers[0].input], [self.model.layers[3].output])
		f1 = layer([image_arr])[0]
		re = np.squeeze(np.transpose(f1, (0,3,1,2)))

		print (re.shape)
		for i in range(re.shape[0]):
			temp_im = re[i]
			temp_im = (temp_im-np.min(temp_im))/(np.max(temp_im)-np.min(temp_im))
			io.imsave(os.path.join(vis_out_dir, f'layer2_{i}.jpg'), temp_im)

		return 


	def __vis_layer_cam(self, datasets, dataset_name, flag = 'train'):


		X, time, event = zip(*datasets)
		X = np.concatenate(X, axis=0)
		time = np.concatenate(time, axis=0)
		event = np.concatenate(event, axis=0)
		assert len(X) == len(time) == len(event), print ('X, time, event len are not equal!!!')

		#index = 1

		for index in range(0, len(X)):
		
		    x_in = X[index]
		    name = dataset_name[index]
		    print (name, index)
		    raise

		    vis_out_dir = f'vis_layer_cam'
		    os.makedirs(vis_out_dir, exist_ok=True)

		    print (x_in.shape)
		    print (x_in.max(), x_in.min())
		    #x_in = (x_in-np.min(x_in))/(np.max(x_in)-np.min(x_in))
    
		    io.imsave(os.path.join(vis_out_dir, f'ori_{index}.jpg'),  x_in)

		    image_arr=np.reshape(x_in, (-1,160,160,3))

		    predictions = self.model.predict(image_arr)
		    predicted_class = np.argmax(predictions)
		    print (predictions, predicted_class)

		    #for layer in self.model.layers:
		    	#print (layer.name)
		
		    cam, heatmap = grad_cam(self.model, image_arr, predicted_class, "conv2d_17")
		    cv2.imwrite(os.path.join(vis_out_dir, f"gradcam_{index}.jpg"), cam)

		    register_gradient()
		    guided_model = modify_backprop(self.model, 'GuidedBackProp')
		    saliency_fn = compile_saliency_function(guided_model)
		    saliency = saliency_fn([image_arr, 0])
		    gradcam = saliency[0] * heatmap[..., np.newaxis]
		    cv2.imwrite(os.path.join(vis_out_dir, f"guided_gradcam_{index}.jpg"), deprocess_image(gradcam))
		    raise
		return 

