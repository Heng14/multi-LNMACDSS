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
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from resnet_101 import resnet101_model
from resnet_50 import resnet50_model
from test_net import testnet_model
from resnet18 import ResnetBuilder
from snet import snet
#from snet_true_fusion import snet
#from snet_true import snet
from test_snet import test_snet
from grad_cam import *
from imgaug import augmenters as iaa
import yaml

datagen = ImageDataGenerator(
    #rotation_range=30,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #zoom_range = 0.2,
    #horizontal_flip=False,
    #vertical_flip=False
)

#    shear_range=0.5,


seq = iaa.Sequential([
    iaa.Crop(percent=(0, 0.1)), 
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Add((-0.1, 0.1)),
    iaa.Multiply((0.9, 1.1)),
    #iaa.OneOf([
    #    iaa.Affine(scale=(0.9, 1.1)),
    #    iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
    #    iaa.Affine(rotate=(-10, 10)),
    #    iaa.Affine(shear=(-10, 10)),
    #    ]),
])


def batch_factory(X, Y, batch_size):
    print (X.shape, Y.shape)
    n = X.shape[0]
    num_batches = n//batch_size
    idx = np.random.permutation(n)
    X, Y = X[idx], Y[idx] # randomly shuffle data
    start = 0
    def next_batch():
        nonlocal start
        X_batch = X[start:start+batch_size]
        Y_batch = Y[start:start+batch_size]
        start = (start+batch_size)%n
        return X_batch, Y_batch
    return next_batch, num_batches


class SurvivalModel:

    def __init__(self):

        self.model_builder = ResnetBuilder.build_resnet_18


    def run(self, datasets_train, datasets_val, datasets_test, train_name=None, val_name=None, test_name=None,  epochs=500, lr=0.001, mode='train', batch_size = 8):

        self.datasets_train = datasets_train
        self.datasets_val = datasets_val
        self.datasets_test = datasets_test
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.train_name = train_name
        self.val_name = val_name
        self.test_name = test_name
        self.model = ResnetBuilder.build_resnet_18((12, 160, 160), 2)
        self.model_index = 418
        self.data_index = '12a'

        if mode=='train':
            self.__train()
        elif mode=='infer':
            self.__infer()
        elif mode=='pred':
            self.__pred()
        elif mode=='vis':
            self.__vis()
        elif mode=='vis_cam':
            self.__vis_cam()

    def __train(self):
        opt = keras.optimizers.Adagrad(lr=self.lr, epsilon=1e-06)
        #opt = keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        X, Y = zip(*self.datasets_train)
        X_train = np.concatenate(X, axis=0)
        Y_train = np.concatenate(Y, axis=0)
        assert len(X_train) == len(Y_train), 'X_train, Y_train len are not equal!!!'
        iter_n = int(len(X_train)/self.batch_size)

        X, Y = zip(*self.datasets_val)
        X_val = np.concatenate(X, axis=0)
        Y_val = np.concatenate(Y, axis=0)

        X, Y = zip(*self.datasets_test)
        X_test = np.concatenate(X, axis=0)
        Y_test = np.concatenate(Y, axis=0)


        #next_batch, num_batches = utils.batch_factory(X_train, Y_train, self.batch_size)
        #for _ in range(num_batches):
        #    X_batch, Y_batch = next_batch()
        #    X_batch = seq.augment_images(X_batch)

        datagen = ImageDataGenerator(preprocessing_function=seq.augment_image)

        model_checkpoint = ModelCheckpoint('tmp_ckpts/resnet18_cls_{epoch:03d}.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
        model_lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=50, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

        self.model.fit_generator(datagen.flow(X_train, Y_train, batch_size=self.batch_size),steps_per_epoch=iter_n, epochs=self.epochs, verbose=1, callbacks=[model_checkpoint, model_lr_decay], validation_data=(X_val, Y_val), workers=4)
        scores = self.model.evaluate(X_test, Y_test, verbose=1)
        print (scores)


    #def __evaluate(self, ):


    def __infer(self, save=False, save_prob=True):

        X, Y = zip(*self.datasets_train)
        X_train = np.concatenate(X, axis=0)
        Y_train = np.concatenate(Y, axis=0)
        assert len(X_train) == len(Y_train), 'X_train, Y_train len are not equal!!!'
        iter_n = int(len(X_train)/self.batch_size)

        X, Y = zip(*self.datasets_val)
        X_val = np.concatenate(X, axis=0)
        Y_val = np.concatenate(Y, axis=0)

        X, Y = zip(*self.datasets_test)
        X_test = np.concatenate(X, axis=0)
        Y_test = np.concatenate(Y, axis=0)

        opt = keras.optimizers.Adagrad(lr=self.lr, epsilon=1e-06)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.model.load_weights(f"NO{self.data_index}_model/resnet18_cls_{self.model_index}.hdf5")

        scores_train = self.model.evaluate(X_train, Y_train, verbose=1)
        print ("scores_train loss and acc: ", scores_train)
        scores_val = self.model.evaluate(X_val, Y_val, verbose=1)
        print ("scores_val loss and acc: ", scores_val)
        scores_test = self.model.evaluate(X_test, Y_test, verbose=1)
        print ("scores_test loss and acc: ", scores_test)

        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        y_test_pred = self.model.predict(X_test)

        #print (np.argmax(y_train_pred, axis=1))
        #print (np.argmax(y_val_pred, axis=1))
        #print (np.argmax(y_test_pred, axis=1))

        if save:
            #print ("y_train_pred and Y_train")
            np.save(f'NO{self.data_index}_model/y_train_pred_{self.model_index}', np.argmax(y_train_pred, axis=1)) 
            np.save(f'NO{self.data_index}_model/Y_train_{self.model_index}', np.argmax(Y_train, axis=1))

            #print ("y_val_pred and Y_val")
            np.save(f'NO{self.data_index}_model/y_val_pred_{self.model_index}', np.argmax(y_val_pred, axis=1)) 
            np.save(f'NO{self.data_index}_model/Y_val_{self.model_index}', np.argmax(Y_val, axis=1))

            #print ("y_test_pred and Y_test")
            np.save(f'NO{self.data_index}_model/y_test_pred_{self.model_index}', np.argmax(y_test_pred, axis=1)) 
            np.save(f'NO{self.data_index}_model/Y_test_{self.model_index}', np.argmax(Y_test, axis=1))

        if save_prob:
            np.save(f'NO{self.data_index}_model/y_train_pred_{self.model_index}_prob', y_train_pred[:,0]) 
            np.save(f'NO{self.data_index}_model/y_val_pred_{self.model_index}_prob', y_val_pred[:,0]) 
            np.save(f'NO{self.data_index}_model/y_test_pred_{self.model_index}_prob', y_test_pred[:,0]) 

            train_f = open(f'NO{self.data_index}_model/train_name.yaml','w')
            yaml.dump(self.train_name, train_f)

            val_f = open(f'NO{self.data_index}_model/val_name.yaml','w')
            yaml.dump(self.val_name, val_f)

            test_f = open(f'NO{self.data_index}_model/test_name.yaml','w')
            yaml.dump(self.test_name, test_f)

    def __pred(self):

        X, Y = zip(*self.datasets_train)
        X_train = np.concatenate(X, axis=0)
        Y_train = np.concatenate(Y, axis=0)
        assert len(X_train) == len(Y_train), 'X_train, Y_train len are not equal!!!'

        X, Y = zip(*self.datasets_val)
        X_val = np.concatenate(X, axis=0)
        Y_val = np.concatenate(Y, axis=0)

        X, Y = zip(*self.datasets_test)
        X_test = np.concatenate(X, axis=0)
        Y_test = np.concatenate(Y, axis=0)

        self.model.load_weights("NO1_model/resnet18_cls_318.hdf5")
        y_pred = self.model.predict(X_test)
        print (np.argmax(y_pred, axis=1))
        print (Y_test)


    def __vis_cam(self):

        weights_path = "NO1_model/resnet18_cls_318.hdf5"
        self.model.load_weights(weights_path, by_name=True)

        X, Y = zip(*self.datasets_train)
        X_train = np.concatenate(X, axis=0)
        Y_train = np.concatenate(Y, axis=0)
        assert len(X_train) == len(Y_train), print ('X, time, event len are not equal!!!')


        for index in range(0, 1):

            x_in = X[index]
            name = self.train_name[index]
            print (name, index)
            #raise

            vis_out_dir = f'vis_layer_cam'
            os.makedirs(vis_out_dir, exist_ok=True)

            print (x_in.shape)
            print (x_in.max(), x_in.min())
            #x_in = (x_in-np.min(x_in))/(np.max(x_in)-np.min(x_in))
            x_in_sq = np.squeeze(x_in)
            for i in range(len(x_in_sq[0,0,:])):
                io.imsave(os.path.join(vis_out_dir, f'ori_{index}_{i}.jpg'),  x_in_sq[:,:,i])

            image_arr=np.reshape(x_in, (-1,160,160,12))

            predictions = self.model.predict(image_arr)
            predicted_class = np.argmax(predictions)
            print (predictions, predicted_class)

            #for layer in self.model.layers:
    	        #print (layer.name)

            cam, heatmap = grad_cam(self.model, image_arr, predicted_class, "conv2d_17")
            cv2.imwrite(os.path.join(vis_out_dir, f"gradcam_{index}.jpg"), heatmap)
            #raise

            register_gradient()
            guided_model = modify_backprop(self.model, 'GuidedBackProp')
            saliency_fn = compile_saliency_function(guided_model)
            saliency = saliency_fn([image_arr, 0])
            gradcam = saliency[0] * heatmap[..., np.newaxis]
            de_img = deprocess_image(gradcam)
            for i in range(len(de_img[0,0,:])):
                io.imsave(os.path.join(vis_out_dir, f'guided_gradcam_{index}_{i}.jpg'),  de_img[:,:,i])

            #cv2.imwrite(os.path.join(vis_out_dir, f"guided_gradcam_{index}.jpg"), deprocess_image(gradcam))
            raise
        return 

















