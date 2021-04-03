#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPool1D, Flatten, Dense, LeakyReLU, UpSampling1D, Lambda, Dropout, Multiply, Reshape, Cropping1D
from keras.optimizers import RMSprop, SGD, Adam
from keras import initializers
from keras import activations
from keras.utils import plot_model
import keras.losses
import numpy as np

from layers import MaxPoolingWithArgmax1D, MaxUnpooling1D

class Architecture:
    def __init__(self, series_length, rep_length, learning_rate=0.001, n_channels=1, loss_weights = [1,1,2]):
        self.series_length = series_length
        self.rep_length = rep_length
        self.input_shape = (series_length, n_channels)
        
        self.learning_rate = learning_rate
        
        self.loss_weights = loss_weights
        
    def create_new_architecture(self, seed=1234):
        time_series_1 = Input(shape=self.input_shape)
        time_series_2 = Input(shape=self.input_shape)
        
        rep_layer_1, AE1_out = self._create_autoencoder(time_series_1, 'AE1', seed, self.rep_length)
        rep_layer_2, AE2_out = self._create_autoencoder(time_series_2, 'AE2', seed, self.rep_length)
        
        l2_square_distance_layer = Lambda(
                    lambda tensors: K.sum((tensors[0] - tensors[1])**2, axis=-1, keepdims=True), name='rep_diff')
        
        rep_diff = l2_square_distance_layer([rep_layer_1, rep_layer_2])
        
        model = Model(inputs=[time_series_1, time_series_2], outputs=[AE1_out, AE2_out, rep_diff])
        
        def custom_loss(y_true, y_pred):
            return  y_pred
        
        
        keras.losses.custom_loss=custom_loss
        
        losses = {
                'AE1':'mean_squared_error',
                'AE2':'mean_squared_error',
                'rep_diff':custom_loss,
                }
        loss_weights = {
                'AE1':self.loss_weights[0],
                'AE2':self.loss_weights[1],
                'rep_diff':self.loss_weights[2],
                }
        
#        model.compile(optimizer=SGD(lr=self.learning_rate), loss=losses, loss_weights=loss_weights)
        model.compile(optimizer=Adam(lr=self.learning_rate), loss=losses, loss_weights=loss_weights)
        
        self.model = model
        return model

    def get_layers(self):
        "Prints the index and name of each layer in the architecture"
        if not hasattr(self, 'model'):
            "Model object has not been created yet. Create one."
            self.create_new_architecture()
            
        for idx, layer in enumerate(self.model.layers):
            print(idx, layer.name)

    def get_summary(self):
        "Return summary of the model."
        if not hasattr(self, 'model'):
            "Model object has not been created yet. Create one."
            self.create_new_architecture()
        
        self.model.summary()

    def get_output_index(self):
        return [24, 25]

    def _create_autoencoder(self, time_series, name, seed, rep_len):
        #encoder
        # conv16@7
        x = Conv1D(filters=16, kernel_size=5, strides=1, padding="same", kernel_initializer=initializers.he_normal(seed=seed), data_format='channels_last')(time_series)
        x = LeakyReLU(alpha=0.3)(x)
        # conv16@5
        x = Conv1D(filters=16, kernel_size=5, strides=1, padding="same", kernel_initializer=initializers.he_normal(seed=seed), data_format="channels_last")(x)
        x = LeakyReLU(alpha=0.3)(x)
        maxpool1_fanin = x._keras_shape[-2]
        # maxpooling@3
        x = MaxPool1D(pool_size=3, padding='same')(x)
        # conv32@3
        x = Conv1D(filters=32, kernel_size=3, strides=1, padding="same", kernel_initializer=initializers.he_normal(seed=seed), data_format="channels_last")(x)
        x = LeakyReLU(alpha=0.3)(x)
        # conv32@3
        x = Conv1D(filters=32, kernel_size=3, strides=1, padding="same", kernel_initializer=initializers.he_normal(seed=seed), data_format="channels_last")(x)
        x = LeakyReLU(alpha=0.3)(x)
        # maxpooling@0.1L
        maxpool2_fanin = x._keras_shape[-2]
        pool_width = int(np.ceil(0.1 * maxpool2_fanin))
        #signet pooling
        x, mask = MaxPoolingWithArgmax1D(pool_size=pool_width, padding="same")(x) #M,16
        num_channels = x._keras_shape[-1]

        x = Flatten()(x) # 16M length
        num_nodes = x._keras_shape[-1]
        # dense@rep_len
        rep = Dense(units=rep_len, kernel_initializer=initializers.he_normal(seed=seed))(x) ## 10/40
        
        # decoder
        dense_upsample = Dense(units=num_nodes, kernel_initializer=initializers.he_normal(seed=seed))(rep) #160
        x = Reshape((int(num_nodes/num_channels),num_channels))(dense_upsample) #M,16        
        # upsampling@0.1L
        #x = UpSampling1D(size=pool_width)(x)
        x = MaxUnpooling1D(pool_width)([x, mask])
        upsample1_fanout = x._keras_shape[-2]
        # crop excess from right end.
        x = Cropping1D(cropping=(0, upsample1_fanout - maxpool2_fanin))(x) # L,16
        # conv32@3
        x = Conv1D(filters=32, kernel_size=3, strides=1, padding="same", kernel_initializer=initializers.he_normal(seed=seed), data_format="channels_last")(x)
        x = LeakyReLU(alpha=0.3)(x)
        # conv32@3
        x = Conv1D(filters=16, kernel_size=3, strides=1, padding="same", kernel_initializer=initializers.he_normal(seed=seed), data_format="channels_last")(x)
        x = LeakyReLU(alpha=0.3)(x)
        # upsampling@3
        x = UpSampling1D(size=3)(x)
        # Crop excess from the right end.
        upsample2_fanout = x._keras_shape[-2]
        x = Cropping1D(cropping=(0, upsample2_fanout - maxpool1_fanin))(x)
        # conv16@5
        x = Conv1D(filters=16, kernel_size=5, strides=1, padding="same", kernel_initializer=initializers.he_normal(seed=seed), data_format="channels_last")(x)
        x = LeakyReLU(alpha=0.3)(x)
        # conv16@7
        x = Conv1D(filters=1, kernel_size=7, strides=1, padding="same", kernel_initializer=initializers.he_normal(seed=seed), data_format="channels_last")(x)
        out = Lambda(lambda y:y, name=name)(x)

        return rep, out

