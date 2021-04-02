#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import keras
import copy
import os
import utils
import random

class InterpolationGenerator(keras.utils.Sequence):
    'Generates data for vanilla generator.'
    def __init__(self, list_IDs, series_length, df, batch_size=32, n_channels=1,shuffle=True, num_warps=10, seed=1234,point=True):
        'Initialization'
        np.random.seed(seed)
        self.batch_size=batch_size
        self.batch_size_trimmed = int(self.batch_size / 2)
        self.series_length = series_length
        self.dims  = (series_length,)
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.num_warps = int(self.series_length * num_warps / 100)
        self.df = df
        self.on_epoch_end()
        self.point=point
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs)/self.batch_size_trimmed))
    
    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indices of the batch.
        if index == self.__len__() - 1:
            indices = self.indices[index * self.batch_size_trimmed:]
        else:
            indices = self.indices[index * self.batch_size_trimmed:(index + 1) * self.batch_size_trimmed]
        
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        
        # Generate data.
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y
    
    def on_epoch_end(self):
        "Update indexes after each epoch"
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
    
    def interpolation_warp(self, series, l):
    
            num_warpings = 1
            if self.point:
                window_len=np.random.randint(4,7)
            else:
                window_len=np.random.randint(int(0.05*self.series_length),int(0.1*self.series_length))   
            
            pos = np.random.randint(0, self.series_length - window_len)

            if l:
                left = copy.deepcopy(series)
                left[pos+1]=left[pos+window_len-1]
                for num in range(2,window_len):

                   left[pos + num] = ((window_len-num-1)*left[pos+ window_len-1] + num*left[pos+ window_len])/(window_len-1)

                return left

            else:
                # Right.
                  right = copy.deepcopy(series)
                  right[pos+window_len-1]=right[pos+1]
                  for num in range(1,window_len-1):

                      right[pos + num] =((window_len-num-1)*right[pos] + num*right[pos+1])/(window_len-1)

                  return right

              
    
    def __data_generation(self, list_IDs_temp):
          "Generates data containing batch_size samples"
          # Initialization.
          X = np.empty((10 * len(list_IDs_temp), *self.dims, self.n_channels))

          # Generate data.

          j = 0
          i=0
          for ID in list_IDs_temp:
            for i in range(10):
              X[j,] = np.expand_dims(self.df.iloc[ID, :].values, axis=-1)
              j+=1

          X_left = np.zeros((X.shape))
          X_right = np.zeros((X.shape))


          # We always do warping. To the right if odd. left if even.
          for ind in range(len(X)):
                series = X[ind].reshape((self.series_length)).tolist()
                l=bool(random.getrandbits(1))   
                warp=self.interpolation_warp(series,l)

                if l:
                    X_left[ind]=np.array(warp).reshape((self.series_length, 1))
                    X_right[ind]=X[ind]
                else:
                    X_left[ind]=X[ind]
                    X_right[ind]=np.array(warp).reshape((self.series_length, 1))


          return [X_left, X_right], [X, X, np.zeros((10 * len(list_IDs_temp)))]


class CopyGenerator(keras.utils.Sequence):
    'Generates data for vanilla generator.'
    def __init__(self, list_IDs, series_length, df, batch_size=32, n_channels=1,shuffle=True, num_warps=10, seed=1234,point=True):
        'Initialization'
        np.random.seed(seed)
        self.batch_size=batch_size
        self.batch_size_trimmed = int(self.batch_size / 2)
        self.series_length = series_length
        self.dims  = (series_length,)
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.df = df
        self.num_warps = 1
        self.on_epoch_end()
        self.point=point
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs)/self.batch_size_trimmed))
    
    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indices of the batch.
        if index == self.__len__() - 1:
            indices = self.indices[index * self.batch_size_trimmed:]
        else:
            indices = self.indices[index * self.batch_size_trimmed:(index + 1) * self.batch_size_trimmed]
        
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        
        # Generate data.
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y
    
    def on_epoch_end(self):
        "Update indexes after each epoch"
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
    
    def copy_warp(self, series, l):
        num_warpings = 1
        if self.point:
                window_len=np.random.randint(4,7)
        else:
                window_len=np.random.randint(int(0.05*self.series_length),int(0.1*self.series_length))
        pos = np.random.randint(0, self.series_length - window_len)
        
        if l:
            left = copy.deepcopy(series)
        
            for num in range(1,window_len):
                
               left[pos + num] = left[pos + window_len]
                
            return left
            
            
        else:
            
            
            # Right.
            right = copy.deepcopy(series)
            
            for num in range(1,window_len):
                
                right[pos + num] =right[pos] 
                
            return right
            
            
            
    
    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"
        # Initialization.
        X = np.empty((10 * len(list_IDs_temp), *self.dims, self.n_channels))
        
        # Generate data.
        j = 0
        i=0
        for ID in list_IDs_temp:
          for i in range(10):
            X[j,] = np.expand_dims(self.df.iloc[ID, :].values, axis=-1)
            j+=1
           
        X_left = np.zeros((X.shape))
        X_right = np.zeros((X.shape))
        
       
        # We always do warping. To the right if odd. left if even.
        for ind in range(len(X)):
              series = X[ind].reshape((self.series_length)).tolist()
              l=bool(random.getrandbits(1))   
              warp=self.copy_warp(series,l)

              if l:
                  X_left[ind]=np.array(warp).reshape((self.series_length, 1))
                  X_right[ind]=X[ind]
              else:
                  X_left[ind]=X[ind]
                  X_right[ind]=np.array(warp).reshape((self.series_length, 1))
        	
                
        #np.savetxt("warp_left.csv",np.squeeze(X_left),delimiter=',',fmt='%10.5f')
        #np.savetxt("warp_right.csv",np.squeeze(X_right),delimiter=',',fmt='%10.5f')
               
        return [X_left, X_right], [X, X, np.zeros((10 * len(list_IDs_temp)))]


class CopyInterpolationGenerator(keras.utils.Sequence):
    'Generates data for vanilla generator.'
    def __init__(self, list_IDs, series_length, df, batch_size=32, n_channels=1,shuffle=True, num_warps=10, seed=1234,point=True):
        'Initialization'
        np.random.seed(seed)
        self.batch_size=batch_size
        self.batch_size_trimmed = int(self.batch_size / 2)
        self.series_length = series_length
        self.dims  = (series_length,)
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.df = df
        self.num_warps = int(self.series_length * num_warps/ 100)
        self.on_epoch_end()
        self.point=point
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs)/self.batch_size_trimmed))
    
    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indices of the batch.
        if index == self.__len__() - 1:
            indices = self.indices[index * self.batch_size_trimmed:]
        else:
            indices = self.indices[index * self.batch_size_trimmed:(index + 1) * self.batch_size_trimmed]
        
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        
        # Generate data.
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y
    
    def on_epoch_end(self):
        "Update indexes after each epoch"
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
    
    def copy_interpolation_warp(self, series, l):


            num_warpings = 1
            if self.point:
                window_len=np.random.randint(4,7)
            else:
                window_len=np.random.randint(int(0.05*self.series_length),int(0.1*self.series_length))
            pos = np.random.randint(0, self.series_length - window_len)
			
            
            if l:
                left = copy.deepcopy(series)
                copy_warp = np.random.choice([True, False])

                if copy_warp:
                  for num in range(1,window_len):
                        left[pos + num] = left[pos + window_len]
                else:
                  # Interpolation.
                  left[pos+1]=left[pos+window_len-1]
                  for num in range(2,window_len):
                        left[pos + num] = ((window_len-num-1)*left[pos+ window_len-1] + num*left[pos+ window_len])/(window_len-1)

                return left
            else:
                right = copy.deepcopy(series)
                copy_warp = np.random.choice([True, False])

                if copy_warp:
                  for num in range(1,window_len):
                      right[pos + num] =right[pos]

                else:
                   # Interpolation.
                   right[pos+window_len-1]=right[pos+1]
                   for num in range(1,window_len-1):
                          right[pos + num] =((window_len-num-1)*right[pos] + num*right[pos+1])/(window_len-1)

                return right
    
    def __data_generation(self, list_IDs_temp):
          "Generates data containing batch_size samples"
          # Initialization.
          X = np.empty((10 * len(list_IDs_temp), *self.dims, self.n_channels))

          # Generate data.

          j = 0
          i=0
          for ID in list_IDs_temp:
            for i in range(10):
              X[j,] = np.expand_dims(self.df.iloc[ID, :].values, axis=-1)
              j+=1

          X_left = np.zeros((X.shape))
          X_right = np.zeros((X.shape))


          # We always do warping. To the right if odd. left if even.
          for ind in range(len(X)):
                series = X[ind].reshape((self.series_length)).tolist()
                l=bool(random.getrandbits(1))   
                warp=self.copy_interpolation_warp(series,l)

                if l:
                    X_left[ind]=np.array(warp).reshape((self.series_length, 1))
                    X_right[ind]=X[ind]
                else:
                    X_left[ind]=X[ind]
                    X_right[ind]=np.array(warp).reshape((self.series_length, 1))


          return [X_left, X_right], [X, X, np.zeros((10 * len(list_IDs_temp)))]
