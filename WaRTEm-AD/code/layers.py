from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils

class MaxPoolingWithArgmax1D(Layer):

    def __init__(
            self,
            pool_size=2,
            strides=None,
            padding='same',
            **kwargs):
        super(MaxPoolingWithArgmax1D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = pool_size if strides is None else strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, 1, pool_size, 1] #TODO accommodate different data_formats.
            padding = padding.upper()
            strides = [1, 1, strides, 1] #TODO accommodate different data_formats.
            expanding_dim = 1 #TODO accommodate different data_formats.
            inputs = tf.expand_dims(inputs, expanding_dim)
            output, argmax = tf.nn.max_pool_with_argmax(
                    inputs,
                    ksize=ksize,
                    strides=strides,
                    padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(
                    K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        output = tf.squeeze(output, expanding_dim)
        argmax = tf.squeeze(argmax, expanding_dim)
        return [output, argmax]
    
    def get_config(self):
        config = super(MaxPoolingWithArgmax1D, self).get_config()
        config.update({'padding':self.padding, 'pool_size':self.pool_size, 'strides':self.strides})
        return config

    def compute_output_shape(self, input_shape):
        #ratio = (1, 2, 2, 1)
        
#        output_shape = [
#                dim//ratio[idx]
#                if dim is not None else None
#                for idx, dim in enumerate(input_shape)]
        
        steps = input_shape[1] #TODO accommodate different data_formats.
        channels = input_shape[2] #TODO accommodate different data_formats.
        length = conv_utils.conv_output_length(steps,
                                               self.pool_size,
                                               self.padding,
                                               self.strides)
        output_shape = tuple([input_shape[0], length, channels]) #TODO accommodate different data_formats.
        #output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]
    
    
class MaxUnpooling1D(Layer):
    def __init__(self, size=2, **kwargs):
        super(MaxUnpooling1D, self).__init__(**kwargs)
        self.size = int(size)

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with tf.compat.v1.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = tf.shape(input=updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                        input_shape[0],
                        input_shape[1]*self.size,
                        input_shape[2])
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate(
                    [[input_shape[0]], [1], [1]],
                    axis=0)
            batch_range = K.reshape(
                    tf.range(output_shape[0], dtype='int32'),
                    shape=batch_shape)
            b = one_like_mask * batch_range
            #y = mask // (output_shape[2] * output_shape[3])
            #x = (mask // output_shape[3]) % output_shape[2]
            x = (mask // output_shape[2]) % output_shape[1]
            feature_range = tf.range(output_shape[2], dtype='int32')
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = tf.size(input=updates)
            indices = K.transpose(K.reshape(
                K.stack([b, x, f]),
                [3, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
                mask_shape[0],
                mask_shape[1]*self.size,
                mask_shape[2]
                )
    def get_config(self):
        config = super(MaxUnpooling1D, self).get_config()
        config.update({'size':self.size})
        return config