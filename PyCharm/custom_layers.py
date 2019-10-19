from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import activations


class MaskedDense(layers.Layer):

    def __init__(self,
                 units,
                 activation=None,
                 kernel=None,
                 mask=None,
                 use_bias=True,
                 bias=None,
                 activity_regularizer=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(MaskedDense, self).__init__(activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        self.activation = activations.get(activation)
        if kernel is None:
            # TODO: Raise real exception
            print("No weight kernel has been supplied for the custom MaskedDense-layer!")
            print("A Critical Error is imminent!")
        else:
            if mask is None:
                print("No mask has been supplied for the custom MaskedDense-layer!")
                print("While the layer is still functional it's reduced to a poor version of tf.keras.layers.Dense")
        if mask is not None:
            self.masked_kernel = tf.multiply(kernel, mask)
        else:
            self.masked_kernel = kernel
        self.use_bias = use_bias
        if bias is None:
            # TODO Raise real exception
            print("No bias has been supplied for the custom MaskedDense-layer!")
            if use_bias:
                print("As the layer is set to use biases a critical Error is imminent!")
            else:
                print("As the layer is not set to use biases no problem may occur, but you are inviting problems.")
        else:
            self.bias = bias

    def call(self, inputs):
        if self.use_bias:
            return tf.matmul(inputs, self.masked_kernel) + self.bias
        else:
            return tf.matmul(inputs, self.masked_kernel)

