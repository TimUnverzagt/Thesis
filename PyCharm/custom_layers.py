from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import tensorflow as tf
from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.keras import constraints
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers

from tensorflow.python.keras import backend as K
from tensorflow.python.eager import context
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops


class MaskedDense(keras_layers.Dense):

    def __init__(self,
                 units,
                 initialization_weights=None,
                 initialization_biases=None,
                 mask=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(MaskedDense, self).__init__(units,
                                          activation=activation,
                                          use_bias=use_bias,
                                          kernel_initializer=kernel_initializer,
                                          bias_initializer=bias_initializer,
                                          kernel_regularizer=kernel_regularizer,
                                          bias_regularizer=bias_regularizer,
                                          activity_regularizer=activity_regularizer,
                                          kernel_constraint=kernel_constraint,
                                          bias_constraint=bias_constraint,
                                          **kwargs)

        if initialization_weights is None:
            print("No weight kernel has been supplied for the custom MaskedDense-layer!")
            print("A Critical Error is imminent!")
        else:
            self.init_kernel = initialization_weights
        if mask is None:
            print("No mask has been supplied for the custom MaskedDense-layer!")
            print("While the layer is still functional it's reduced to a poor version of tf.keras.layers.Dense")
        else:
            self.mask = tf.cast(mask, self.dtype)
        self.use_bias = use_bias
        if initialization_biases is None:
            print("No bias has been supplied for the custom MaskedDense-layer!")
            if use_bias:
                print("As the layer is set to use biases a critical Error is imminent!")
            else:
                print("As the layer is not set to use biases no problem may occur, but you are inviting problems.")
        else:
            self.init_bias = initialization_biases

    def build(self, input_shape):
        super(MaskedDense, self).build(input_shape)
        if (self.init_kernel is not None) & (self.init_bias is not None):
            self.set_weights((self.init_kernel, self.init_bias))
        elif(self.init_kernel is not None) | (self.init_bias is not None):
            print("An initial weights or biases have been supplied, but the matching counterpart is missing.")
            print("As such the given initial values are ignored!")

    def call(self, inputs):
        rank = len(inputs.shape)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, standard_ops.mul(self.kernel, self.mask), [[rank - 1], [0]])
            # outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            inputs = math_ops.cast(inputs, self._compute_dtype)
            outputs = gen_math_ops.mat_mul(inputs, gen_math_ops.mul(self.kernel, self.mask))

        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class MaskedConv2D(keras_layers.Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 initialization_weights=None,
                 initialization_bias=None,
                 mask=None,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MaskedConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        if initialization_weights is None:
            print("No weight kernel has been supplied for the custom MaskedConv2D-layer!")
            print("A Critical Error is imminent!")
        else:
            self.init_kernel = initialization_weights
        if mask is None:
            print("No mask has been supplied for the custom MaskedConv2D-layer!")
            print("While the layer is still functional it's reduced to a poor version of tf.keras.layers.Conv2D")
        else:
            self.mask = tf.cast(mask, self.dtype)
        self.use_bias = use_bias
        if initialization_bias is None:
            print("No bias has been supplied for the custom MaskedConv2D-layer!")
            if use_bias:
                print("As the layer is set to use biases a critical Error is imminent!")
            else:
                print("As the layer is not set to use biases no problem may occur, but you are inviting problems.")
        else:
            self.init_bias = initialization_bias

    def build(self, input_shape):
        super(MaskedConv2D, self).build(input_shape)
        if (self.init_kernel is not None) & (self.init_bias is not None):
            self.set_weights((self.init_kernel, self.init_bias))
        elif(self.init_kernel is not None) | (self.init_bias is not None):
            print("An initial weights or biases have been supplied, but the matching counterpart is missing.")
            print("As such the given initial values are ignored!")

    def call(self, inputs):
        outputs = self._convolution_op(inputs, gen_math_ops.mul(self.kernel, self.mask))

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class MaskedEmbedding(keras_layers.Embedding):
    def __init__(self,
                 input_dim,
                 output_dim,
                 initialization_weights=None,
                 mask=None,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 **kwargs):
        super(MaskedEmbedding, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero,
            input_length=input_length,
            **kwargs
        )

        if initialization_weights is None:
            print("No weight kernel has been supplied for the custom MaskedConv2D-layer!")
            print("A Critical Error is imminent!")
        else:
            self.init_kernel = initialization_weights
        if mask is None:
            print("No mask has been supplied for the custom MaskedConv2D-layer!")
            print("While the layer is still functional it's reduced to a poor version of tf.keras.layers.Conv2D")
        else:
            self.mask = tf.cast(mask, self.dtype)

    def build(self, input_shape):
        super(MaskedEmbedding, self).build(input_shape)
        if self.init_kernel is not None:
            self.set_weights([self.init_kernel])

    def call(self, inputs):
        dtype = K.dtype(inputs)
        if dtype != 'int32' and dtype != 'int64':
            inputs = math_ops.cast(inputs, 'int32')
        out = embedding_ops.embedding_lookup(standard_ops.mul(self.weights[0], self.mask), inputs)
        return out

