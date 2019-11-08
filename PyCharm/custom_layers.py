from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras import constraints
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import activations


from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.eager import context
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops


class MaskedDense(layers.Layer):

    def __init__(self,
                 units,
                 activation=None,
                 kernel=None,
                 mask=None,
                 use_bias=True,
                 bias=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(MaskedDense, self).__init__(activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        self.activation = activations.get(activation)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        if kernel is None:
            # TODO: Raise real exception
            print("No weight kernel has been supplied for the custom MaskedDense-layer!")
            print("A Critical Error is imminent!")
        else:
            self.kernel = kernel
        if mask is None:
                print("No mask has been supplied for the custom MaskedDense-layer!")
                print("While the layer is still functional it's reduced to a poor version of tf.keras.layers.Dense")
        else:
            self.mask =tf.cast(mask, self.dtype)
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

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `MaskedDense` layer with non-floating point '
                            'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `MaskedDense` '
                             'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={-1: last_dim})
        self.built = True

    def call(self, inputs):
        # self.kernel = math_ops.mul(self.kernel, tf.dtypes.cast(self.mask, dtype=tf.float32))
        rank = len(inputs.shape)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, standard_ops.mul(self.kernel, self.mask), [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            inputs = math_ops.cast(inputs, self._compute_dtype)
            '''
            if K.is_sparse(inputs):
                masked_kernel = sparse_ops.sparse_tensor_dense_mul
                outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, self.kernel)
            else:
                outputs = gen_math_ops.mat_mul(inputs, self.kernel)
            '''
            outputs = gen_math_ops.mat_mul(inputs, gen_math_ops.mul(self.mask, self.kernel))
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(MaskedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

