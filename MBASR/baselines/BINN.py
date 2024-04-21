from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import datetime
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from augmentation import augmentation
from evaluation import evaluate_rib, evaluate_rib_dup_nondup
import random
import collections

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.deprecation import deprecated

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

# This can be used with self.assertRaisesRegexp for assert_like_rnncell.
ASSERT_LIKE_RNNCELL_ERROR_REGEXP = "is not an RNNCell"


def _concat(prefix, suffix, static=False):
    """Concat that enables int, Tensor, or TensorShape values.

  This function takes a size specification, which can be an integer, a
  TensorShape, or a Tensor, and converts it into a concatenated Tensor
  (if static = False) or a list of integers (if static = True).

  Args:
    prefix: The prefix; usually the batch size (and/or time step size).
      (TensorShape, int, or Tensor.)
    suffix: TensorShape, int, or Tensor.
    static: If `True`, return a python list with possibly unknown dimensions.
      Otherwise return a `Tensor`.

  Returns:
    shape: the concatenation of prefix and suffix.

  Raises:
    ValueError: if `suffix` is not a scalar or vector (or TensorShape).
    ValueError: if prefix or suffix was `None` and asked for dynamic
      Tensors out.
  """
    if isinstance(prefix, ops.Tensor):
        p = prefix
        p_static = tensor_util.constant_value(prefix)
        if p.shape.ndims == 0:
            p = array_ops.expand_dims(p, 0)
        elif p.shape.ndims != 1:
            raise ValueError("prefix tensor must be either a scalar or vector, "
                             "but saw tensor: %s" % p)
    else:
        p = tensor_shape.as_shape(prefix)
        p_static = p.as_list() if p.ndims is not None else None
        p = (constant_op.constant(p.as_list(), dtype=dtypes.int32) if p.is_fully_defined() else None)
    if isinstance(suffix, ops.Tensor):
        s = suffix
        s_static = tensor_util.constant_value(suffix)
        if s.shape.ndims == 0:
            s = array_ops.expand_dims(s, 0)
        elif s.shape.ndims != 1:
            raise ValueError("suffix tensor must be either a scalar or vector, "
                             "but saw tensor: %s" % s)
    else:
        s = tensor_shape.as_shape(suffix)
        s_static = s.as_list() if s.ndims is not None else None
        s = (constant_op.constant(s.as_list(), dtype=dtypes.int32) if s.is_fully_defined() else None)

    if static:
        shape = tensor_shape.as_shape(p_static).concatenate(s_static)
        shape = shape.as_list() if shape.ndims is not None else None
    else:
        if p is None or s is None:
            raise ValueError("Provided a prefix or suffix of None: %s and %s" % (prefix, suffix))
        shape = array_ops.concat((p, s), 0)
    return shape


def _zero_state_tensors(state_size, batch_size, dtype):
    """Create tensors of zeros based on state_size, batch_size, and dtype."""

    def get_state_shape(s):
        """Combine s with batch_size to get a proper tensor shape."""
        c = _concat(batch_size, s)
        size = array_ops.zeros(c, dtype=dtype)
        if not context.executing_eagerly():
            c_static = _concat(batch_size, s, static=True)
            size.set_shape(c_static)
        return size

    return nest.map_structure(get_state_shape, state_size)


class RNNCell(base_layer.Layer):
    """Abstract object representing an RNN cell.

  Every `RNNCell` must have the properties below and implement `call` with
  the signature `(output, next_state) = call(input, state)`.  The optional
  third input argument, `scope`, is allowed for backwards compatibility
  purposes; but should be left off for new subclasses.

  This definition of cell differs from the definition used in the literature.
  In the literature, 'cell' refers to an object with a single scalar output.
  This definition refers to a horizontal array of such units.

  An RNN cell, in the most abstract setting, is anything that has
  a state and performs some operation that takes a matrix of inputs.
  This operation results in an output matrix with `self.output_size` columns.
  If `self.state_size` is an integer, this operation also results in a new
  state matrix with `self.state_size` columns.  If `self.state_size` is a
  (possibly nested tuple of) TensorShape object(s), then it should return a
  matching structure of Tensors having shape `[batch_size].concatenate(s)`
  for each `s` in `self.batch_size`.
  """

    def __init__(self, trainable=True, name=None, dtype=None, **kwargs):
        super(RNNCell, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        # Attribute that indicates whether the cell is a TF RNN cell, due the slight
        # difference between TF and Keras RNN cell.
        self._is_tf_rnn_cell = True

    def __call__(self, inputs, state, scope=None):
        """Run this RNN cell on inputs, starting from the given state.

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: if `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size, self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple
        with shapes `[batch_size, s] for s in self.state_size`.
      scope: VariableScope for the created subgraph; defaults to class name.

    Returns:
      A pair containing:

      - Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
      - New state: Either a single `2-D` tensor, or a tuple of tensors matching
        the arity and shapes of `state`.
    """
        if scope is not None:
            with vs.variable_scope(scope, custom_getter=self._rnn_get_variable) as scope:
                return super(RNNCell, self).__call__(inputs, state, scope=scope)
        else:
            scope_attrname = "rnncell_scope"
            scope = getattr(self, scope_attrname, None)
            if scope is None:
                scope = vs.variable_scope(vs.get_variable_scope(), custom_getter=self._rnn_get_variable)
                setattr(self, scope_attrname, scope)
            with scope:
                return super(RNNCell, self).__call__(inputs, state)

    def _rnn_get_variable(self, getter, *args, **kwargs):
        variable = getter(*args, **kwargs)
        if context.executing_eagerly():
            trainable = variable._trainable  # pylint: disable=protected-access
        else:
            trainable = (variable in tf_variables.trainable_variables() or (isinstance(variable, tf_variables.PartitionedVariable)
                                                                            and list(variable)[0] in tf_variables.trainable_variables()))
        if trainable and variable not in self._trainable_weights:
            self._trainable_weights.append(variable)
        elif not trainable and variable not in self._non_trainable_weights:
            self._non_trainable_weights.append(variable)
        return variable

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.

    It can be represented by an Integer, a TensorShape or a tuple of Integers
    or TensorShapes.
    """
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    def build(self, _):
        # This tells the parent Layer object that it's OK to call
        # self.add_variable() inside the call() method.
        pass

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            # Validate the given batch_size and dtype against inputs if provided.
            inputs = ops.convert_to_tensor(inputs, name="inputs")
            if batch_size is not None:
                if tensor_util.is_tensor(batch_size):
                    static_batch_size = tensor_util.constant_value(batch_size, partial=True)
                else:
                    static_batch_size = batch_size
                if inputs.shape.dims[0].value != static_batch_size:
                    raise ValueError("batch size from input tensor is different from the "
                                     "input param. Input tensor batch: {}, batch_size: {}".format(inputs.shape.dims[0].value, batch_size))

            if dtype is not None and inputs.dtype != dtype:
                raise ValueError("dtype from input tensor is different from the "
                                 "input param. Input tensor dtype: {}, dtype: {}".format(inputs.dtype, dtype))

            batch_size = inputs.shape.dims[0].value or array_ops.shape(inputs)[0]
            dtype = inputs.dtype
        if None in [batch_size, dtype]:
            raise ValueError("batch_size and dtype cannot be None while constructing initial "
                             "state: batch_size={}, dtype={}".format(batch_size, dtype))
        return self.zero_state(batch_size, dtype)

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).

    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.

    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size, state_size]` filled with zeros.

      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
      the shapes `[batch_size, s]` for each s in `state_size`.
    """
        # Try to use the last cached zero_state. This is done to avoid recreating
        # zeros, especially when eager execution is enabled.
        state_size = self.state_size
        is_eager = context.executing_eagerly()
        if is_eager and hasattr(self, "_last_zero_state"):
            (last_state_size, last_batch_size, last_dtype, last_output) = getattr(self, "_last_zero_state")
            if (last_batch_size == batch_size and last_dtype == dtype and last_state_size == state_size):
                return last_output
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            output = _zero_state_tensors(state_size, batch_size, dtype)
        if is_eager:
            self._last_zero_state = (state_size, batch_size, dtype, output)
        return output


class LayerRNNCell(RNNCell):
    """Subclass of RNNCells that act like proper `tf.Layer` objects.

  For backwards compatibility purposes, most `RNNCell` instances allow their
  `call` methods to instantiate variables via `tf.get_variable`.  The underlying
  variable scope thus keeps track of any variables, and returning cached
  versions.  This is atypical of `tf.layer` objects, which separate this
  part of layer building into a `build` method that is only called once.

  Here we provide a subclass for `RNNCell` objects that act exactly as
  `Layer` objects do.  They must provide a `build` method and their
  `call` methods do not access Variables `tf.get_variable`.
  """

    def __call__(self, inputs, state, scope=None, *args, **kwargs):
        """Run this RNN cell on inputs, starting from the given state.

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: if `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size, self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple
        with shapes `[batch_size, s] for s in self.state_size`.
      scope: optional cell scope.
      *args: Additional positional arguments.
      **kwargs: Additional keyword arguments.

    Returns:
      A pair containing:

      - Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
      - New state: Either a single `2-D` tensor, or a tuple of tensors matching
        the arity and shapes of `state`.
    """
        # Bypass RNNCell's variable capturing semantics for LayerRNNCell.
        # Instead, it is up to subclasses to provide a proper build
        # method.  See the class docstring for more details.
        return base_layer.Layer.__call__(self, inputs, state, scope=scope, *args, **kwargs)


_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class LSTMStateTuple(_LSTMStateTuple):
    """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
  and `h` is the output.

  Only used when `state_is_tuple=True`.
  """
    __slots__ = ()

    @property
    def dtype(self):
        (c, h) = self
        if c.dtype != h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" % (str(c.dtype), str(h.dtype)))
        return c.dtype


class CLSTMCell(LayerRNNCell):
    """Long short-term memory unit (LSTM) recurrent network cell.

  The default non-peephole implementation is based on:

    https://pdfs.semanticscholar.org/1154/0131eae85b2e11d53df7f1360eeb6476e7f4.pdf

  Felix Gers, Jurgen Schmidhuber, and Fred Cummins.
  "Learning to forget: Continual prediction with LSTM." IET, 850-855, 1999.

  The peephole implementation is based on:

    https://research.google.com/pubs/archive/43905.pdf

  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.

  The class uses optional peep-hole connections, optional cell clipping, and
  an optional projection layer.

  Note that this cell is not optimized for performance. Please use
  `tf.contrib.cudnn_rnn.CudnnLSTM` for better performance on GPU, or
  `tf.contrib.rnn.LSTMBlockCell` and `tf.contrib.rnn.LSTMBlockFusedCell` for
  better performance on CPU.
  """

    @deprecated(None, "This class is equivalent as tf.keras.layers.LSTMCell,"
                " and will be replaced by that in Tensorflow 2.0.")
    def __init__(self,
                 num_units,
                 use_peepholes=False,
                 cell_clip=None,
                 initializer=None,
                 num_proj=None,
                 proj_clip=None,
                 num_unit_shards=None,
                 num_proj_shards=None,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=None,
                 reuse=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        """Initialize the parameters for an LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
        provided, then the projected values are clipped elementwise to within
        `[-proj_clip, proj_clip]`.
      num_unit_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      num_proj_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training. Must set it manually to `0.0` when restoring from
        CudnnLSTM trained checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  This latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`. It
        could also be string that is within Keras activation function names.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
      dtype: Default dtype of the layer (default of `None` means use the type
        of the first input). Required when `build` is called before `call`.
      **kwargs: Dict, keyword named properties for common layer attributes, like
        `trainable` etc when constructing the cell from configs of get_config().

      When restoring from CudnnLSTM-trained checkpoints, use
      `CudnnCompatibleLSTMCell` instead.
    """
        super(CLSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if num_unit_shards is not None or num_proj_shards is not None:
            logging.warn(
                "%s: The num_unit_shards and proj_unit_shards parameters are "
                "deprecated and will be removed in Jan 2017.  "
                "Use a variable scope with a partitioner instead.", self)
        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn(
                "%s: Note that this cell is not optimized for performance. "
                "Please use tf.contrib.cudnn_rnn.CudnnLSTM for better "
                "performance on GPU.", self)

        # Inputs must be 2-dimensional.
        self.input_spec = input_spec.InputSpec(ndim=2)

        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializers.get(initializer)
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh

        if num_proj:
            self._state_size = (LSTMStateTuple(num_units, num_proj) if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (LSTMStateTuple(num_units, num_units) if state_is_tuple else 2 * num_units)
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % str(inputs_shape))

        input_depth = inputs_shape[-1]
        h_depth = self._num_units if self._num_proj is None else self._num_proj
        maybe_partitioner = (partitioned_variables.fixed_size_partitioner(self._num_unit_shards)
                             if self._num_unit_shards is not None else None)
        
        # but we also add cprev here,so h_depth double.
        self._kernel = self.add_variable(_WEIGHTS_VARIABLE_NAME,
                                         shape=[input_depth + 2 * h_depth, 4 * self._num_units],
                                         initializer=self._initializer,
                                         partitioner=maybe_partitioner)
        if self.dtype is None:
            initializer = init_ops.zeros_initializer
        else:
            initializer = init_ops.zeros_initializer(dtype=self.dtype)
        self._bias = self.add_variable(_BIAS_VARIABLE_NAME, shape=[4 * self._num_units], initializer=initializer)
        if self._use_peepholes:
            self._w_f_diag = self.add_variable("w_f_diag", shape=[self._num_units], initializer=self._initializer)
            self._w_i_diag = self.add_variable("w_i_diag", shape=[self._num_units], initializer=self._initializer)
            self._w_o_diag = self.add_variable("w_o_diag", shape=[self._num_units], initializer=self._initializer)

        if self._num_proj is not None:
            maybe_proj_partitioner = (partitioned_variables.fixed_size_partitioner(self._num_proj_shards)
                                      if self._num_proj_shards is not None else None)
            self._proj_kernel = self.add_variable("projection/%s" % _WEIGHTS_VARIABLE_NAME,
                                                  shape=[self._num_units, self._num_proj],
                                                  initializer=self._initializer,
                                                  partitioner=maybe_proj_partitioner)

        self.built = True

    def call(self, inputs, state):
        """Run one step of LSTM.

    Args:
      inputs: input Tensor, must be 2-D, `[batch, input_size]`.
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, [batch, state_size]`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.

    Returns:
      A tuple containing:

      - A `2-D, [batch, output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
        num_proj = self._num_units if self._num_proj is None else self._num_proj
        sigmoid = math_ops.sigmoid

        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])
        # cut the input into item and behavior.
        inputs, behaviors = array_ops.split(value=inputs, num_or_size_splits=2, axis=1)
        input_size = inputs.get_shape().with_rank(2).dims[1].value
        if input_size is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # add c_prev,behavior type
        lstm_matrix = math_ops.matmul(array_ops.concat([inputs, m_prev, behaviors, c_prev], 1), self._kernel)  #
        lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)  

        #new input can not calculate by lstm_matrix,becaute it doesn't take c_prev into consideration.
        j_matrix = math_ops.matmul(array_ops.concat([inputs, m_prev, behaviors], 1), self._kernel[0:3 * self._num_units])  #
        j_matrix = nn_ops.bias_add(j_matrix, self._bias)
        _, j, _, _ = array_ops.split(value=j_matrix, num_or_size_splits=4, axis=1)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, _, f, o = array_ops.split(value=lstm_matrix, num_or_size_splits=4, axis=1)
        # Diagonal connections
        if self._use_peepholes:
            c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
                 sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))  #tanh(j)
        else:
            c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) * self._activation(j))

        if self._cell_clip is not None:
            # pylint: disable=invalid-unary-operand-type
            c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
            # pylint: enable=invalid-unary-operand-type
        if self._use_peepholes:
            m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
        else:
            m = sigmoid(o) * self._activation(c)

        if self._num_proj is not None:
            m = math_ops.matmul(m, self._proj_kernel)

            if self._proj_clip is not None:
                # pylint: disable=invalid-unary-operand-type
                m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
                # pylint: enable=invalid-unary-operand-type

        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else array_ops.concat([c, m], 1))
        return m, new_state

    def get_config(self):
        config = {
            "num_units": self._num_units,
            "use_peepholes": self._use_peepholes,
            "cell_clip": self._cell_clip,
            "initializer": initializers.serialize(self._initializer),
            "num_proj": self._num_proj,
            "proj_clip": self._proj_clip,
            "num_unit_shards": self._num_unit_shards,
            "num_proj_shards": self._num_proj_shards,
            "forget_bias": self._forget_bias,
            "state_is_tuple": self._state_is_tuple,
            "activation": activations.serialize(self._activation),
            "reuse": self._reuse,
        }
        base_config = super(CLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=10000, help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='datasets/Tmall/data', help='data directory')
    # parser.add_argument('--pretrain', type=int, default=1,
    #                     help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--emb_size', type=int, default=64, help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--random_seed', default=0, type=float)
    parser.add_argument('--early_stop_epoch', default=20, type=int)
    parser.add_argument('--alpha', type=float, default=0.8, help='sub.')
    parser.add_argument('--gamma', type=float, default=0.4, help='del')
    parser.add_argument('--beta', type=float, default=0.4, help='reorder.')
    parser.add_argument('--lamda', type=float, default=0.4, help='swap behaivor.')
    parser.add_argument('--tag', type=int, default=2, help='1->del 2->sub 3->reorder')
    return parser.parse_args()


class GRUnetwork:

    def __init__(self, emb_size, learning_rate, item_num, state_size):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.behavior_num = 2
        self.item_num = int(item_num)

        self.all_embeddings = self.initialize_embeddings()

        self.item_seq = tf.placeholder(tf.int32, [None, state_size], name='item_seq')
        self.len_seq = tf.placeholder(tf.int32, [None], name='len_seq')
        self.target = tf.placeholder(tf.int32, [None], name='target')  # target item, to calculate ce loss
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.behavior_seq = tf.placeholder(tf.int32, [None, state_size])

        self.behavior_emb = tf.nn.embedding_lookup(self.all_embeddings['behavior_embeddings'], self.behavior_seq)
        self.input_emb = tf.nn.embedding_lookup(self.all_embeddings['item_embeddings'], self.item_seq)
        self.new_input_emb = tf.concat([self.input_emb, self.behavior_emb], axis=2)

        # forward clstm cell
        cell_fw = CLSTMCell(num_units=self.emb_size, state_is_tuple=True)
        # backward clstm cell
        cell_bw = CLSTMCell(num_units=self.emb_size, state_is_tuple=True)
        (self.outputs_fw, self.outputs_bw), ((_, self.outputs_state_fw), (_, self.outputs_state_bw)) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            inputs=self.new_input_emb,
            dtype=tf.float32,
            sequence_length=self.len_seq,
        )

        self.h_s = tf.concat([self.outputs_fw, self.outputs_bw], axis=2)  # (batch,input_size,2*emb_size)
        self.PBL = tf.reduce_sum(self.h_s, axis=1) / tf.reshape(tf.cast(self.len_seq, tf.float32),
                                                                (-1, 1))  # (batch,2*emb_size),
        self.SBL = self.outputs_state_fw
        # self.gru_out, (_, self.states_hidden)= tf.nn.dynamic_rnn(
        #     # tf.contrib.rnn.GRUCell(self.emb_size),
        #     CLSTMCell(num_units=self.emb_size, state_is_tuple=True),
        #     self.new_input_emb,
        #     dtype=tf.float32,
        #     sequence_length=self.len_seq,
        # )

        self.final_state = tf.concat([self.PBL, self.SBL], axis=1)
        with tf.name_scope("dropout"):
            self.final_state = tf.layers.dropout(self.final_state,
                                                 rate=args.dropout_rate,
                                                 seed=args.random_seed,
                                                 training=tf.convert_to_tensor(self.is_training))

        self.output = tf.contrib.layers.fully_connected(self.final_state, self.item_num, activation_fn=tf.nn.softmax, scope='fc')
        self.loss = tf.keras.losses.sparse_categorical_crossentropy(self.target, self.output)
        self.loss = tf.reduce_mean(self.loss)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def initialize_embeddings(self):
        all_embeddings = dict()
        item_embeddings = tf.Variable(tf.random_normal([self.item_num, self.hidden_size], 0.0, 0.01), name='item_embeddings')
        padding = tf.zeros([1, self.hidden_size], dtype=tf.float32)
        item_embeddings = tf.concat([item_embeddings, padding], axis=0)
        behavior_embeddings = tf.Variable(tf.random_normal([self.behavior_num, self.hidden_size], 0.0, 0.01), name='behavior_embeddings')
        padding = tf.zeros([1, self.hidden_size], dtype=tf.float32)
        behavior_embeddings = tf.concat([behavior_embeddings, padding], axis=0)
        all_embeddings['item_embeddings'] = item_embeddings
        all_embeddings['behavior_embeddings'] = behavior_embeddings
        return all_embeddings


if __name__ == '__main__':
    # Network parameters
    args = parse_args()
    tag, alpha, beta, gamma, lamda = args.tag, args.alpha, args.beta, args.gamma, args.lamda

    data_directory = args.data
    data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    topk = [5, 10, 20]
    # save_file = 'pretrain-GRU/%d' % (hidden_size)
    tf.reset_default_graph()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    GRUnet = GRUnetwork(emb_size=args.emb_size, learning_rate=args.lr, item_num=item_num, state_size=state_size)

    saver = tf.train.Saver(max_to_keep=10000)

    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if tag == 0:
        label = 'origin'
    elif tag == 1 or tag == 11:
        label = gamma
    elif tag == 2:
        label = alpha
    elif tag == 3 or tag == 33:
        label = beta
    elif tag == 4 or tag == 5:
        label = lamda
    
    save_dir = './model/Tmall/newBINN/7/tag_{}_param_{}_{}'.format(args.tag, label, nowTime)

    isExists = os.path.exists(save_dir)
    # if not isExists:
    #     os.makedirs(save_dir)

    data_loader = pd.read_pickle(os.path.join(data_directory, 'train.df'))
    print("data number of click :{} , data number of purchase :{}".format(
        data_loader[data_loader['is_buy'] == 0].shape[0],
        data_loader[data_loader['is_buy'] == 1].shape[0],
    ))

    total_step = 0
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        # evaluate(sess)
        num_rows = data_loader.shape[0]
        num_batches = int(num_rows / args.batch_size)
        print(num_rows, num_batches)
        best_hit_5 = -1
        count = 0
        for i in range(args.epoch):
            print(i)
            start_time_i = datetime.datetime.now()  # 

            for j in range(num_batches):
                batch = data_loader.sample(n=args.batch_size).to_dict()
                item_seq = list(batch['item_seq'].values())
                behavior_seq = list(batch['behavior_seq'].values())
                len_seq = list(batch['len_seq'].values())

                # len_seq = [len(row) for row in item_seq]
                
                len_seq = [np.sum(seq!=item_num) for seq in item_seq]
                len_seq = [ss if ss > 0 else 1 for ss in len_seq]
                # len_seq = [len(row) for row in item_seq]
                item_seq = [list(item_seq[r][:l1]) for r,l1 in enumerate(len_seq)]
                behavior_seq = [list(behavior_seq[r][:l1]) for r,l1 in enumerate(len_seq)]

                item_seq, behavior_seq, len_seq = augmentation(item_seq, behavior_seq, len_seq, item_num, state_size, tag, alpha, beta, gamma, lamda)


                target = list(batch['target'].values())

                loss, _ = sess.run(
                    [GRUnet.loss, GRUnet.opt],
                    feed_dict={
                        GRUnet.item_seq: item_seq,
                        GRUnet.len_seq: len_seq,
                        GRUnet.behavior_seq: behavior_seq,
                        GRUnet.target: target,
                        GRUnet.is_training: True
                    })
                total_step += 1
                if total_step % 200 == 0:
                    print("the loss in %dth batch is: %f" % (total_step, loss))

            over_time_i = datetime.datetime.now()  # 
            total_time_i = (over_time_i - start_time_i).total_seconds()
            print('total times: %s' % total_time_i)

            hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_rib_dup_nondup(sess,
                                                                                GRUnet,
                                                                                data_directory,
                                                                                topk,
                                                                                have_dropout=True,
                                                                                have_user_emb=False,
                                                                                is_test=True)
            if hit5 > best_hit_5:
                best_hit_5 = hit5
                count = 0
                save_root = os.path.join(
                    save_dir, 'epoch_{}_hit@5_{:.4f}_ndcg@5_{:.4f}_hit@10_{:.4f}_ndcg@10_{:.4f}_hit@20_{:.4f}_ndcg@20_{:.4f}'.format(
                        i, hit5, ndcg5, hit10, ndcg10, hit20, ndcg20))
                isExists = os.path.exists(save_root)
                if not isExists:
                    os.makedirs(save_root)
                model_name = 'binn.ckpt'
                save_root = os.path.join(save_root, model_name)
                saver.save(sess, save_root)

            else:
                count += 1
            if count == args.early_stop_epoch:
                break

    with tf.Session() as sess :
        saver.restore(sess, './model/new/RIB/emb_64_dropout_0.2_20201117_004601/epoch_23_hit@5_0.1878_ndcg@5_0.1268_hit@10_0.2686_ndcg@10_0.1530_hit@20_0.3833_ndcg@20_0.1818/gru4rec.ckpt')
        hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_rib(sess, GRUnet, data_directory, topk, have_dropout=True,
                                                                 have_user_emb=False, is_test=True)
