"""
Custom RNN and bidirectional_RNN functions, modified for working with attention models when the encoder is bidirectional
"""

from tensorflow.python.ops.rnn import _reverse_seq, _rnn_step
import tensorflow as tf
import numpy as np
import ipdb

SEED = 42
np.random.seed(SEED)


# modified to retrieve c1,h1 to feed the decoder initial states
def custom_rnn(cell, inputs, initial_state=None, dtype=None,
        sequence_length=None, scope=None):
  """Creates a recurrent neural network specified by RNNCell "cell".

  The simplest form of RNN network generated is:
    state = cell.zero_state(...)
    outputs = []
    for input_ in inputs:
      output, state = cell(input_, state)
      outputs.append(output)
    return (outputs, state)

  However, a few other options are available:

  An initial state can be provided.
  If the sequence_length vector is provided, dynamic calculation is performed.
  This method of calculation does not compute the RNN steps past the maximum
  sequence length of the minibatch (thus saving computational time),
  and properly propagates the state at an example's sequence length
  to the final state output.

  The dynamic calculation performed is, at time t for batch row b,
    (output, state)(b, t) =
      (t >= sequence_length(b))
        ? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
        : cell(input(b, t), state(b, t - 1))

  Args:
    cell: An instance of RNNCell.
    inputs: A length T list of inputs, each a tensor of shape
      [batch_size, cell.input_size].
    initial_state: (optional) An initial state for the RNN.  This must be
      a tensor of appropriate type and shape [batch_size x cell.state_size].
    dtype: (optional) The data type for the initial state.  Required if
      initial_state is not provided.
    sequence_length: Specifies the length of each sequence in inputs.
      An int32 or int64 vector (tensor) size [batch_size].  Values in [0, T).
    scope: VariableScope for the created subgraph; defaults to "RNN".

  Returns:
    A pair (outputs, state) where:
      outputs is a length T list of outputs (one for each input)
      state is the final state

  Raises:
    TypeError: If "cell" is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """

  if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
    raise TypeError("cell must be an instance of RNNCell")
  if not isinstance(inputs, list):
    raise TypeError("inputs must be a list")
  if not inputs:
    raise ValueError("inputs must not be empty")

  outputs = []
  # Create a new scope in which the caching device is either
  # determined by the parent scope, or is set to place the cached
  # Variable using the same placement as for the rest of the RNN.
  with tf.variable_scope(scope or "RNN") as varscope:
    fixed_batch_size = inputs[0].get_shape().with_rank_at_least(1)[0]
    if fixed_batch_size.value:
      batch_size = fixed_batch_size.value
    else:
      batch_size = array_ops.shape(inputs[0])[0]
    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If no initial_state is provided, dtype must be.")
      state = cell.zero_state(batch_size, dtype)

    if sequence_length is not None:
      sequence_length = tf.to_int32(sequence_length)

    if sequence_length:  # Prepare variables
      zero_output = tf.zeros(
          tf.pack([batch_size, cell.output_size]), inputs[0].dtype)
      zero_output.set_shape(
          tf.TensorShape([fixed_batch_size.value, cell.output_size]))
      min_sequence_length = tf.reduce_min(sequence_length)
      max_sequence_length = tf.reduce_max(sequence_length)

    c1h1 = []

    for time, input_ in enumerate(inputs):
      if time > 0: tf.get_variable_scope().reuse_variables()
      # pylint: disable=cell-var-from-loop
      call_cell = lambda: cell(input_, state)
      # pylint: enable=cell-var-from-loop
      if sequence_length:
        (output, state) = _rnn_step(
            time, sequence_length, min_sequence_length, max_sequence_length,
            zero_output, state, call_cell)
      else:
        (output, state) = call_cell()
      if time==0:
        c1h1 = state
      outputs.append(output)

    return (outputs, state, c1h1)





# copy of tensorflow.tensorflow.python.ops.rnn / modified so it retrieves last memory cell as well
def bidirectional_rnn(cell_fw, cell_bw, inputs,
                      initial_state_fw=None, initial_state_bw=None,
                      dtype=None, sequence_length=None, scope=None):
  """Creates a bidirectional recurrent neural network.

  Similar to the unidirectional case above (rnn) but takes input and builds
  independent forward and backward RNNs with the final forward and backward
  outputs depth-concatenated, such that the output will have the format
  [time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
  forward and backward cell must match. The initial state for both directions
  is zero by default (but can be set optionally) and no intermediate states are
  ever returned -- the network is fully unrolled for the given (passed in)
  length(s) of the sequence(s) or completely unrolled if length(s) is not given.

  Args:
    cell_fw: An instance of RNNCell, to be used for forward direction.
    cell_bw: An instance of RNNCell, to be used for backward direction.
    inputs: A length T list of inputs, each a tensor of shape
      [batch_size, cell.input_size].
    initial_state_fw: (optional) An initial state for the forward RNN.
      This must be a tensor of appropriate type and shape
      [batch_size x cell.state_size].
    initial_state_bw: (optional) Same as for initial_state_fw.
    dtype: (optional) The data type for the initial state.  Required if either
      of the initial states are not provided.
    sequence_length: (optional) An int32/int64 vector, size [batch_size],
      containing the actual lengths for each of the sequences.
    scope: VariableScope for the created subgraph; defaults to "BiRNN"

  Returns:
    A set of output `Tensors` where:
      outputs is a length T list of outputs (one for each input), which
      are depth-concatenated forward and backward outputs

  Raises:
    TypeError: If "cell_fw" or "cell_bw" is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """

  if not isinstance(cell_fw, tf.nn.rnn_cell.RNNCell):
    raise TypeError("cell_fw must be an instance of RNNCell")
  if not isinstance(cell_bw, tf.nn.rnn_cell.RNNCell):
    raise TypeError("cell_bw must be an instance of RNNCell")
  if not isinstance(inputs, list):
    raise TypeError("inputs must be a list")
  if not inputs:
    raise ValueError("inputs must not be empty")

  name = scope or "BiRNN"
  # Forward direction
  with tf.variable_scope(name + "_FW") as fw_scope:
    output_fw, _ = tf.nn.rnn(cell_fw, inputs, initial_state_fw, dtype,
                       		  sequence_length, scope=fw_scope)

  # Backward direction
  with tf.variable_scope(name + "_BW") as bw_scope:
    tmp,_,c1h1 = custom_rnn(cell_bw, _reverse_seq(inputs, sequence_length),
                 				  initial_state_bw, dtype, sequence_length, scope=bw_scope)
  output_bw = _reverse_seq(tmp, sequence_length)
  # Concat each of the forward/backward outputs
  outputs = [tf.concat(1, [fw, bw])
             for fw, bw in zip(output_fw, output_bw)]
  # get last memory state of bw cell
  c1,h1 = tf.split(1,2,c1h1)	# turns out that final_state is [c,h]
  return outputs,c1,h1





class CustomLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
  def __init__(self, num_units, forget_bias=1.0, input_size=None):
    tf.nn.rnn_cell.BasicLSTMCell.__init__(self,num_units,forget_bias,input_size)

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):  # "CustomLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      c, h = tf.split(1, 2, state)
      concat = linear([inputs, h], 4 * self._num_units, True)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(1, 4, concat)

      new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
      new_h = tf.tanh(new_c) * tf.sigmoid(o)

    return new_h, tf.concat(1, [new_c, new_h])



def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  assert args
  if not isinstance(args, (list, tuple)):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix",
                              shape = (total_arg_size,output_size),
                              initializer = weight_initializer)
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size],
        initializer=tf.constant_initializer(bias_start))
  return res + bias_term



def weight_initializer(shape,dtype):
    _u,_,_v = np.linalg.svd(np.random.random(size=shape),full_matrices=False )
    weight_init = []
    if _u.shape==tuple(shape):
      weight_init = _u
    else:
      weight_init = _v
    dtype_np = np.float32 if dtype==tf.float32 else np.float64

    return tf.constant(np.array(weight_init,dtype=dtype_np))
