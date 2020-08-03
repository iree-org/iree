# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf

TIME_SIZE = 2
FEATURE_SIZE = 2
BATCH_SIZE = 1


class RingBuffer(tf.Module):
  """Implements a RingBuffer."""

  def __init__(self, buffer_size, dims, dtype):
    self._buffer_size = buffer_size
    self._dims = dims

    # buffer has size [buffer_size, dims]
    # only the first dimension is used for updating buffer in a ring manner
    self._buffer = tf.Variable(
        tf.zeros((self._buffer_size,) + dims, dtype=dtype),
        trainable=False,
        name="RingBuffer")
    # Size of the data available for reading
    self._data_size = tf.Variable(
        0, trainable=False, dtype=tf.int32, name="FramerBuffer/Size")
    # The index pointing to the head of the data available for reading
    self._read_head = tf.Variable(
        0, trainable=False, dtype=tf.int32, name="FramerBuffer/Head")

  @property
  def dtype(self):
    return self._buffer.dtype

  @property
  def dims(self):
    return self._dims

  @tf.function
  def get_write_headroom(self):
    """Gets the available writable headroom.

    Returns:
      integer scalar tensor of headroom.
    """
    return self._buffer_size - self._data_size

  @tf.function
  def get_read_available(self):
    """Gets the available readable entries.

    Returns:
      integer scalar tensor of headroom.
    """
    return self._data_size

  @tf.function
  def write(self, elements):
    """Writes elements to the ringbuffer.

    Args:
      elements: Elements to write.

    Returns:
      Whether the write was successful (always True for now).
    """
    elements_size = tf.shape(elements)[0]
    start = tf.math.floormod(
        self._read_head.read_value() + self._data_size.read_value(),
        self._buffer_size)
    indices = tf.math.floormod(
        tf.range(start, limit=start + elements_size), self._buffer_size)

    tf.compat.v1.scatter_update(self._buffer, indices, elements)

    # special case when addition of new data, exceed _buffer_size:
    # we start overwriting existing data in circular manner
    # and need to update _read_head
    if tf.greater(self._data_size + elements_size, self._buffer_size):
      self._read_head.assign(
          tf.math.floormod(
              self._read_head.read_value() + self._data_size +
              tf.math.floormod(elements_size, self._buffer_size),
              self._buffer_size))

    self._data_size.assign(
        tf.minimum(self._data_size + elements_size, self._buffer_size))
    return tf.convert_to_tensor(True)

  @tf.function
  def read(self, length, offset=0, consume=True):
    """Reads elements from the ringbuffer.

    This will unconditionally read from the buffer and will produce undefined
    outputs if attempting to read past the end. This does not consume from
    the read buffer.

    Args:
      length: The length of data to read.
      offset: The offset into the readable area to begin.
      consume: Consumes the read data (default true).

    Returns:
      Tensor of elements with shape [length, dims...].
    """
    start = self._read_head + offset
    indices = tf.math.floormod(
        tf.range(start, limit=start + length), self._buffer_size)
    result = tf.gather(self._buffer, indices)
    if consume:
      self.consume(length, offset)
    return result

  @tf.function
  def consume(self, length, offset=0):
    """Consumes elements from the buffer.

    Args:
      length: The length of data to read.
      offset: The offset into the readable area to begin.
    """
    start = self._read_head + offset
    self._read_head.assign(tf.math.floormod(start + length, self._buffer_size))
    self._data_size.assign(self._data_size - length)


class StatefulRingBuffer(tf.keras.layers.Layer):

  def __init__(self, state_shape=None, consume=False, **kwargs):
    super(StatefulRingBuffer, self).__init__(**kwargs)
    self.state_shape = state_shape
    self.consume = consume

  def build(self, input_shape):
    super(StatefulRingBuffer, self).build(input_shape)
    buffer_size = self.state_shape[1]
    self.rb = RingBuffer(
        buffer_size=buffer_size, dims=(self.state_shape[2],), dtype=tf.float32)

  def call(self, inputs):
    self.rb.write(inputs)
    return self.rb.read(1, consume=self.consume)

  def get_config(self):
    config = {
        "state_shape": self.state_shape,
        "consume": self.consume,
    }
    base_config = super(StatefulRingBuffer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class StatefulRingBufferModule(tf.Module):

  def __init__(self):
    super(StatefulRingBufferModule, self).__init__()
    state_shape = [BATCH_SIZE, TIME_SIZE, FEATURE_SIZE]
    self.rb = StatefulRingBuffer(state_shape=state_shape)

  @tf.function(
      input_signature=[tf.TensorSpec([BATCH_SIZE, FEATURE_SIZE], tf.float32)])
  def predict(self, x):
    return self.rb(x)


@tf_test_utils.compile_module(
    StatefulRingBufferModule, exported_names=["predict"])
class StatefulRingBufferTest(tf_test_utils.TracedModuleTestCase):

  def test_stateful_ringbuffer(self):

    def stateful_ringbuffer(module):
      input1 = np.array([[1.0, 2.0]], dtype=np.float32)
      module.predict(input1)
      # output = np.array([[1.0, 2.0]], dtype=np.float32)

      # ring buffer is not filled yet so data from first cycle will be returned.
      input2 = np.array([[3.0, 4.0]], dtype=np.float32)
      module.predict(input2)
      # output = np.array([[1.0, 2.0]], dtype=np.float32)

      # on 3rd cycle we overwrite oldest data and return data from 2nd cycle.
      input3 = np.array([[5.0, 6.0]], dtype=np.float32)
      module.predict(input3)
      # output = np.array([[3.0, 4.0]], dtype=np.float32)

    self.compare_backends(stateful_ringbuffer)


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
