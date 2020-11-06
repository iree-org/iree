# Lint as: python3
# Copyright 2020 Google LLC
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
"""Tests of tf.keras.layer.Layer subclasses."""

import os
from typing import Any, Sequence, Union

from absl import app
from absl import flags
from pyiree.tf.support import tf_test_utils
from pyiree.tf.support import tf_utils
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

DROPOUT = 0.5
DIM = 4
RANK_2_INPUT = [DIM] * 2
RANK_3_INPUT = [DIM] * 3
RANK_4_INPUT = [DIM] * 4

CONV_1D_INPUT = [2, 8, 3]
CONV_2D_INPUT = [2, 8, 8, 3]
CONV_3D_INPUT = [2, 8, 8, 8, 3]

LAYER_TO_INPUT_SHAPES = {
    'Activation': [RANK_2_INPUT],
    'ActivityRegularization': [RANK_2_INPUT],
    'Add': [RANK_2_INPUT, RANK_2_INPUT],
    'AdditiveAttention': [RANK_3_INPUT, RANK_3_INPUT, RANK_3_INPUT],
    'AlphaDropout': [RANK_2_INPUT],
    'Attention': [RANK_3_INPUT, RANK_3_INPUT, RANK_3_INPUT],
    'Average': [RANK_2_INPUT, RANK_2_INPUT],
    'AveragePooling1D': [CONV_1D_INPUT],
    'AveragePooling2D': [CONV_2D_INPUT],
    'AveragePooling3D': [CONV_3D_INPUT],
    'BatchNormalization': [RANK_2_INPUT],
    'Concatenate': [RANK_4_INPUT, RANK_4_INPUT],
    'Conv1D': [CONV_1D_INPUT],
    'Conv1DTranspose': [CONV_1D_INPUT],
    'Conv2D': [CONV_2D_INPUT],
    'Conv2DTranspose': [CONV_2D_INPUT],
    'Conv3D': [CONV_3D_INPUT],
    'Conv3DTranspose': [CONV_3D_INPUT],
    'ConvLSTM2D': [CONV_3D_INPUT],
    'Cropping1D': [CONV_1D_INPUT],
    'Cropping2D': [CONV_2D_INPUT],
    'Cropping3D': [CONV_3D_INPUT],
    'Dense': [RANK_2_INPUT],
    'DepthwiseConv2D': [CONV_2D_INPUT],
    'Dot': [RANK_3_INPUT, RANK_3_INPUT],
    'Dropout': [RANK_3_INPUT],
    'ELU': [RANK_2_INPUT],
    'Embedding': [RANK_2_INPUT],
    'Flatten': [RANK_2_INPUT],
    'GRU': [RANK_3_INPUT],
    'GRUCell': [RANK_2_INPUT, RANK_2_INPUT],
    'GaussianDropout': [RANK_2_INPUT],
    'GaussianNoise': [RANK_2_INPUT],
    'GlobalAveragePooling1D': [CONV_1D_INPUT],
    'GlobalAveragePooling2D': [CONV_2D_INPUT],
    'GlobalAveragePooling3D': [CONV_3D_INPUT],
    'GlobalMaxPool1D': [CONV_1D_INPUT],
    'GlobalMaxPool2D': [CONV_2D_INPUT],
    'GlobalMaxPool3D': [CONV_3D_INPUT],
    'InputLayer': [RANK_2_INPUT],
    'LSTM': [RANK_3_INPUT],
    'LSTMCell': [RANK_2_INPUT, RANK_2_INPUT],
    'Lambda': [RANK_2_INPUT],
    'LayerNormalization': [RANK_2_INPUT],
    'LeakyReLU': [RANK_2_INPUT],
    'LocallyConnected1D': [CONV_1D_INPUT],
    'LocallyConnected2D': [CONV_2D_INPUT],
    'Masking': [RANK_2_INPUT],
    'MaxPool1D': [CONV_1D_INPUT],
    'MaxPool2D': [CONV_2D_INPUT],
    'MaxPool3D': [CONV_3D_INPUT],
    'Maximum': [RANK_2_INPUT, RANK_2_INPUT],
    'Minimum': [RANK_2_INPUT, RANK_2_INPUT],
    'MultiHeadAttention': [RANK_3_INPUT, RANK_3_INPUT],
    'Multiply': [RANK_2_INPUT, RANK_2_INPUT],
    'PReLU': [RANK_2_INPUT],
    'Permute': [RANK_4_INPUT],
    'ReLU': [RANK_2_INPUT],
    'RepeatVector': [RANK_2_INPUT],
    'Reshape': [RANK_3_INPUT],
    'SeparableConv1D': [CONV_1D_INPUT],
    'SeparableConv2D': [CONV_2D_INPUT],
    'SimpleRNN': [RANK_3_INPUT],
    'SimpleRNNCell': [RANK_2_INPUT, RANK_2_INPUT],
    'Softmax': [RANK_2_INPUT],
    'SpatialDropout1D': [CONV_1D_INPUT],
    'SpatialDropout2D': [CONV_2D_INPUT],
    'SpatialDropout3D': [CONV_3D_INPUT],
    'Subtract': [RANK_2_INPUT, RANK_2_INPUT],
    'ThresholdedReLU': [RANK_2_INPUT],
    'UpSampling1D': [CONV_1D_INPUT],
    'UpSampling2D': [CONV_2D_INPUT],
    'UpSampling3D': [CONV_3D_INPUT],
    'ZeroPadding1D': [CONV_1D_INPUT],
    'ZeroPadding2D': [CONV_2D_INPUT],
    'ZeroPadding3D': [CONV_3D_INPUT],
}

LAYER_TO_KWARGS = {
    'Activation': dict(activation='relu'),
    'ActivityRegularization': dict(),
    'Add': dict(),
    'AdditiveAttention': dict(),
    'AlphaDropout': dict(rate=DROPOUT),
    'Attention': dict(),
    'Average': dict(),
    'AveragePooling1D': dict(),
    'AveragePooling2D': dict(),
    'AveragePooling3D': dict(),
    'BatchNormalization': dict(),
    'Concatenate': dict(),
    'Conv1D': dict(filters=4, kernel_size=3),
    'Conv1DTranspose': dict(filters=4, kernel_size=3),
    'Conv2D': dict(filters=4, kernel_size=3),
    'Conv2DTranspose': dict(filters=4, kernel_size=3),
    'Conv3D': dict(filters=4, kernel_size=3),
    'Conv3DTranspose': dict(filters=4, kernel_size=3),
    'ConvLSTM2D': dict(filters=4, kernel_size=3),
    'Cropping1D': dict(cropping=2),
    'Cropping2D': dict(cropping=2),
    'Cropping3D': dict(cropping=2),
    'Dense': dict(units=4),
    'DepthwiseConv2D': dict(kernel_size=3),
    'Dot': dict(axes=(1, 2)),
    'Dropout': dict(rate=DROPOUT),
    'ELU': dict(),
    'Embedding': dict(input_dim=4, output_dim=2),
    'Flatten': dict(),
    'GRU': dict(units=4, return_sequences=True),
    'GRUCell': dict(units=4),
    'GaussianDropout': dict(rate=DROPOUT),
    'GaussianNoise': dict(stddev=1.0),
    'GlobalAveragePooling1D': dict(),
    'GlobalAveragePooling2D': dict(),
    'GlobalAveragePooling3D': dict(),
    'GlobalMaxPool1D': dict(),
    'GlobalMaxPool2D': dict(),
    'GlobalMaxPool3D': dict(),
    'InputLayer': dict(),
    'LSTM': dict(units=4, return_sequences=True),
    'LSTMCell': dict(units=4),
    'Lambda': dict(function=lambda x: x**2),
    'Layer': dict(),
    'LayerNormalization': dict(),
    'LeakyReLU': dict(),
    'LocallyConnected1D': dict(filters=4, kernel_size=3),
    'LocallyConnected2D': dict(filters=4, kernel_size=3),
    'Masking': dict(),
    'MaxPool1D': dict(),
    'MaxPool2D': dict(),
    'MaxPool3D': dict(),
    'Maximum': dict(),
    'Minimum': dict(),
    'MultiHeadAttention': dict(num_heads=2, key_dim=3),
    'Multiply': dict(),
    'PReLU': dict(),
    'Permute': dict(dims=(3, 1, 2)),
    'ReLU': dict(),
    'RepeatVector': dict(n=3),
    'Reshape': dict(target_shape=[1, 1, 1] + RANK_3_INPUT[1:]),
    'SeparableConv1D': dict(filters=4, kernel_size=3),
    'SeparableConv2D': dict(filters=4, kernel_size=3),
    'SimpleRNN': dict(units=4, return_sequences=True),
    'SimpleRNNCell': dict(units=4),
    'Softmax': dict(),
    'SpatialDropout1D': dict(rate=DROPOUT),
    'SpatialDropout2D': dict(rate=DROPOUT),
    'SpatialDropout3D': dict(rate=DROPOUT),
    'Subtract': dict(),
    'ThresholdedReLU': dict(),
    'UpSampling1D': dict(),
    'UpSampling2D': dict(),
    'UpSampling3D': dict(),
    'ZeroPadding1D': dict(),
    'ZeroPadding2D': dict(),
    'ZeroPadding3D': dict(),
}

# Layers that allow specifying the 'dropout' kwarg.
DROPOUT_LAYERS = [
    'AdditiveAttention', 'Attention', 'ConvLSTM2D', 'GRU', 'GRUCell', 'LSTM',
    'LSTMCell', 'MultiHeadAttention', 'SimpleRNN', 'SimpleRNNCell'
]

flags.DEFINE_string('layer', 'Dense',
                    f'One of {list(LAYER_TO_INPUT_SHAPES.keys())}.')
flags.DEFINE_bool(
    'dynamic_batch', False,
    'Whether or not to compile the layer with a dynamic batch size.')
flags.DEFINE_bool('training', False,
                  'Whether or not to compile the layer in training mode.')


def get_input(shape: Sequence[int]) -> tf.keras.layers.Input:
  batch_size = None if FLAGS.dynamic_batch else shape[0]
  return tf.keras.layers.Input(batch_size=batch_size, shape=shape[1:])


def normalize(inputs: Sequence[Any]) -> Union[Any, Sequence[Any]]:
  """Unpacks inputs if it has length one."""
  return inputs[0] if len(inputs) == 1 else inputs


class KerasLayersModule(tf.Module):

  def __init__(self):
    super().__init__()
    layer_class = getattr(tf.keras.layers, FLAGS.layer)
    input_shapes = LAYER_TO_INPUT_SHAPES[FLAGS.layer]
    kwargs = LAYER_TO_KWARGS[FLAGS.layer]
    if FLAGS.training and FLAGS.layer in DROPOUT_LAYERS:
      kwargs['dropout'] = DROPOUT

    # Create a wrapped keras layer.
    inputs = normalize([get_input(shape) for shape in input_shapes])
    if FLAGS.layer == 'MultiHeadAttention':
      outputs = layer_class(**kwargs)(*inputs)
    else:
      outputs = layer_class(**kwargs)(inputs)
    self.m = tf.keras.Model(inputs, outputs)

    # Wrap the layer in a tf.function.
    if FLAGS.dynamic_batch:
      input_shapes = [[None] + shape[1:] for shape in input_shapes]
    input_signature = [tf.TensorSpec(shape) for shape in input_shapes]
    if len(input_signature) > 1:
      input_signature = [input_signature]
    self.call = tf.function(input_signature=input_signature)(
        lambda x: self.m(x, training=FLAGS.training))


class KerasLayersTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(KerasLayersModule,
                                                    exported_names=['call'])

  def test_call(self):

    def call(module):
      input_shapes = LAYER_TO_INPUT_SHAPES[FLAGS.layer]
      inputs = normalize([tf_utils.uniform(shape) for shape in input_shapes])
      module.call(inputs)

    self.compare_backends(call, self._modules)


def main(argv):
  del argv  # Unused.
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()

  if FLAGS.layer not in LAYER_TO_INPUT_SHAPES:
    raise ValueError(f"Unrecognized layer: '{FLAGS.layer}'.")
  dynamic_batch_str = 'dynamic_batch' if FLAGS.dynamic_batch else 'static_batch'
  KerasLayersModule.__name__ = os.path.join('keras_layers', FLAGS.layer,
                                            dynamic_batch_str)

  tf.test.main()


if __name__ == '__main__':
  app.run(main)
