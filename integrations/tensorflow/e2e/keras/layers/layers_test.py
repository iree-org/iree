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

import collections
import copy
import os
from typing import Any, Dict, Sequence, Union

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

# Configs are namedtuples storing keyword arguments and shapes to test a
# tf.keras.layers.Layer with. They are used in two ways:
#   1. To directly specify the kwargs and shapes for a layers test.
#   2. In 'generate_configs', to specify how to change a default config to
#      specify a non-default test. In this case, the overriding Config will
#      exclusively specify the shape of the test if its shape is not None, and
#      the overriding Config will extend/update the kwargs of the default
#      Config.
Config = collections.namedtuple('Config', ['kwargs', 'shapes'])
# Use old default API for compatibility with Python 3.6.
Config.__new__.__defaults__ = (dict(), None)


def generate_configs(default_config: Config,
                     override_configs: Dict[str, Config]) -> Dict[str, Config]:
  """Generates a dict of 'Config's based off changes to a default Config."""
  configs = {'default': default_config}
  for exported_name, config in override_configs.items():
    shapes = default_config.shapes if config.shapes is None else config.shapes

    # Deep copy to avoid inplace mutation of the default.
    kwargs = copy.deepcopy(default_config.kwargs)
    kwargs.update(config.kwargs)  # Adds new and overwrites old kwargs.

    configs[exported_name] = Config(kwargs, shapes)
  return configs


# A dict mapping tf.keras.layers names to either a single Config (representing
# the kwargs and shapes to use to test a Layer) or a dict mapping exported_names
# to Configs. The latter case is usually automatically generated via
# 'generate_configs', with the 'Config's in 'override_configs' specifying how
# to modify the 'default_config's kwargs and shapes.
#
# Each entry will be normalized to be a dict mapping exported_names to Configs,
# with a default exported_name of 'default'.
LAYER_TO_UNITTEST_CONFIGURATIONS = {
    'Activation':
        Config(dict(activation='relu'), [RANK_2_INPUT]),
    'ActivityRegularization':
        Config(dict(l1=0.1, l2=0.1), shapes=[RANK_2_INPUT]),
    'Add':
        Config(shapes=[RANK_2_INPUT, RANK_2_INPUT]),
    'AdditiveAttention':
        generate_configs(
            default_config=Config(
                shapes=[RANK_3_INPUT, RANK_3_INPUT, RANK_3_INPUT],),
            override_configs={
                'causal': Config(dict(causal=True)),
            },
        ),
    'AlphaDropout':
        Config(dict(rate=DROPOUT), [RANK_2_INPUT]),
    'Attention':
        generate_configs(
            default_config=Config(
                shapes=[RANK_3_INPUT, RANK_3_INPUT, RANK_3_INPUT],),
            override_configs={
                'causal': Config(dict(causal=True)),
            },
        ),
    'Average':
        Config(shapes=[RANK_2_INPUT, RANK_2_INPUT]),
    'AveragePooling1D':
        generate_configs(
            default_config=Config(shapes=[CONV_1D_INPUT]),
            override_configs={
                'strides': Config(dict(strides=3)),
                'padding_same': Config(dict(padding='same')),
                'channels_first': Config(dict(data_format='channels_first')),
            },
        ),
    'AveragePooling2D':
        generate_configs(
            default_config=Config(shapes=[CONV_2D_INPUT]),
            override_configs={
                'strides': Config(dict(strides=3)),
                'padding_same': Config(dict(padding='same')),
                # TF: Default AvgPoolingOp only supports NHWC on device type CPU
                # 'channels_first': Config(dict(data_format='channels_first')),
            },
        ),
    'AveragePooling3D':
        generate_configs(
            default_config=Config(shapes=[CONV_3D_INPUT]),
            override_configs={
                'strides': Config(dict(strides=3)),
                'padding_same': Config(dict(padding='same')),
                'channels_first': Config(dict(data_format='channels_first')),
            },
        ),
    'BatchNormalization':
        generate_configs(
            default_config=Config(shapes=[RANK_2_INPUT]),
            override_configs={'renorm': Config(dict(renorm=True))},
        ),
    'Concatenate':
        Config(shapes=[RANK_4_INPUT, RANK_4_INPUT]),
    'Conv1D':
        generate_configs(
            default_config=Config(
                kwargs=dict(filters=4, kernel_size=3),
                shapes=[CONV_1D_INPUT],
            ),
            override_configs={
                'strides': Config(dict(strides=3)),
                'padding_same': Config(dict(padding='same')),
                # TF: The Conv2D op currently only supports the NHWC tensor
                #     format on the CPU.
                # 'channels_first': Config(dict(data_format='channels_first')),
                'dilation_rate': Config(dict(dilation_rate=3)),
            },
        ),
    'Conv1DTranspose':
        generate_configs(
            default_config=Config(
                kwargs=dict(filters=4, kernel_size=3),
                shapes=[CONV_1D_INPUT],
            ),
            override_configs={
                'strides': Config(dict(strides=3)),
                'padding_same': Config(dict(padding='same')),
                # TF: Conv2DCustomBackpropInputOp only supports NHWC
                # 'channels_first': Config(dict(data_format='channels_first')),
                # TF: Current libxsmm and customized CPU implementations do not
                # yet support dilation rates larger than 1.
                # 'dilation_rate': Config(dict(dilation_rate=3)),
            },
        ),
    'Conv2D':
        generate_configs(
            default_config=Config(
                kwargs=dict(filters=4, kernel_size=3),
                shapes=[CONV_2D_INPUT],
            ),
            override_configs={
                'strides': Config(dict(strides=3)),
                'padding_same': Config(dict(padding='same')),
                # TF: The Conv2D op currently only supports the NHWC tensor
                #     format on the CPU.
                # 'channels_first': Config(dict(data_format='channels_first')),
                'dilation_rate': Config(dict(dilation_rate=3)),
            },
        ),
    'Conv2DTranspose':
        generate_configs(
            default_config=Config(
                kwargs=dict(filters=4, kernel_size=3),
                shapes=[CONV_2D_INPUT],
            ),
            override_configs={
                'strides': Config(dict(strides=3)),
                'padding_same': Config(dict(padding='same')),
                'channels_first': Config(dict(data_format='channels_first')),
                'dilation_rate': Config(dict(dilation_rate=3)),
            },
        ),
    'Conv3D':
        generate_configs(
            default_config=Config(
                kwargs=dict(filters=4, kernel_size=3),
                shapes=[CONV_3D_INPUT],
            ),
            override_configs={
                'strides': Config(dict(strides=3)),
                'padding_same': Config(dict(padding='same')),
                # TF: The Conv3D op currently only supports the NHWC tensor
                #     format on the CPU.
                # 'channels_first': Config(dict(data_format='channels_first')),
                'dilation_rate': Config(dict(dilation_rate=3)),
            },
        ),
    'Conv3DTranspose':
        generate_configs(
            default_config=Config(
                kwargs=dict(filters=4, kernel_size=3),
                shapes=[CONV_3D_INPUT],
            ),
            override_configs={
                'strides': Config(dict(strides=3)),
                'padding_same': Config(dict(padding='same')),
                # TF: Conv3DBackpropInputOpV2 only supports NDHWC on the CPU.
                # 'channels_first': Config(dict(data_format='channels_first')),
                'dilation_rate': Config(dict(dilation_rate=3)),
            },
        ),
    'ConvLSTM2D':
        generate_configs(
            default_config=Config(
                kwargs=dict(filters=4, kernel_size=3, return_state=True),
                shapes=[CONV_3D_INPUT],
            ),
            override_configs={
                'strides': Config(dict(strides=3)),
                'padding': Config(dict(padding='same')),
                'channels_first': Config(dict(data_format='channels_first')),
                'dilation_rate': Config(dict(dilation_rate=3)),
                'go_backwards': Config(dict(go_backwards=True)),
                'stateful': Config(dict(stateful=True)),
            },
        ),
    'Cropping1D':
        Config(dict(cropping=2), [CONV_1D_INPUT]),
    'Cropping2D':
        Config(dict(cropping=2), [CONV_2D_INPUT]),
    'Cropping3D':
        Config(dict(cropping=2), [CONV_3D_INPUT]),
    'Dense':
        Config(dict(units=4), [RANK_2_INPUT]),
    'DepthwiseConv2D':
        generate_configs(
            default_config=Config(
                kwargs=dict(kernel_size=3),
                shapes=[CONV_2D_INPUT],
            ),
            override_configs={
                'strides': Config(dict(strides=3)),
                'padding_same': Config(dict(padding='same')),
                'channels_first': Config(dict(data_format='channels_first')),
                'depth_multiplier': Config(dict(depth_multiplier=2)),
                'dilation_rate': Config(dict(dilation_rate=2)),
            },
        ),
    'Dot':
        Config(dict(axes=(1, 2)), [RANK_3_INPUT, RANK_3_INPUT]),
    'Dropout':
        Config(dict(rate=DROPOUT), [RANK_3_INPUT]),
    'ELU':
        Config(shapes=[RANK_2_INPUT]),
    'Embedding':
        Config(dict(input_dim=4, output_dim=2), [RANK_2_INPUT]),
    'Flatten':
        Config(shapes=[RANK_2_INPUT]),
    'GRU':
        generate_configs(
            default_config=Config(
                kwargs=dict(units=4, return_sequences=True),
                shapes=[RANK_3_INPUT],
            ),
            override_configs={
                'implementation_1': Config(dict(implementation=1)),
                'go_backwards': Config(dict(go_backwards=True)),
                'time_major': Config(dict(time_major=True)),
                'stateful': Config(dict(stateful=True)),
            },
        ),
    'GRUCell':
        Config(dict(units=4), [RANK_2_INPUT, RANK_2_INPUT]),
    'GaussianDropout':
        Config(dict(rate=DROPOUT), [RANK_2_INPUT]),
    'GaussianNoise':
        Config(dict(stddev=1.0), [RANK_2_INPUT]),
    'GlobalAveragePooling1D':
        generate_configs(
            default_config=Config(shapes=[CONV_1D_INPUT]),
            override_configs={
                'channels_first': Config(dict(data_format='channels_first')),
            },
        ),
    'GlobalAveragePooling2D':
        generate_configs(
            default_config=Config(shapes=[CONV_2D_INPUT]),
            override_configs={
                'channels_first': Config(dict(data_format='channels_first')),
            },
        ),
    'GlobalAveragePooling3D':
        generate_configs(
            default_config=Config(shapes=[CONV_3D_INPUT]),
            override_configs={
                'channels_first': Config(dict(data_format='channels_first')),
            },
        ),
    'GlobalMaxPool1D':
        generate_configs(
            default_config=Config(shapes=[CONV_1D_INPUT]),
            override_configs={
                'channels_first': Config(dict(data_format='channels_first')),
            },
        ),
    'GlobalMaxPool2D':
        generate_configs(
            default_config=Config(shapes=[CONV_2D_INPUT]),
            override_configs={
                'channels_first': Config(dict(data_format='channels_first')),
            },
        ),
    'GlobalMaxPool3D':
        generate_configs(
            default_config=Config(shapes=[CONV_3D_INPUT]),
            override_configs={
                'channels_first': Config(dict(data_format='channels_first')),
            },
        ),
    'InputLayer':
        Config(shapes=[RANK_2_INPUT]),
    'LSTM':
        generate_configs(
            default_config=Config(
                kwargs=dict(units=4, return_sequences=True),
                shapes=[RANK_3_INPUT],
            ),
            override_configs={
                'implementation_1': Config(dict(implementation=1)),
                'go_backwards': Config(dict(go_backwards=True)),
                'time_major': Config(dict(time_major=True)),
                'stateful': Config(dict(stateful=True)),
            },
        ),
    'LSTMCell':
        Config(dict(units=4), [RANK_2_INPUT, RANK_2_INPUT]),
    'Lambda':
        Config(dict(function=lambda x: x**2), [RANK_2_INPUT]),
    'LayerNormalization':
        Config(shapes=[RANK_2_INPUT]),
    'LeakyReLU':
        Config(shapes=[RANK_2_INPUT]),
    'LocallyConnected1D':
        generate_configs(
            default_config=Config(
                kwargs=dict(filters=4, kernel_size=3),
                shapes=[CONV_1D_INPUT],
            ),
            override_configs={
                'strides': Config(dict(strides=3)),
                'padding_same': Config(dict(padding='same', implementation=2)),
                'channels_first': Config(dict(data_format='channels_first')),
                'sparse_implementation': Config(dict(implementation=3)),
            },
        ),
    'LocallyConnected2D':
        generate_configs(
            default_config=Config(
                kwargs=dict(filters=4, kernel_size=3),
                shapes=[CONV_2D_INPUT],
            ),
            override_configs={
                'strides': Config(dict(strides=3)),
                'padding_same': Config(dict(padding='same', implementation=2)),
                'channels_first': Config(dict(data_format='channels_first')),
                'sparse_implementation': Config(dict(implementation=3)),
            },
        ),
    'Masking':
        Config(shapes=[RANK_2_INPUT]),
    'MaxPool1D':
        generate_configs(
            default_config=Config(shapes=[CONV_1D_INPUT]),
            override_configs={
                'strides': Config(dict(strides=3)),
                'padding_same': Config(dict(padding='same')),
                'channels_first': Config(dict(data_format='channels_first')),
            },
        ),
    'MaxPool2D':
        generate_configs(
            default_config=Config(shapes=[CONV_2D_INPUT]),
            override_configs={
                'strides': Config(dict(strides=3)),
                'padding_same': Config(dict(padding='same')),
                # TF: Default MaxPoolingOp only supports NHWC on device type CPU
                # 'channels_first': Config(dict(data_format='channels_first')),
            },
        ),
    'MaxPool3D':
        generate_configs(
            default_config=Config(shapes=[CONV_3D_INPUT]),
            override_configs={
                'strides': Config(dict(strides=3)),
                'padding_same': Config(dict(padding='same')),
                'channels_first': Config(dict(data_format='channels_first')),
            },
        ),
    'Maximum':
        Config(shapes=[RANK_2_INPUT, RANK_2_INPUT]),
    'Minimum':
        Config(shapes=[RANK_2_INPUT, RANK_2_INPUT]),
    'MultiHeadAttention':
        Config(dict(num_heads=2, key_dim=3), [RANK_3_INPUT, RANK_3_INPUT]),
    'Multiply':
        Config(shapes=[RANK_2_INPUT, RANK_2_INPUT]),
    'PReLU':
        Config(shapes=[RANK_2_INPUT]),
    'Permute':
        Config(dict(dims=(3, 1, 2)), [RANK_4_INPUT]),
    'ReLU':
        Config(shapes=[RANK_2_INPUT]),
    'RepeatVector':
        Config(dict(n=3), [RANK_2_INPUT]),
    'Reshape':
        Config(dict(target_shape=[1, 1, 1] + RANK_3_INPUT[1:]), [RANK_3_INPUT]),
    'SeparableConv1D':
        generate_configs(
            default_config=Config(
                kwargs=dict(filters=4, kernel_size=3),
                shapes=[CONV_1D_INPUT],
            ),
            override_configs={
                'strides': Config(dict(strides=3)),
                'padding_same': Config(dict(padding='same')),
                # TF: Depthwise convolution on CPU is only supported for NHWC
                #     format
                # 'channels_first': Config(dict(data_format='channels_first')),
                'depth_multiplier': Config(dict(depth_multiplier=2)),
                'dilation_rate': Config(dict(dilation_rate=2)),
            },
        ),
    'SeparableConv2D':
        generate_configs(
            default_config=Config(
                kwargs=dict(filters=4, kernel_size=3),
                shapes=[CONV_2D_INPUT],
            ),
            override_configs={
                'strides': Config(dict(strides=3)),
                'padding_same': Config(dict(padding='same')),
                # TF: Depthwise convolution on CPU is only supported for NHWC
                #     format
                # 'channels_first': Config(dict(data_format='channels_first')),
                'depth_multiplier': Config(dict(depth_multiplier=2)),
                'dilation_rate': Config(dict(dilation_rate=2)),
            },
        ),
    'SimpleRNN':
        generate_configs(
            default_config=Config(
                kwargs=dict(units=4, return_sequences=True),
                shapes=[RANK_3_INPUT],
            ),
            override_configs={
                'go_backwards': Config(dict(go_backwards=True)),
                'stateful': Config(dict(stateful=True)),
            },
        ),
    'SimpleRNNCell':
        Config(dict(units=4), [RANK_2_INPUT, RANK_2_INPUT]),
    'Softmax':
        Config(shapes=[RANK_2_INPUT]),
    'SpatialDropout1D':
        Config(dict(rate=DROPOUT), [CONV_1D_INPUT]),
    'SpatialDropout2D':
        Config(dict(rate=DROPOUT), [CONV_2D_INPUT]),
    'SpatialDropout3D':
        Config(dict(rate=DROPOUT), [CONV_3D_INPUT]),
    'Subtract':
        Config(shapes=[RANK_2_INPUT, RANK_2_INPUT]),
    'ThresholdedReLU':
        Config(shapes=[RANK_2_INPUT]),
    'UpSampling1D':
        Config(shapes=[CONV_1D_INPUT]),
    'UpSampling2D':
        Config(shapes=[CONV_2D_INPUT]),
    'UpSampling3D':
        Config(shapes=[CONV_3D_INPUT]),
    'ZeroPadding1D':
        Config(shapes=[CONV_1D_INPUT]),
    'ZeroPadding2D':
        Config(shapes=[CONV_2D_INPUT]),
    'ZeroPadding3D':
        Config(shapes=[CONV_3D_INPUT]),
}

# Normalize LAYER_TO_UNITTEST_CONFIGURATIONS
for key, value in LAYER_TO_UNITTEST_CONFIGURATIONS.items():
  if isinstance(value, Config):
    LAYER_TO_UNITTEST_CONFIGURATIONS[key] = {'default': value}

# Layers that allow specifying the 'dropout' kwarg.
DROPOUT_LAYERS = [
    'AdditiveAttention', 'Attention', 'ConvLSTM2D', 'GRU', 'GRUCell', 'LSTM',
    'LSTMCell', 'MultiHeadAttention', 'SimpleRNN', 'SimpleRNNCell'
]

flags.DEFINE_string('layer', 'Dense',
                    f'One of {list(LAYER_TO_UNITTEST_CONFIGURATIONS.keys())}.')
flags.DEFINE_bool(
    'dynamic_batch', False,
    'Whether or not to compile the layer with a dynamic batch size.')
flags.DEFINE_bool('training', False,
                  'Whether or not to compile the layer in training mode.')
flags.DEFINE_bool(
    'test_full_api', False,
    'Whether or not to test multiple layer configurations using non-required '
    'kwargs.')
flags.DEFINE_bool(
    'list_layers_with_full_api_tests', False,
    'Whether or not to print out all layers with non-default configurations '
    '(and skip running the tests).')


def get_configs() -> Dict[str, Config]:
  """Gets the configs that we want to test for FLAGS.layer."""
  configs = LAYER_TO_UNITTEST_CONFIGURATIONS[FLAGS.layer]
  if not FLAGS.test_full_api:
    return {'default': configs['default']}
  return configs  # pytype: disable=bad-return-type


def get_input(shape: Sequence[int]) -> tf.keras.layers.Input:
  """Gets the input shape(s) that we want to test."""
  batch_size = None if FLAGS.dynamic_batch else shape[0]
  return tf.keras.layers.Input(batch_size=batch_size, shape=shape[1:])


def keras_input_normalizer(inputs: Sequence[Any]) -> Union[Any, Sequence[Any]]:
  """Unpacks inputs if it has length one."""
  return inputs[0] if len(inputs) == 1 else inputs


def keras_arg_wrapper(*args):
  """Wrapper to convert multiple positional args into a list of values."""
  return list(args) if isinstance(args, tuple) else args


def create_wrapped_keras_layer(config: Config) -> tf.keras.Model:
  """Wraps a keras layer in a model for compilation."""
  layer_class = getattr(tf.keras.layers, FLAGS.layer)

  if FLAGS.training and FLAGS.layer in DROPOUT_LAYERS:
    config.kwargs['dropout'] = DROPOUT

  inputs = keras_input_normalizer([get_input(shape) for shape in config.shapes])
  if FLAGS.layer == 'MultiHeadAttention':
    # TODO(meadowlark): Remove specialization if API changes.
    outputs = layer_class(**config.kwargs)(*inputs)
  else:
    outputs = layer_class(**config.kwargs)(inputs)
  return tf.keras.Model(inputs, outputs)


def create_tf_function_unit_test(config: Config, exported_name: str,
                                 model: tf.keras.Model) -> tf.function:
  """Wrap the model's __call__ function in a tf.function for testing."""
  input_shapes = config.shapes
  if FLAGS.dynamic_batch:
    input_shapes = [[None] + shape[1:] for shape in input_shapes]

  input_signature = [tf.TensorSpec(shape) for shape in input_shapes]
  if len(input_signature) > 1:
    input_signature = [input_signature]

  call = lambda *args: model(keras_arg_wrapper(*args), training=FLAGS.training)
  return tf_test_utils.tf_function_unit_test(input_signature=input_signature,
                                             name=exported_name)(call)


class KerasLayersModule(tf_test_utils.TestModule):

  @classmethod
  def configure_class(cls):
    """Configure each tf_function_unit_test and define it on the cls."""
    for i, (exported_name, config) in enumerate(get_configs().items()):
      model = create_wrapped_keras_layer(config)
      setattr(cls, exported_name,
              create_tf_function_unit_test(config, exported_name, model))

  def __init__(self):
    super().__init__()
    self.models = []
    for i, (exported_name, config) in enumerate(get_configs().items()):
      model = create_wrapped_keras_layer(config)
      self.models.append(model)
      setattr(self, exported_name,
              create_tf_function_unit_test(config, exported_name, model))


class KerasLayersTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(
        KerasLayersModule,
        exported_names=KerasLayersModule.get_tf_function_unit_tests())


def main(argv):
  del argv  # Unused.
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()

  if FLAGS.layer not in LAYER_TO_UNITTEST_CONFIGURATIONS:
    raise ValueError(f"Unrecognized layer: '{FLAGS.layer}'.")

  if FLAGS.list_layers_with_full_api_tests:
    for layer, configs in sorted(LAYER_TO_UNITTEST_CONFIGURATIONS.items()):
      if len(configs) > 1:
        print(f'    "{layer}",')
    return

  # Set up name for saving artifacts.
  dynamic_batch_str = 'dynamic_batch' if FLAGS.dynamic_batch else 'static_batch'
  training_str = 'training' if FLAGS.training else 'non_training'
  full_api_str = 'full_api' if FLAGS.test_full_api else 'default_api'
  settings_str = f'{full_api_str}_{dynamic_batch_str}_{training_str}'
  KerasLayersModule.__name__ = os.path.join('keras_layers', FLAGS.layer,
                                            settings_str)

  # Use the configurations for FLAGS.layer to add the tf.functions we wish
  # to test to the KerasLayersModule, and then generate unittests for each of
  # them.
  KerasLayersModule.configure_class()
  KerasLayersTest.generate_unit_tests(KerasLayersModule)
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
