# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests of tf.keras.layer.Layer subclasses."""

import collections
import copy
import inspect
import os
from typing import Any, Dict, List, Sequence, Tuple, Union

from absl import app
from absl import flags
from absl import logging
from iree.tf.support import tf_test_utils
from iree.tf.support import tf_utils
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

DROPOUT = 0.5
CONV_FILTERS = 2
CONV_KERNEL_SIZE = 2
DIM = 3

# Used for attention layers and recurrent layers.
RANK_3_SHAPE = [DIM] * 3
# Highest rank that tf.keras will allow for all layers.
RANK_5_SHAPE = [DIM] * 5

UNARY_SIGNATURE_SHAPES = [[RANK_5_SHAPE]]
BINARY_SIGNATURE_SHAPES = [[RANK_5_SHAPE] * 2]
TERNARY_SIGNATURE_SHAPES = [[RANK_5_SHAPE] * 3]

CONV_1D_SIGNATURE_SHAPES = [[[2, 8, 3]]]
CONV_2D_SIGNATURE_SHAPES = [[[2, 8, 8, 3]]]
CONV_3D_SIGNATURE_SHAPES = [[[2, 8, 8, 8, 3]]]

RNN_SIGNATURE_SHAPES = [[RANK_3_SHAPE]]
RNN_KWARGS_TO_VALUES = dict(units=[4],
                            return_sequences=[False, True],
                            stateful=[False, True])

POOLING_KWARGS_TO_VALUES = dict(strides=[None, 2],
                                padding=["valid", "same"],
                                data_format=[None, "channels_first"])
CONV_KWARGS_TO_VALUES = dict(filters=[CONV_FILTERS],
                             kernel_size=[CONV_KERNEL_SIZE],
                             strides=[1, 2],
                             padding=["valid", "same"],
                             data_format=[None, "channels_first"],
                             dilation_rate=[1, 1])
# Address pooling and conv layers having different default values for
# 'data_format' for 1D layers.
POOLING_1D_KWARGS_TO_VALUES = copy.deepcopy(POOLING_KWARGS_TO_VALUES)
POOLING_1D_KWARGS_TO_VALUES.update(
    {"data_format": ["channels_last", "channels_first"]})
CONV_1D_KWARGS_TO_VALUES = copy.deepcopy(CONV_KWARGS_TO_VALUES)
CONV_1D_KWARGS_TO_VALUES.update(
    {"data_format": ["channels_last", "channels_first"]})

# Unsupported by TensorFlow (at least on CPU).
LAYERS_TO_TF_UNSUPPORTED_NON_DEFAULT_KWARGS = {
    "AveragePooling2D": ["data_format"],
    "Conv1D": ["data_format"],
    "Conv1DTranspose": ["data_format", "dilation_rate"],
    "Conv2D": ["data_format"],
    "Conv3D": ["data_format"],
    "Conv3DTranspose": ["data_format"],
    "LocallyConnected1D": ["padding"],
    "LocallyConnected2D": ["padding"],
    "MaxPool2D": ["data_format"],
}

# Some layers have kwargs which cannot both have non-default values.
LAYERS_TO_MUTUALLY_EXCLUSIVE_KWARGS = {
    "Conv1D": ["strides", "dilation_rate"],
    "Conv2D": ["strides", "dilation_rate"],
    "Conv2DTranspose": ["strides", "dilation_rate"],
    "Conv3D": ["strides", "dilation_rate"],
    "ConvLSTM2D": ["strides", "dilation_rate"],
}

# Some layers cannot operate on a fully dynamic shape, only dynamic batch size.
LAYERS_WITH_BATCH_ONLY_DYNAMIC_SHAPE = [
    "AveragePooling1D",  # uses tf.nn.avg_pool2d with reshape, which is illegal
    "BatchNormalization",  # failed to materialize conversion
    "Conv3D",  # tf.Conv3D op illegal
    "MaxPool1D",  # uses tf.nn.max_pool2d with reshape, which is illegal
]


def get_default_kwargs_values(layer: str) -> Dict[str, Any]:
  """Gets the default kwargs for a tf.keras.layers layer."""
  layer_class = getattr(tf.keras.layers, layer)
  layer_parameters = inspect.signature(layer_class.__init__).parameters
  kwargs_to_default_values = {
      kwarg: value.default
      for kwarg, value in layer_parameters.items()
      if value.default is not inspect.Parameter.empty
  }
  return kwargs_to_default_values


def _equal_or_splat_equal(value: Any, sequence: Any) -> bool:
  """Returns True if value==sequence or value==(every element in sequence)."""
  if value == sequence:
    return True
  elif isinstance(sequence, (list, tuple)):
    for element in sequence:
      if not _equal_or_splat_equal(value, element):
        return False
    return True
  return False


def get_non_default_kwargs(
    layer: str, unit_test_spec: tf_test_utils.UnitTestSpec) -> List[str]:
  """Returns all non-default optional kwargs in unit_test_spec."""
  kwargs_to_defaults = get_default_kwargs_values(layer)
  non_default_kwargs = []
  for kwarg, value in unit_test_spec.kwargs.items():
    if (kwarg in kwargs_to_defaults and
        not _equal_or_splat_equal(value, kwargs_to_defaults[kwarg])):
      non_default_kwargs.append(kwarg)
  return non_default_kwargs


def unsupported_by_tf(layer: str,
                      unit_test_spec: tf_test_utils.UnitTestSpec) -> bool:
  """True if unit_test_spec specifies tf-unsupported non-default kwargs."""
  if layer in LAYERS_TO_TF_UNSUPPORTED_NON_DEFAULT_KWARGS:
    unsupported_kwargs = LAYERS_TO_TF_UNSUPPORTED_NON_DEFAULT_KWARGS[layer]
    non_default_kwargs = get_non_default_kwargs(layer, unit_test_spec)
    return any(kwarg in unsupported_kwargs for kwarg in non_default_kwargs)
  return False


def has_mutually_exclusive_kwargs(
    layer: str, unit_test_spec: tf_test_utils.UnitTestSpec) -> bool:
  """True if unit_test_spec specifies mutually exclusive non-default kwargs."""
  if layer in LAYERS_TO_MUTUALLY_EXCLUSIVE_KWARGS:
    mutually_exclusive_kwargs = LAYERS_TO_MUTUALLY_EXCLUSIVE_KWARGS[layer]
    non_default_kwargs = get_non_default_kwargs(layer, unit_test_spec)
    return set(mutually_exclusive_kwargs).issubset(set(non_default_kwargs))
  return False


# A dictionary mapping tf.keras.layers names to lists of UnitTestSpecs.
# Each unit_test_name will have the tf.keras.layer name prepended to it.
#
# Each layer is required to have a UnitTestSpec with all-default values for
# unrequired kwargs. This allows us to seperately test the basic api and the
# full api.
LAYERS_TO_UNIT_TEST_SPECS = {
    "Activation":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            kwargs_to_values=dict(activation=["relu"])),
    "ActivityRegularization":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            kwargs_to_values=dict(l1=[0.0, 0.1], l2=[0.0, 0.1])),
    "Add":
        tf_test_utils.unit_test_specs_from_signatures(BINARY_SIGNATURE_SHAPES),
    "AdditiveAttention":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=[(RANK_3_SHAPE, RANK_3_SHAPE, RANK_3_SHAPE)],
            kwargs_to_values=dict(causal=[False, True])),
    "AlphaDropout":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            kwargs_to_values=dict(rate=[DROPOUT])),
    "Attention":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=[(RANK_3_SHAPE, RANK_3_SHAPE, RANK_3_SHAPE)],
            kwargs_to_values=dict(causal=[False, True])),
    "Average":
        tf_test_utils.unit_test_specs_from_signatures(BINARY_SIGNATURE_SHAPES),
    "AveragePooling1D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_1D_SIGNATURE_SHAPES,
            kwargs_to_values=POOLING_1D_KWARGS_TO_VALUES),
    "AveragePooling2D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_2D_SIGNATURE_SHAPES,
            kwargs_to_values=POOLING_KWARGS_TO_VALUES),
    "AveragePooling3D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_3D_SIGNATURE_SHAPES,
            kwargs_to_values=POOLING_KWARGS_TO_VALUES),
    "BatchNormalization":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            kwargs_to_values=dict(renorm=[False, True])),
    "Concatenate":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            kwargs_to_values=dict(axis=[-1, 0])),
    "Conv1D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_1D_SIGNATURE_SHAPES,
            kwargs_to_values=CONV_1D_KWARGS_TO_VALUES),
    "Conv1DTranspose":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_1D_SIGNATURE_SHAPES,
            kwargs_to_values=CONV_KWARGS_TO_VALUES),
    "Conv2D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_2D_SIGNATURE_SHAPES,
            kwargs_to_values=CONV_KWARGS_TO_VALUES),
    "Conv2DTranspose":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_2D_SIGNATURE_SHAPES,
            kwargs_to_values=CONV_KWARGS_TO_VALUES),
    "Conv3D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_3D_SIGNATURE_SHAPES,
            kwargs_to_values=CONV_KWARGS_TO_VALUES),
    "Conv3DTranspose":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_3D_SIGNATURE_SHAPES,
            kwargs_to_values=CONV_KWARGS_TO_VALUES),
    "ConvLSTM2D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_3D_SIGNATURE_SHAPES,
            kwargs_to_values=dict(filters=[CONV_FILTERS],
                                  kernel_size=[CONV_KERNEL_SIZE],
                                  return_state=[False, True],
                                  strides=[1, 2],
                                  dilation_rate=[1, 1],
                                  stateful=[False, True])),
    "Cropping1D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_1D_SIGNATURE_SHAPES,
            kwargs_to_values=dict(cropping=[1, (1, 2)])),
    "Cropping2D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_2D_SIGNATURE_SHAPES,
            kwargs_to_values=dict(cropping=[0, ((1, 2), (2, 1))])),
    "Cropping3D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_3D_SIGNATURE_SHAPES,
            kwargs_to_values=dict(cropping=[1, ((1, 2), (2, 1), (1, 0))])),
    "Dense":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            kwargs_to_values=dict(units=[8])),
    "DepthwiseConv2D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_2D_SIGNATURE_SHAPES,
            kwargs_to_values=dict(kernel_size=[CONV_KERNEL_SIZE],
                                  strides=[1, 2],
                                  padding=["valid", "same"],
                                  dilation_rate=[1, 1],
                                  depth_multiplier=[1, 2])),
    "Dot":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            kwargs_to_values=dict(axes=[(1, 2)])),
    "Dropout":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            kwargs_to_values=dict(rate=[DROPOUT])),
    "ELU":
        tf_test_utils.unit_test_specs_from_signatures(UNARY_SIGNATURE_SHAPES),
    "Embedding":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            kwargs_to_values=dict(input_dim=[4], output_dim=[2])),
    "Flatten":
        tf_test_utils.unit_test_specs_from_signatures(UNARY_SIGNATURE_SHAPES),
    "GRU":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=RNN_SIGNATURE_SHAPES,
            kwargs_to_values=RNN_KWARGS_TO_VALUES),
    "GaussianDropout":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=RNN_SIGNATURE_SHAPES,
            kwargs_to_values=dict(rate=[DROPOUT])),
    "GaussianNoise":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=RNN_SIGNATURE_SHAPES,
            kwargs_to_values=dict(stddev=[1.0])),
    "GlobalAveragePooling1D":
        tf_test_utils.unit_test_specs_from_signatures(CONV_1D_SIGNATURE_SHAPES),
    "GlobalAveragePooling2D":
        tf_test_utils.unit_test_specs_from_signatures(CONV_2D_SIGNATURE_SHAPES),
    "GlobalAveragePooling3D":
        tf_test_utils.unit_test_specs_from_signatures(CONV_3D_SIGNATURE_SHAPES),
    "GlobalMaxPool1D":
        tf_test_utils.unit_test_specs_from_signatures(CONV_1D_SIGNATURE_SHAPES),
    "GlobalMaxPool2D":
        tf_test_utils.unit_test_specs_from_signatures(CONV_2D_SIGNATURE_SHAPES),
    "GlobalMaxPool3D":
        tf_test_utils.unit_test_specs_from_signatures(CONV_3D_SIGNATURE_SHAPES),
    "InputLayer":
        tf_test_utils.unit_test_specs_from_signatures(UNARY_SIGNATURE_SHAPES),
    "LSTM":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=RNN_SIGNATURE_SHAPES,
            kwargs_to_values=RNN_KWARGS_TO_VALUES),
    "Lambda":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            kwargs_to_values=dict(function=[lambda x: x * x])),
    "LayerNormalization":
        tf_test_utils.unit_test_specs_from_signatures(UNARY_SIGNATURE_SHAPES),
    "LeakyReLU":
        tf_test_utils.unit_test_specs_from_signatures(UNARY_SIGNATURE_SHAPES),
    "LocallyConnected1D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_1D_SIGNATURE_SHAPES,
            kwargs_to_values=dict(filters=[CONV_FILTERS],
                                  kernel_size=[CONV_KERNEL_SIZE],
                                  strides=[1, 3],
                                  padding=["valid", "same"],
                                  implementation=[1, 3])),
    "LocallyConnected2D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_2D_SIGNATURE_SHAPES,
            kwargs_to_values=dict(filters=[CONV_FILTERS],
                                  kernel_size=[CONV_KERNEL_SIZE],
                                  strides=[1, 3],
                                  padding=["valid", "same"],
                                  implementation=[1, 3])),
    "Masking":
        tf_test_utils.unit_test_specs_from_signatures(UNARY_SIGNATURE_SHAPES),
    "MaxPool1D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_1D_SIGNATURE_SHAPES,
            kwargs_to_values=POOLING_1D_KWARGS_TO_VALUES),
    "MaxPool2D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_2D_SIGNATURE_SHAPES,
            kwargs_to_values=POOLING_KWARGS_TO_VALUES),
    "MaxPool3D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_3D_SIGNATURE_SHAPES,
            kwargs_to_values=POOLING_KWARGS_TO_VALUES),
    "Maximum":
        tf_test_utils.unit_test_specs_from_signatures(BINARY_SIGNATURE_SHAPES),
    "Minimum":
        tf_test_utils.unit_test_specs_from_signatures(BINARY_SIGNATURE_SHAPES),
    "MultiHeadAttention":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=[(RANK_3_SHAPE, RANK_3_SHAPE)],
            kwargs_to_values=dict(num_heads=[2], key_dim=[3])),
    "Multiply":
        tf_test_utils.unit_test_specs_from_signatures(BINARY_SIGNATURE_SHAPES),
    "PReLU":
        tf_test_utils.unit_test_specs_from_signatures(UNARY_SIGNATURE_SHAPES),
    "Permute":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            kwargs_to_values=dict(dims=[(3, 1, 4, 2)])),
    "ReLU":
        tf_test_utils.unit_test_specs_from_signatures(UNARY_SIGNATURE_SHAPES),
    "RepeatVector":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=[((2, 2),)], kwargs_to_values=dict(n=[3])),
    "Reshape":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=[((3, 2, 2, 2),)],
            kwargs_to_values=dict(target_shape=[(2, 1, 4, 1)])),
    "SeparableConv1D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_1D_SIGNATURE_SHAPES,
            kwargs_to_values=dict(filters=[CONV_FILTERS],
                                  kernel_size=[CONV_KERNEL_SIZE],
                                  strides=[1, 2],
                                  padding=["valid", "same"],
                                  dilation_rate=[1, 1],
                                  depth_multiplier=[1, 2])),
    "SeparableConv2D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_2D_SIGNATURE_SHAPES,
            kwargs_to_values=dict(filters=[CONV_FILTERS],
                                  kernel_size=[CONV_KERNEL_SIZE],
                                  strides=[1, 2],
                                  padding=["valid", "same"],
                                  dilation_rate=[1, 1],
                                  depth_multiplier=[1, 2])),
    "SimpleRNN":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=RNN_SIGNATURE_SHAPES,
            kwargs_to_values=RNN_KWARGS_TO_VALUES),
    "Softmax":
        tf_test_utils.unit_test_specs_from_signatures(UNARY_SIGNATURE_SHAPES),
    "SpatialDropout1D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_1D_SIGNATURE_SHAPES,
            kwargs_to_values=dict(rate=[DROPOUT])),
    "SpatialDropout2D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_2D_SIGNATURE_SHAPES,
            kwargs_to_values=dict(rate=[DROPOUT])),
    "SpatialDropout3D":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=CONV_3D_SIGNATURE_SHAPES,
            kwargs_to_values=dict(rate=[DROPOUT])),
    "Subtract":
        tf_test_utils.unit_test_specs_from_signatures(BINARY_SIGNATURE_SHAPES),
    "ThresholdedReLU":
        tf_test_utils.unit_test_specs_from_signatures(UNARY_SIGNATURE_SHAPES),
    "UpSampling1D":
        tf_test_utils.unit_test_specs_from_signatures(CONV_1D_SIGNATURE_SHAPES),
    "UpSampling2D":
        tf_test_utils.unit_test_specs_from_signatures(CONV_2D_SIGNATURE_SHAPES),
    "UpSampling3D":
        tf_test_utils.unit_test_specs_from_signatures(CONV_3D_SIGNATURE_SHAPES),
    "ZeroPadding1D":
        tf_test_utils.unit_test_specs_from_signatures(CONV_1D_SIGNATURE_SHAPES),
    "ZeroPadding2D":
        tf_test_utils.unit_test_specs_from_signatures(CONV_2D_SIGNATURE_SHAPES),
    "ZeroPadding3D":
        tf_test_utils.unit_test_specs_from_signatures(CONV_3D_SIGNATURE_SHAPES),
}

for layer, specs in LAYERS_TO_UNIT_TEST_SPECS.items():
  # Update using 'with_name' to avoid updating shared UnitTestSpecs.
  specs = [spec.with_name(f"{layer}__{spec.unit_test_name}") for spec in specs]
  LAYERS_TO_UNIT_TEST_SPECS[layer] = specs

  # Validate that there are not multiple UnitTestSpecs with the same name.
  seen_unit_test_names = set()
  for spec in specs:
    if spec.unit_test_name in seen_unit_test_names:
      raise ValueError(
          f"Found multiple UnitTestSpecs with the name '{spec.unit_test_name}'")
    seen_unit_test_names.add(spec.unit_test_name)

  # Validate that there is one spec that has default values for all unrequired
  # kwargs.
  has_default_unrequired_kwargs = False
  for spec in specs:
    if not get_non_default_kwargs(layer, spec):
      has_default_unrequired_kwargs = True

  if not has_default_unrequired_kwargs:
    raise ValueError(
        f"The configuration for '{layer}' did not have a UnitTestSpec with all "
        "default kwargs.")

# Layers that allow specifying the 'dropout' kwarg.
DROPOUT_LAYERS = [
    "AdditiveAttention", "Attention", "ConvLSTM2D", "GRU", "LSTM",
    "MultiHeadAttention", "SimpleRNN"
]

flags.DEFINE_string("layer", None,
                    f"One of {list(LAYERS_TO_UNIT_TEST_SPECS.keys())}.")
flags.DEFINE_bool(
    "dynamic_dims", False,
    "Whether or not to compile the layer with a dynamic dimension sizes.")
flags.DEFINE_bool("training", False,
                  "Whether or not to compile the layer in training mode.")
flags.DEFINE_bool(
    "test_default_kwargs_only", True,
    "Whether or not to test multiple layer configurations using non-required "
    "kwargs.")
flags.DEFINE_bool(
    "list_layers_with_full_api_tests", False,
    "Whether or not to print out all layers with non-default configurations "
    "(and skip running the tests).")


def get_input(shape: Sequence[int]) -> tf.keras.layers.Input:
  """Converts a shape into a tf.keras.Input."""
  # Most keras layers are only compatible with dynamic batch sizes.
  batch_size = None if FLAGS.dynamic_dims else shape[0]
  return tf.keras.layers.Input(batch_size=batch_size, shape=shape[1:])


def keras_input_normalizer(inputs: Sequence[Any]) -> Union[Any, Sequence[Any]]:
  """Unpacks inputs if it has length one."""
  return inputs[0] if len(inputs) == 1 else inputs


def create_wrapped_keras_layer(
    layer: str, unit_test_spec: tf_test_utils.UnitTestSpec) -> tf.keras.Model:
  """Wraps a keras layer in a model for compilation."""
  layer_class = getattr(tf.keras.layers, layer)

  kwargs = copy.deepcopy(unit_test_spec.kwargs)
  if FLAGS.training and layer in DROPOUT_LAYERS:
    kwargs["dropout"] = DROPOUT

  if "dtype" not in unit_test_spec.kwargs:
    kwargs["dtype"] = unit_test_spec.input_signature[0].dtype

  inputs = keras_input_normalizer(
      [get_input(spec.shape) for spec in unit_test_spec.input_signature])
  if layer == "MultiHeadAttention":
    # TODO(meadowlark): Remove specialization if API changes.
    outputs = layer_class(**kwargs)(*inputs)
  else:
    outputs = layer_class(**kwargs)(inputs)
  return tf.keras.Model(inputs, outputs)


def create_layer_unit_test(
    model: tf.keras.Model,
    unit_test_spec: tf_test_utils.UnitTestSpec) -> tf.function:
  """Wrap the model's __call__ function in a tf.function for testing."""
  static_signature = unit_test_spec.input_signature

  dynamic_signature = static_signature
  if FLAGS.dynamic_dims:

    def make_batch_size_dynamic(tensor_spec: tf.TensorSpec) -> tf.TensorSpec:
      return tf.TensorSpec([None] + tensor_spec.shape[1:], tensor_spec.dtype)

    if FLAGS.layer in LAYERS_WITH_BATCH_ONLY_DYNAMIC_SHAPE:
      dynamic_signature = tf_utils.apply_function(dynamic_signature,
                                                  make_batch_size_dynamic)
    else:
      dynamic_signature = tf_utils.apply_function(dynamic_signature,
                                                  tf_utils.make_dims_dynamic)

  if len(static_signature) > 1:
    static_signature = [static_signature]
    dynamic_signature = [dynamic_signature]

  call = lambda *args: model(*args, training=FLAGS.training)
  return tf_test_utils.tf_function_unit_test(
      input_signature=dynamic_signature,
      static_signature=static_signature,
      input_generator=unit_test_spec.input_generator,
      input_args=unit_test_spec.input_args,
      name=unit_test_spec.unit_test_name)(call)


class KerasLayersModule(tf_test_utils.TestModule):

  def __init__(self):
    super().__init__()
    self.models = []
    for unit_test_spec in LAYERS_TO_UNIT_TEST_SPECS[FLAGS.layer]:
      if (FLAGS.test_default_kwargs_only and
          get_non_default_kwargs(FLAGS.layer, unit_test_spec)):
        # Skip all UnitTestSpecs with non-default unrequired kwargs.
        continue

      if (unsupported_by_tf(FLAGS.layer, unit_test_spec) or
          has_mutually_exclusive_kwargs(FLAGS.layer, unit_test_spec)):
        # Filter out UnitTestSpecs with kwargs that TensorFlow can't run on
        # CPU or that are mutually exclusive. This allows us to take a product
        # like that in CONV_KWARGS_TO_VALUE and filter out the configurations
        # lacking support for particular layers.
        continue

      model = create_wrapped_keras_layer(FLAGS.layer, unit_test_spec)
      # IREE requires that the models are stored on the module instance.
      self.models.append(model)
      layer_unit_test = create_layer_unit_test(model, unit_test_spec)
      setattr(self, unit_test_spec.unit_test_name, layer_unit_test)


def get_relative_artifacts_dir() -> str:
  dynamic_str = "dynamic" if FLAGS.dynamic_dims else "static"
  training_str = "training" if FLAGS.training else "non_training"
  full_api_str = "default_api" if FLAGS.test_default_kwargs_only else "full_api"
  settings_str = f"{full_api_str}_{dynamic_str}_{training_str}"
  return os.path.join("tf", "keras", "layers", FLAGS.layer, settings_str)


class KerasLayersTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(
        KerasLayersModule,
        exported_names=KerasLayersModule.get_tf_function_unit_tests(),
        relative_artifacts_dir=get_relative_artifacts_dir())


def main(argv):
  del argv  # Unused.
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()

  if FLAGS.list_layers_with_full_api_tests:
    for layer, unit_test_specs in sorted(LAYERS_TO_UNIT_TEST_SPECS.items()):
      if len(unit_test_specs) > 1:
        print(f'    "{layer}",')
    return

  if FLAGS.layer not in LAYERS_TO_UNIT_TEST_SPECS:
    raise ValueError(f"Unrecognized layer: '{FLAGS.layer}'")

  unit_tests = KerasLayersModule.get_tf_function_unit_tests()
  logging.info("Testing the following %s functions: %s", len(unit_tests),
               unit_tests)

  KerasLayersTest.generate_unit_tests(KerasLayersModule)
  tf.test.main()


if __name__ == "__main__":
  app.run(main)
