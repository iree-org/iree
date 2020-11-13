import collections
import os
from typing import Any, Dict, Sequence, Type, Union

from absl import app
from absl import flags
import numpy as np
from pyiree.tf.support import tf_test_utils
from pyiree.tf.support import tf_utils
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

# Controls the input signature, call kwargs, input generator and args for a
# unittest of a tf.math function.
Config = collections.namedtuple("Config",
                                ["signature", "kwargs", "generator", "args"])
# Use old default API for compatibility with Python 3.6.
Config.__new__.__defaults__ = (None, dict(), None, None)


def create_signature_from_args(config: Config) -> Config:
  """Uses a Config's args to generate an signature for tf.function."""
  if config.signature is not None:
    raise ValueError(
        "Tried to add a signature to a config that already has one.")
  if config.args is None:
    raise ValueError(
        "Tried to generate a signature from a config without any args.")

  signature = tf_utils.apply_function(
      config.args, lambda x: tf.TensorSpec.from_tensor(tf.convert_to_tensor(x)))
  return Config(signature, config.kwargs, config.generator, config.args)


def configure_dtypes(base_config: Config,
                     dtypes: Sequence[tf.DType],
                     exported_name: str = None) -> Dict[str, Config]:
  """Uses a Config's signature to replicate the it across multiple dtypes."""
  if base_config.signature is None:
    raise ValueError("'base_config' must have a signature.")
  incorrect_dtypes = [
      dtype for dtype in dtypes if not isinstance(dtype, tf.DType)
  ]
  if len(incorrect_dtypes):
    raise ValueError("Expected all elements of dtypes to be tf.DType "
                     f"subclasses, but got {incorrect_dtypes}.")

  configs = dict()
  for dtype in dtypes:
    # Update the base_config's spec.
    update_dtype = lambda spec: tf.TensorSpec(spec.shape, dtype)
    signature = tf_utils.apply_function(base_config.signature, update_dtype)
    config = Config(signature, base_config.kwargs, base_config.generator,
                    base_config.args)

    # Use the dtype to distinguish the name to export.
    if exported_name is not None:
      config_name = f"{exported_name}_{dtype.name}"
    else:
      config_name = dtype.name
    configs[config_name] = config
  return configs


def is_complex(config: Config) -> bool:
  # Only allow complex datatypes for flat signatures.
  return (all([isinstance(spec, tf.TensorSpec) for spec in config.signature])
          and any([spec.dtype.is_complex for spec in config.signature]))


# As high as tf goes without breaking.
RANK_7_INPUT = [1, 1, 1, 2, 2, 2, 2]

# "Untyped" signatures – for use with 'configure_dtypes'
UNARY = [tf.TensorSpec(RANK_7_INPUT)]
BINARY = [tf.TensorSpec(RANK_7_INPUT)] * 2
TERNARY = [tf.TensorSpec(RANK_7_INPUT)] * 2

# Typed signatures with default precision.
UNARY_FLOAT = [tf.TensorSpec(RANK_7_INPUT, tf.float32)]
BINARY_FLOAT = [tf.TensorSpec(RANK_7_INPUT, tf.float32)] * 2
TERNARY_FLOAT = [tf.TensorSpec(RANK_7_INPUT, tf.float32)] * 3
UNARY_INT = [tf.TensorSpec(RANK_7_INPUT, tf.int32)]
BINARY_INT = [tf.TensorSpec(RANK_7_INPUT, tf.int32)] * 2
UNARY_BOOL = [tf.TensorSpec(RANK_7_INPUT, tf.bool)]
BINARY_BOOL = [tf.TensorSpec(RANK_7_INPUT, tf.bool)] * 2
UNARY_COMPLEX = [tf.TensorSpec(RANK_7_INPUT, tf.complex64)]
BINARY_COMPLEX = [tf.TensorSpec(RANK_7_INPUT, tf.complex64)] * 2

ALL_NUMBER_TYPES = [tf.int32, tf.float32, tf.complex64]
REAL_NUMBER_TYPES = [tf.int32, tf.float32]
FLOATING_NUMBER_TYPES = [tf.float32, tf.complex64]

# Reused Configs.
SEGMENT_CONFIG = create_signature_from_args(
    Config(args=[
        tf.constant([
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [5, 6, 7, 8],
        ], np.float32),
        np.array([0, 0, 1], np.int32),
    ]))
UNSORTED_SEGMENT_CONFIG = create_signature_from_args(
    Config(args=[
        tf.constant([
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [5, 6, 7, 8],
        ], np.float32),
        np.array([0, 0, 1], np.int32),
        2,
    ]))

# A dict mapping tf.math function names to either a single Config (representing
# the signature and other metadata to use to test that function) or a dict
# mapping exported_names to Configs. The latter case is usually automatically
# generated via 'configure_dtypes'.
#
# Each entry will be normalized to be a dict mapping exported_names to Configs,
# with a default exported_name matching the name of the tf.math function.
FUNCTION_TO_CONFIGS = {
    "abs":
        configure_dtypes(Config(signature=UNARY), ALL_NUMBER_TYPES),
    "accumulate_n":
        configure_dtypes(Config(signature=[TERNARY]), REAL_NUMBER_TYPES),
    "acos":
        Config(signature=UNARY_FLOAT),
    "acosh":
        Config(signature=UNARY_FLOAT, generator=tf_utils.ndarange),
    "add":
        configure_dtypes(Config(signature=BINARY), ALL_NUMBER_TYPES),
    "add_n":
        configure_dtypes(Config(signature=[TERNARY]), REAL_NUMBER_TYPES),
    "angle":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "argmax":
        configure_dtypes(Config(signature=UNARY), REAL_NUMBER_TYPES),
    "argmin":
        configure_dtypes(Config(signature=UNARY), REAL_NUMBER_TYPES),
    "asin":
        Config(signature=UNARY_FLOAT),
    "asinh":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "atan":
        Config(signature=UNARY_FLOAT),
    "atan2":
        Config(signature=BINARY_FLOAT),
    "atanh":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "bessel_i0":
        Config(signature=UNARY_FLOAT),
    "bessel_i0e":
        Config(signature=UNARY_FLOAT),
    "bessel_i1":
        Config(signature=UNARY_FLOAT),
    "bessel_i1e":
        Config(signature=UNARY_FLOAT),
    "betainc":
        Config(signature=TERNARY_FLOAT),
    "bincount":
        Config(signature=UNARY_INT, generator=tf_utils.ndarange),
    "ceil":
        Config(signature=UNARY_FLOAT),
    "confusion_matrix":
        configure_dtypes(
            base_config=create_signature_from_args(
                Config(args=[tf.constant([1, 2, 4]),
                             tf.constant([2, 2, 4])])),
            dtypes=REAL_NUMBER_TYPES,
        ),
    "conj":
        configure_dtypes(Config(signature=UNARY_COMPLEX), ALL_NUMBER_TYPES),
    "cos":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "cosh":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "count_nonzero":
        configure_dtypes(Config(signature=UNARY, generator=tf_utils.ndarange),
                         ALL_NUMBER_TYPES),
    "cumprod":
        configure_dtypes(Config(signature=UNARY), ALL_NUMBER_TYPES),
    "cumsum":
        configure_dtypes(Config(signature=UNARY), ALL_NUMBER_TYPES),
    "cumulative_logsumexp":
        Config(signature=UNARY_FLOAT),
    "digamma":
        Config(signature=UNARY_FLOAT),
    "divide":
        configure_dtypes(Config(signature=BINARY), ALL_NUMBER_TYPES),
    "divide_no_nan":
        configure_dtypes(Config(signature=BINARY), FLOATING_NUMBER_TYPES),
    "equal":
        configure_dtypes(Config(signature=BINARY), REAL_NUMBER_TYPES),
    "erf":
        Config(signature=UNARY_FLOAT),
    "erfc":
        Config(signature=UNARY_FLOAT),
    "erfinv":
        Config(signature=UNARY_FLOAT),
    "exp":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "expm1":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "floor":
        Config(signature=UNARY_FLOAT),
    "floordiv":
        Config(signature=BINARY),
    "floormod":
        Config(signature=BINARY),
    "greater":
        configure_dtypes(Config(signature=BINARY), REAL_NUMBER_TYPES),
    "greater_equal":
        configure_dtypes(Config(signature=BINARY), REAL_NUMBER_TYPES),
    "igamma":
        Config(signature=BINARY),
    "igammac":
        Config(signature=BINARY),
    "imag":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "in_top_k":
        Config(signature=[tf.TensorSpec([8], tf.int32),
                          tf.TensorSpec([8, 3])],
               kwargs=dict(k=3),
               generator=tf_utils.ndarange),
    "invert_permutation":
        Config(signature=[tf.TensorSpec([8], tf.int32)],
               generator=tf_utils.random_permutation),
    "is_finite":
        create_signature_from_args(
            Config(args=[tf.constant([[1., np.nan], [np.inf, 2.]])])),
    "is_inf":
        create_signature_from_args(
            Config(args=[tf.constant([[1., np.nan], [np.inf, 2.]])])),
    "is_nan":
        create_signature_from_args(
            Config(args=[tf.constant([[1., np.nan], [np.inf, 2.]])])),
    "is_non_decreasing":
        configure_dtypes(Config(signature=UNARY), REAL_NUMBER_TYPES),
    "is_strictly_increasing":
        configure_dtypes(Config(signature=UNARY), REAL_NUMBER_TYPES),
    "l2_normalize":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "lbeta":
        Config(signature=UNARY_FLOAT),
    "less":
        configure_dtypes(Config(signature=BINARY), REAL_NUMBER_TYPES),
    "less_equal":
        configure_dtypes(Config(signature=BINARY), REAL_NUMBER_TYPES),
    "lgamma":
        Config(signature=UNARY_FLOAT),
    "log":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "log1p":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "log_sigmoid":
        Config(signature=UNARY_FLOAT),
    "log_softmax":
        Config(signature=UNARY_FLOAT),
    "logical_and":
        Config(signature=BINARY_BOOL),
    "logical_not":
        Config(signature=UNARY_BOOL),
    "logical_or":
        Config(signature=BINARY_BOOL),
    "logical_xor":
        Config(signature=BINARY_BOOL),
    "maximum":
        configure_dtypes(Config(signature=BINARY), REAL_NUMBER_TYPES),
    "minimum":
        configure_dtypes(Config(signature=BINARY), REAL_NUMBER_TYPES),
    "mod":
        configure_dtypes(
            Config(signature=BINARY,
                   generator=lambda *args: tf_utils.ndarange(*args) + 1),
            REAL_NUMBER_TYPES),
    "multiply":
        configure_dtypes(Config(signature=BINARY), ALL_NUMBER_TYPES),
    "multiply_no_nan":
        configure_dtypes(Config(signature=BINARY), FLOATING_NUMBER_TYPES),
    "ndtri":
        Config(signature=UNARY_FLOAT),
    "negative":
        configure_dtypes(Config(signature=UNARY), ALL_NUMBER_TYPES),
    "nextafter":
        Config(signature=BINARY),
    "not_equal":
        configure_dtypes(Config(signature=BINARY), REAL_NUMBER_TYPES),
    "polygamma":
        create_signature_from_args(
            Config(args=[tf.ones(16), tf.linspace(0.5, 4, 16)])),
    "polyval":
        Config(signature=[TERNARY_FLOAT, tf.TensorSpec([])]),
    "pow": {
        # Use ndarange to avoid negative integer powers.
        "int32": Config(signature=BINARY_INT, generator=tf_utils.ndarange),
        "float32": Config(signature=BINARY_FLOAT),
        "complex64": Config(signature=BINARY_COMPLEX),
    },
    "real":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "reciprocal":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "reciprocal_no_nan":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "reduce_all":
        Config(signature=UNARY_BOOL),
    "reduce_any":
        Config(signature=UNARY_BOOL),
    "reduce_euclidean_norm":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "reduce_logsumexp":
        Config(signature=UNARY_FLOAT),
    "reduce_max":
        configure_dtypes(Config(signature=UNARY), REAL_NUMBER_TYPES),
    "reduce_mean":
        configure_dtypes(Config(signature=UNARY), REAL_NUMBER_TYPES),
    "reduce_min":
        configure_dtypes(Config(signature=UNARY), REAL_NUMBER_TYPES),
    "reduce_prod":
        configure_dtypes(Config(signature=UNARY), REAL_NUMBER_TYPES),
    "reduce_std":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "reduce_sum":
        configure_dtypes(Config(signature=UNARY), REAL_NUMBER_TYPES),
    "reduce_variance":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "rint":
        Config(signature=UNARY_FLOAT),
    "round":
        Config(signature=UNARY_FLOAT),
    "rsqrt":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "scalar_mul":
        Config(
            signature=[tf.TensorSpec([]), tf.TensorSpec([8])]),
    "segment_max":
        SEGMENT_CONFIG,
    "segment_mean":
        SEGMENT_CONFIG,
    "segment_min":
        SEGMENT_CONFIG,
    "segment_prod":
        SEGMENT_CONFIG,
    "segment_sum":
        SEGMENT_CONFIG,
    "sigmoid":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "sign":
        configure_dtypes(Config(signature=UNARY), ALL_NUMBER_TYPES),
    "sin":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "sinh":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "sobol_sample":
        create_signature_from_args(Config(args=[4, 3])),
    "softmax":
        Config(signature=UNARY_FLOAT),
    "softplus":
        Config(signature=UNARY_FLOAT),
    "softsign":
        Config(signature=UNARY_FLOAT),
    "sqrt":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "square":
        configure_dtypes(Config(signature=UNARY), ALL_NUMBER_TYPES),
    "squared_difference":
        configure_dtypes(Config(signature=BINARY), ALL_NUMBER_TYPES),
    "subtract":
        configure_dtypes(Config(signature=BINARY), ALL_NUMBER_TYPES),
    "tan":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "tanh":
        configure_dtypes(Config(signature=UNARY), FLOATING_NUMBER_TYPES),
    "top_k":
        Config(signature=UNARY_FLOAT, kwargs=dict(k=2)),
    "truediv":
        configure_dtypes(Config(signature=BINARY), FLOATING_NUMBER_TYPES),
    "unsorted_segment_max":
        UNSORTED_SEGMENT_CONFIG,
    "unsorted_segment_mean":
        UNSORTED_SEGMENT_CONFIG,
    "unsorted_segment_min":
        UNSORTED_SEGMENT_CONFIG,
    "unsorted_segment_prod":
        UNSORTED_SEGMENT_CONFIG,
    "unsorted_segment_sqrt_n":
        UNSORTED_SEGMENT_CONFIG,
    "unsorted_segment_sum":
        UNSORTED_SEGMENT_CONFIG,
    "xdivy":
        configure_dtypes(Config(signature=BINARY), FLOATING_NUMBER_TYPES),
    "xlog1py":
        configure_dtypes(Config(signature=BINARY), FLOATING_NUMBER_TYPES),
    "xlogy":
        configure_dtypes(Config(signature=BINARY), FLOATING_NUMBER_TYPES),
    "zero_fraction":
        configure_dtypes(Config(signature=UNARY), ALL_NUMBER_TYPES),
    "zeta":
        Config(
            signature=BINARY_FLOAT,
            # The function is poorly behaved near zero, so we test this range
            # to avoid outputing all nans.
            generator=lambda *args: tf_utils.uniform(*args, low=3.0, high=4.0)),
}

# Normalize FUNCTION_TO_CONFIGS as described above.
for function_name, configs in FUNCTION_TO_CONFIGS.items():
  if isinstance(configs, Config):
    FUNCTION_TO_CONFIGS[function_name] = {function_name: configs}
  else:
    # Prepend the function names to existing exported names to make the logs
    # readable.
    FUNCTION_TO_CONFIGS[function_name] = {
        f"{function_name}_{config_name}": config
        for config_name, config in FUNCTION_TO_CONFIGS[function_name].items()
    }

flags.DEFINE_list(
    "functions", "abs",
    f"Any of {list(FUNCTION_TO_CONFIGS.keys())}. If more than one function is "
    "provided then len(--target_backends) must be one.")
flags.DEFINE_bool(
    "dynamic_dims", False,
    "Whether or not to compile the layer with dynamic dimensions.")
flags.DEFINE_bool(
    "test_complex", False,
    "Whether or not to test or ignore function signatures with complex types.")
flags.DEFINE_bool(
    'list_functions_with_complex_tests', False,
    'Whether or not to print out all functions with complex inputs '
    '(and skip running the tests).')


def _complex_wrapper(function):
  """Wraps a tf.function to allow compiling functions of complex numbers."""

  def decorator(*args, **kwargs):
    inputs = []
    for real, imag in zip(args[::2], args[1::2]):
      inputs.append(tf.complex(real, imag))
    result = function(*inputs, **kwargs)
    # Directly returning a complex number causes an error.
    return tf.math.real(result) + tf.math.imag(result)

  return decorator


def convert_if_complex(function, config: Config):
  """Compatibility layer for testing complex numbers."""
  if is_complex(config):
    if not all([spec.dtype.is_complex for spec in config.signature]):
      raise NotImplementedError("Signatures with mixed complex and non-complex "
                                "tensor specs are not supported.")

    # Rewrite the signature, replacing all complex tensors with pairs of real
    # and imaginary tensors.
    signature = []
    for spec in config.signature:
      new_dtype = tf.float32 if spec.dtype.size == 8 else tf.float64
      signature.append(tf.TensorSpec(spec.shape, new_dtype))
      signature.append(tf.TensorSpec(spec.shape, new_dtype))

    return _complex_wrapper(function), signature
  else:
    return function, config.signature


def _make_dims_dynamic(spec: tf.TensorSpec) -> tf.TensorSpec:
  return tf.TensorSpec([None] * len(spec.shape), spec.dtype)


def create_function_unittest(config: Config, function_name: str,
                             exported_name: str) -> tf.function:
  """Creates a tf_function_unittest from the provided Config."""
  tf_math_function = getattr(tf.math, function_name)
  tf_math_function, signature = convert_if_complex(tf_math_function, config)
  wrapped_function = lambda *args: tf_math_function(*args, **config.kwargs)

  if FLAGS.dynamic_dims:
    signature = tf_utils.apply_function(signature, _make_dims_dynamic)

  return tf_test_utils.tf_function_unittest(
      input_signature=signature,
      input_generator=config.generator,
      input_args=config.args,
      name=exported_name)(wrapped_function)


class TfMathModule(tf_test_utils.TestModule):

  def __init__(self):
    super().__init__()
    for function_name in FLAGS.functions:
      # pytype: disable=attribute-error
      for exported_name, config in FUNCTION_TO_CONFIGS[function_name].items():
        # pytype: enable=attribute-error
        if is_complex(config) and not FLAGS.test_complex:
          continue
        function_unittest = create_function_unittest(config, function_name,
                                                     exported_name)
        setattr(self, exported_name, function_unittest)


class TfMathTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(
        TfMathModule, exported_names=TfMathModule.get_tf_function_unittests())


def main(argv):
  del argv  # Unused.
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()

  if FLAGS.list_functions_with_complex_tests:
    for function_name, configs in FUNCTION_TO_CONFIGS.items():
      # pytype: disable=attribute-error
      for exported_name, config in configs.items():
        # pytype: enable=attribute-error
        if is_complex(config):
          print(f'    "{function_name}",')
    return

  if len(FLAGS.functions) > 1:
    # We only allow testing multiple functions with a single target backend
    # so that we can store the artifacts under:
    #   'artifacts_dir/multiple_functions__backend/...'
    # We specialize the 'multiple_functions' dir by backend to avoid overwriting
    # tf_input.mlir and iree_input.mlir. These are typically identical across
    # backends, but are not when the functions to compile change per-backend.
    if len(FLAGS.target_backends) != 1:
      raise flags.IllegalFlagValueError(
          "Expected len(target_backends) == 1 when len(functions) > 1, but got "
          f"the following values for target_backends: {FLAGS.target_backends}.")
    function_str = f"multiple_functions__{FLAGS.target_backends[0]}"
  else:
    function_str = FLAGS.functions[0]
  dim_str = "dynamic_dims" if FLAGS.dynamic_dims else "static_dims"
  settings_str = os.path.join(function_str, dim_str)
  # The relative artifacts directory path is calculated from the module name
  # TODO(meadowlark): provide a better way of overridding this default.
  TfMathModule.__name__ = os.path.join("tf", "math", settings_str)

  TfMathTest.generate_unittests(TfMathModule)
  tf.test.main()


if __name__ == "__main__":
  app.run(main)
