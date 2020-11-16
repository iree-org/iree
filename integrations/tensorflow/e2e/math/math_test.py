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


class Config:
  """Specifies a unittest."""

  def __init__(self,
               signature: Sequence[tf.TensorSpec],
               args: Sequence[Any] = None,
               kwargs: Dict[str, Any] = None,
               generator: tf_utils.InputGeneratorType = None):
    self.signature = signature
    self.args = args
    self.kwargs = kwargs if kwargs is not None else dict()
    self.generator = generator


class ArgsConfig(Config):
  """Uses input arguments to generate an input signature."""

  def __init__(self,
               args: Sequence[Any],
               kwargs: Dict[str, Any] = None,
               generator: tf_utils.InputGeneratorType = None):
    signature = tf_utils.apply_function(
        args, lambda x: tf.TensorSpec.from_tensor(tf.convert_to_tensor(x)))
    super().__init__(signature, args, kwargs, generator)


class DtypeConfig(Config):
  """Specifies a unittest for each dtype in dtypes."""

  def __init__(self,
               signature: Sequence[tf.TensorSpec],
               args: Sequence[Any] = None,
               kwargs: Dict[str, Any] = None,
               generator: tf_utils.InputGeneratorType = None,
               dtypes: Sequence[tf.DType] = None):
    super().__init__(signature, args, kwargs, generator)
    for dtype in dtypes:
      if not isinstance(dtype, tf.DType):
        raise TypeError(f"All dtypes must be tf.DTypes, but got '{dtype}'.")
    self.dtypes = dtypes

  def __iter__(self):
    for dtype in self.dtypes:
      signature = tf_utils.apply_function(
          self.signature, lambda spec: tf.TensorSpec(spec.shape, dtype))
      dtype_config = Config(signature, self.args, self.kwargs, self.generator)
      yield dtype.name, dtype_config


# As high as tf goes without breaking.
RANK_7_INPUT = [1, 1, 1, 2, 2, 2, 2]

# "Untyped" signatures – for use with the 'dtypes' kwarg.
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
SEGMENT_CONFIG = ArgsConfig(args=[
    tf.constant([
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [5, 6, 7, 8],
    ], np.float32),
    np.array([0, 0, 1], np.int32)
])
UNSORTED_SEGMENT_CONFIG = ArgsConfig(args=[
    tf.constant([
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [5, 6, 7, 8],
    ], np.float32),
    np.array([0, 0, 1], np.int32),
    2,
])

# A dict mapping tf.math function names to either a single Config (representing
# the signature and other metadata to use to test that function) or a dict
# mapping exported_names to Configs.
#
# Each entry will be normalized to be a dict mapping exported_names to Configs,
# with a default exported_name matching the name of the tf.math function.
FUNCTION_TO_CONFIGS = {
    "abs":
        DtypeConfig(signature=UNARY, dtypes=ALL_NUMBER_TYPES),
    "accumulate_n":
        DtypeConfig(signature=[TERNARY], dtypes=REAL_NUMBER_TYPES),
    "acos":
        Config(signature=UNARY_FLOAT),
    "acosh":
        Config(signature=UNARY_FLOAT, generator=tf_utils.ndarange),
    "add":
        DtypeConfig(signature=BINARY, dtypes=ALL_NUMBER_TYPES),
    "add_n":
        DtypeConfig(signature=[TERNARY], dtypes=REAL_NUMBER_TYPES),
    "angle":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "argmax":
        DtypeConfig(signature=UNARY, dtypes=REAL_NUMBER_TYPES),
    "argmin":
        DtypeConfig(signature=UNARY, dtypes=REAL_NUMBER_TYPES),
    "asin":
        Config(signature=UNARY_FLOAT),
    "asinh":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "atan":
        Config(signature=UNARY_FLOAT),
    "atan2":
        Config(signature=BINARY_FLOAT),
    "atanh":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
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
        ArgsConfig(args=[tf.constant([1, 2, 4]),
                         tf.constant([2, 2, 4])]),
    "conj":
        DtypeConfig(signature=UNARY_COMPLEX, dtypes=ALL_NUMBER_TYPES),
    "cos":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "cosh":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "count_nonzero":
        DtypeConfig(signature=UNARY,
                    generator=tf_utils.ndarange,
                    dtypes=ALL_NUMBER_TYPES),
    "cumprod":
        DtypeConfig(signature=UNARY, dtypes=ALL_NUMBER_TYPES),
    "cumsum":
        DtypeConfig(signature=UNARY, dtypes=ALL_NUMBER_TYPES),
    "cumulative_logsumexp":
        Config(signature=UNARY_FLOAT),
    "digamma":
        Config(signature=UNARY_FLOAT),
    "divide":
        DtypeConfig(signature=BINARY, dtypes=ALL_NUMBER_TYPES),
    "divide_no_nan":
        DtypeConfig(signature=BINARY, dtypes=FLOATING_NUMBER_TYPES),
    "equal":
        DtypeConfig(signature=BINARY, dtypes=REAL_NUMBER_TYPES),
    "erf":
        Config(signature=UNARY_FLOAT),
    "erfc":
        Config(signature=UNARY_FLOAT),
    "erfinv":
        Config(signature=UNARY_FLOAT),
    "exp":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "expm1":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "floor":
        Config(signature=UNARY_FLOAT),
    "floordiv":
        Config(signature=BINARY),
    "floormod":
        Config(signature=BINARY),
    "greater":
        DtypeConfig(signature=BINARY, dtypes=REAL_NUMBER_TYPES),
    "greater_equal":
        DtypeConfig(signature=BINARY, dtypes=REAL_NUMBER_TYPES),
    "igamma":
        Config(signature=BINARY),
    "igammac":
        Config(signature=BINARY),
    "imag":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "in_top_k":
        Config(signature=[tf.TensorSpec([8], tf.int32),
                          tf.TensorSpec([8, 3])],
               kwargs=dict(k=3),
               generator=tf_utils.ndarange),
    "invert_permutation":
        Config(signature=[tf.TensorSpec([8], tf.int32)],
               generator=tf_utils.random_permutation),
    "is_finite":
        ArgsConfig(args=[tf.constant([[1., np.nan], [np.inf, 2.]])]),
    "is_inf":
        ArgsConfig(args=[tf.constant([[1., np.nan], [np.inf, 2.]])]),
    "is_nan":
        ArgsConfig(args=[tf.constant([[1., np.nan], [np.inf, 2.]])]),
    "is_non_decreasing":
        DtypeConfig(signature=UNARY, dtypes=REAL_NUMBER_TYPES),
    "is_strictly_increasing":
        DtypeConfig(signature=UNARY, dtypes=REAL_NUMBER_TYPES),
    "l2_normalize":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "lbeta":
        Config(signature=UNARY_FLOAT),
    "less":
        DtypeConfig(signature=BINARY, dtypes=REAL_NUMBER_TYPES),
    "less_equal":
        DtypeConfig(signature=BINARY, dtypes=REAL_NUMBER_TYPES),
    "lgamma":
        Config(signature=UNARY_FLOAT),
    "log":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "log1p":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
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
        DtypeConfig(signature=BINARY, dtypes=REAL_NUMBER_TYPES),
    "minimum":
        DtypeConfig(signature=BINARY, dtypes=REAL_NUMBER_TYPES),
    "mod":
        DtypeConfig(signature=BINARY,
                    generator=lambda *args: tf_utils.ndarange(*args) + 1,
                    dtypes=REAL_NUMBER_TYPES),
    "multiply":
        DtypeConfig(signature=BINARY, dtypes=ALL_NUMBER_TYPES),
    "multiply_no_nan":
        DtypeConfig(signature=BINARY, dtypes=FLOATING_NUMBER_TYPES),
    "ndtri":
        Config(signature=UNARY_FLOAT),
    "negative":
        DtypeConfig(signature=UNARY, dtypes=ALL_NUMBER_TYPES),
    "nextafter":
        Config(signature=BINARY),
    "not_equal":
        DtypeConfig(signature=BINARY, dtypes=REAL_NUMBER_TYPES),
    "polygamma":
        ArgsConfig(args=[tf.ones(16), tf.linspace(0.5, 4, 16)]),
    "polyval":
        Config(signature=[TERNARY_FLOAT, tf.TensorSpec([])]),
    "pow": {
        # Use ndarange to avoid negative integer powers.
        "int32": Config(signature=BINARY_INT, generator=tf_utils.ndarange),
        "float32": Config(signature=BINARY_FLOAT),
        "complex64": Config(signature=BINARY_COMPLEX),
    },
    "real":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "reciprocal":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "reciprocal_no_nan":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "reduce_all":
        Config(signature=UNARY_BOOL),
    "reduce_any":
        Config(signature=UNARY_BOOL),
    "reduce_euclidean_norm":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "reduce_logsumexp":
        Config(signature=UNARY_FLOAT),
    "reduce_max":
        DtypeConfig(signature=UNARY, dtypes=REAL_NUMBER_TYPES),
    "reduce_mean":
        DtypeConfig(signature=UNARY, dtypes=REAL_NUMBER_TYPES),
    "reduce_min":
        DtypeConfig(signature=UNARY, dtypes=REAL_NUMBER_TYPES),
    "reduce_prod":
        DtypeConfig(signature=UNARY, dtypes=REAL_NUMBER_TYPES),
    "reduce_std":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "reduce_sum":
        DtypeConfig(signature=UNARY, dtypes=REAL_NUMBER_TYPES),
    "reduce_variance":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "rint":
        Config(signature=UNARY_FLOAT),
    "round":
        Config(signature=UNARY_FLOAT),
    "rsqrt":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
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
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "sign":
        DtypeConfig(signature=UNARY, dtypes=ALL_NUMBER_TYPES),
    "sin":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "sinh":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "sobol_sample":
        ArgsConfig(args=[4, 3]),
    "softmax":
        Config(signature=UNARY_FLOAT),
    "softplus":
        Config(signature=UNARY_FLOAT),
    "softsign":
        Config(signature=UNARY_FLOAT),
    "sqrt":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "square":
        DtypeConfig(signature=UNARY, dtypes=ALL_NUMBER_TYPES),
    "squared_difference":
        DtypeConfig(signature=BINARY, dtypes=ALL_NUMBER_TYPES),
    "subtract":
        DtypeConfig(signature=BINARY, dtypes=ALL_NUMBER_TYPES),
    "tan":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "tanh":
        DtypeConfig(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "top_k":
        Config(signature=UNARY_FLOAT, kwargs=dict(k=2)),
    "truediv":
        DtypeConfig(signature=BINARY, dtypes=FLOATING_NUMBER_TYPES),
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
        DtypeConfig(signature=BINARY, dtypes=FLOATING_NUMBER_TYPES),
    "xlog1py":
        DtypeConfig(signature=BINARY, dtypes=FLOATING_NUMBER_TYPES),
    "xlogy":
        DtypeConfig(signature=BINARY, dtypes=FLOATING_NUMBER_TYPES),
    "zero_fraction":
        DtypeConfig(signature=UNARY, dtypes=ALL_NUMBER_TYPES),
    "zeta":
        Config(
            signature=BINARY_FLOAT,
            # The function is poorly behaved near zero, so we test this range
            # to avoid outputing all nans.
            generator=lambda *args: tf_utils.uniform(*args, low=3.0, high=4.0)),
}


def normalize_functions_to_configs(
    functions_to_configs: Dict[str, Union[Config, Dict[str, Config]]]
) -> Dict[str, Dict[str, Config]]:
  # Normalize FUNCTION_TO_CONFIGS as described above.
  normalized = dict()
  for function_name, configs in functions_to_configs.items():
    normalized[function_name] = dict()
    if isinstance(configs, Config):
      normalized[function_name] = {function_name: configs}
    elif isinstance(configs, dict):
      # Prepend the function names to existing exported names to make the logs
      # readable. Split into two expressions so pytype can understand it.
      normalized[function_name] = {
          f"{function_name}_{config_name}": config
          for config_name, config in configs.items()
      }
    else:
      raise TypeError(
          f"Unexpected type for value of FUNCTION_TO_CONFIGS {type(configs)}")
  return normalized


FUNCTION_TO_CONFIGS = normalize_functions_to_configs(FUNCTION_TO_CONFIGS)

flags.DEFINE_list(
    "functions", None,
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


def is_complex(tensors: Union[Sequence[tf.TensorSpec], tf.TensorSpec]) -> bool:
  if isinstance(tensors, Sequence):
    for tensor in tensors:
      if is_complex(tensor):
        return True
    return False
  else:
    return tensors.dtype.is_complex  # pytype: disable=attribute-error


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


def setup_complex_signature(function, signature: Sequence[tf.TensorSpec]):
  """Compatibility layer for testing complex numbers."""
  if not all([spec.dtype.is_complex for spec in signature]):
    raise NotImplementedError("Signatures with mixed complex and non-complex "
                              "tensor specs are not supported.")

  # Rewrite the signature, replacing all complex tensors with pairs of real
  # and imaginary tensors.
  real_imag_signature = []
  for spec in signature:
    new_dtype = tf.float32 if spec.dtype.size == 8 else tf.float64
    real_imag_signature.append(tf.TensorSpec(spec.shape, new_dtype))
    real_imag_signature.append(tf.TensorSpec(spec.shape, new_dtype))

  return _complex_wrapper(function), real_imag_signature


def _make_dims_dynamic(spec: tf.TensorSpec) -> tf.TensorSpec:
  return tf.TensorSpec([None] * len(spec.shape), spec.dtype)


def create_function_unittest(function_name: str, config: Config,
                             exported_name: str) -> tf.function:
  """Creates a tf_function_unittest from the provided Config."""
  function = getattr(tf.math, function_name)
  signature = config.signature
  if is_complex(signature):
    function, signature = setup_complex_signature(function, signature)
  wrapped_function = lambda *args: function(*args, **config.kwargs)

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
      for config_name, config in FUNCTION_TO_CONFIGS[function_name].items():
        # Create a unittest for each dtype specified by 'config'.
        if isinstance(config, DtypeConfig):
          for dtype_name, dtype_config in config:
            if is_complex(dtype_config.signature) and not FLAGS.test_complex:
              continue
            exported_name = f"{config_name}_{dtype_name}"
            function_unittest = create_function_unittest(
                function_name, config, exported_name)
            setattr(self, exported_name, function_unittest)
        # Create a unittest for 'config'.
        else:
          if is_complex(config.signature) and not FLAGS.test_complex:
            continue
          function_unittest = create_function_unittest(
              function_name, config, exported_name=config_name)
          setattr(self, config_name, function_unittest)


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
      for exported_name, config in configs.items():
        if ((config.dtypes is not None and
             any(dtype.is_complex for dtype in config.dtypes)) or
            is_complex(config.signature)):
          print(f'    "{function_name}",')
    return

  if FLAGS.functions is None:
    raise flags.IllegalFlagValueError(
        "'--functions' must be specified if "
        "'--list_functions_with_complex_tests' isn't")

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
