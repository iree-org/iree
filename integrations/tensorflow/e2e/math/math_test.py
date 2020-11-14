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
  """Specifies a set of unittests.

  If dtypes is not specified, then only one unittest will be created. If
  dtypes is specified then the other settings will be duplicated across it.
  """

  def __init__(self,
               signature: Sequence[tf.TensorSpec],
               args: Sequence[Any] = None,
               kwargs: Dict[str, Any] = None,
               generator: tf_utils.InputGeneratorType = None,
               dtypes: Sequence[tf.DType] = None):
    self.signature = signature
    self.args = args
    self.kwargs = kwargs if kwargs is not None else dict()
    self.generator = generator
    self.dtypes = dtypes

  def iterate_dtypes(self):
    for dtype in self.dtypes:
      if not isinstance(dtype, tf.DType):
        raise TypeError(f"All dtypes must be tf.DTypes, but got '{dtype}'.")
      signature = tf_utils.apply_function(
          self.signature, lambda spec: tf.TensorSpec(spec.shape, dtype))
      yield dtype.name, signature


class ArgsConfig(Config):
  """Uses input arguments to generate an input signature."""

  def __init__(self,
               args: Sequence[Any],
               kwargs: Dict[str, Any] = None,
               generator: tf_utils.InputGeneratorType = None):
    signature = tf_utils.apply_function(
        args, lambda x: tf.TensorSpec.from_tensor(tf.convert_to_tensor(x)))
    super().__init__(signature, args, kwargs, generator)


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
        Config(signature=UNARY, dtypes=ALL_NUMBER_TYPES),
    "accumulate_n":
        Config(signature=[TERNARY], dtypes=REAL_NUMBER_TYPES),
    "acos":
        Config(signature=UNARY_FLOAT),
    "acosh":
        Config(signature=UNARY_FLOAT, generator=tf_utils.ndarange),
    "add":
        Config(signature=BINARY, dtypes=ALL_NUMBER_TYPES),
    "add_n":
        Config(signature=[TERNARY], dtypes=REAL_NUMBER_TYPES),
    "angle":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "argmax":
        Config(signature=UNARY, dtypes=REAL_NUMBER_TYPES),
    "argmin":
        Config(signature=UNARY, dtypes=REAL_NUMBER_TYPES),
    "asin":
        Config(signature=UNARY_FLOAT),
    "asinh":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "atan":
        Config(signature=UNARY_FLOAT),
    "atan2":
        Config(signature=BINARY_FLOAT),
    "atanh":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
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
        Config(signature=UNARY_COMPLEX, dtypes=ALL_NUMBER_TYPES),
    "cos":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "cosh":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "count_nonzero":
        Config(signature=UNARY,
               generator=tf_utils.ndarange,
               dtypes=ALL_NUMBER_TYPES),
    "cumprod":
        Config(signature=UNARY, dtypes=ALL_NUMBER_TYPES),
    "cumsum":
        Config(signature=UNARY, dtypes=ALL_NUMBER_TYPES),
    "cumulative_logsumexp":
        Config(signature=UNARY_FLOAT),
    "digamma":
        Config(signature=UNARY_FLOAT),
    "divide":
        Config(signature=BINARY, dtypes=ALL_NUMBER_TYPES),
    "divide_no_nan":
        Config(signature=BINARY, dtypes=FLOATING_NUMBER_TYPES),
    "equal":
        Config(signature=BINARY, dtypes=REAL_NUMBER_TYPES),
    "erf":
        Config(signature=UNARY_FLOAT),
    "erfc":
        Config(signature=UNARY_FLOAT),
    "erfinv":
        Config(signature=UNARY_FLOAT),
    "exp":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "expm1":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "floor":
        Config(signature=UNARY_FLOAT),
    "floordiv":
        Config(signature=BINARY),
    "floormod":
        Config(signature=BINARY),
    "greater":
        Config(signature=BINARY, dtypes=REAL_NUMBER_TYPES),
    "greater_equal":
        Config(signature=BINARY, dtypes=REAL_NUMBER_TYPES),
    "igamma":
        Config(signature=BINARY),
    "igammac":
        Config(signature=BINARY),
    "imag":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
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
        Config(signature=UNARY, dtypes=REAL_NUMBER_TYPES),
    "is_strictly_increasing":
        Config(signature=UNARY, dtypes=REAL_NUMBER_TYPES),
    "l2_normalize":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "lbeta":
        Config(signature=UNARY_FLOAT),
    "less":
        Config(signature=BINARY, dtypes=REAL_NUMBER_TYPES),
    "less_equal":
        Config(signature=BINARY, dtypes=REAL_NUMBER_TYPES),
    "lgamma":
        Config(signature=UNARY_FLOAT),
    "log":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "log1p":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
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
        Config(signature=BINARY, dtypes=REAL_NUMBER_TYPES),
    "minimum":
        Config(signature=BINARY, dtypes=REAL_NUMBER_TYPES),
    "mod":
        Config(signature=BINARY,
               generator=lambda *args: tf_utils.ndarange(*args) + 1,
               dtypes=REAL_NUMBER_TYPES),
    "multiply":
        Config(signature=BINARY, dtypes=ALL_NUMBER_TYPES),
    "multiply_no_nan":
        Config(signature=BINARY, dtypes=FLOATING_NUMBER_TYPES),
    "ndtri":
        Config(signature=UNARY_FLOAT),
    "negative":
        Config(signature=UNARY, dtypes=ALL_NUMBER_TYPES),
    "nextafter":
        Config(signature=BINARY),
    "not_equal":
        Config(signature=BINARY, dtypes=REAL_NUMBER_TYPES),
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
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "reciprocal":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "reciprocal_no_nan":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "reduce_all":
        Config(signature=UNARY_BOOL),
    "reduce_any":
        Config(signature=UNARY_BOOL),
    "reduce_euclidean_norm":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "reduce_logsumexp":
        Config(signature=UNARY_FLOAT),
    "reduce_max":
        Config(signature=UNARY, dtypes=REAL_NUMBER_TYPES),
    "reduce_mean":
        Config(signature=UNARY, dtypes=REAL_NUMBER_TYPES),
    "reduce_min":
        Config(signature=UNARY, dtypes=REAL_NUMBER_TYPES),
    "reduce_prod":
        Config(signature=UNARY, dtypes=REAL_NUMBER_TYPES),
    "reduce_std":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "reduce_sum":
        Config(signature=UNARY, dtypes=REAL_NUMBER_TYPES),
    "reduce_variance":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "rint":
        Config(signature=UNARY_FLOAT),
    "round":
        Config(signature=UNARY_FLOAT),
    "rsqrt":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
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
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "sign":
        Config(signature=UNARY, dtypes=ALL_NUMBER_TYPES),
    "sin":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "sinh":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "sobol_sample":
        ArgsConfig(args=[4, 3]),
    "softmax":
        Config(signature=UNARY_FLOAT),
    "softplus":
        Config(signature=UNARY_FLOAT),
    "softsign":
        Config(signature=UNARY_FLOAT),
    "sqrt":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "square":
        Config(signature=UNARY, dtypes=ALL_NUMBER_TYPES),
    "squared_difference":
        Config(signature=BINARY, dtypes=ALL_NUMBER_TYPES),
    "subtract":
        Config(signature=BINARY, dtypes=ALL_NUMBER_TYPES),
    "tan":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "tanh":
        Config(signature=UNARY, dtypes=FLOATING_NUMBER_TYPES),
    "top_k":
        Config(signature=UNARY_FLOAT, kwargs=dict(k=2)),
    "truediv":
        Config(signature=BINARY, dtypes=FLOATING_NUMBER_TYPES),
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
        Config(signature=BINARY, dtypes=FLOATING_NUMBER_TYPES),
    "xlog1py":
        Config(signature=BINARY, dtypes=FLOATING_NUMBER_TYPES),
    "xlogy":
        Config(signature=BINARY, dtypes=FLOATING_NUMBER_TYPES),
    "zero_fraction":
        Config(signature=UNARY, dtypes=ALL_NUMBER_TYPES),
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
  elif isinstance(configs, dict):
    # Prepend the function names to existing exported names to make the logs
    # readable. Split into two expressions so pytype can understand it.
    normalized_dict = {
        f"{function_name}_{config_name}": config
        for config_name, config in configs.items()
    }
    FUNCTION_TO_CONFIGS[function_name] = normalized_dict
  else:
    raise TypeError(
        f"Unexpected type for value of FUNCTION_TO_CONFIGS {type(configs)}")

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


def create_function_unittest(
    function_name: str,
    config: Config,
    exported_name: str,
    override_signature: Sequence[tf.TensorSpec] = None) -> tf.function:
  """Creates a tf_function_unittest from the provided Config."""
  signature = config.signature
  if override_signature is not None:
    signature = override_signature

  function = getattr(tf.math, function_name)
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
      # pytype: disable=attribute-error
      for config_name, config in FUNCTION_TO_CONFIGS[function_name].items():
        # pytype: enable=attribute-error

        # Create a unittest for each dtype specified by 'config'.
        if config.dtypes is not None:
          for dtype_name, signature in config.iterate_dtypes():
            if is_complex(signature) and not FLAGS.test_complex:
              continue
            exported_name = f"{config_name}_{dtype_name}"
            function_unittest = create_function_unittest(
                function_name, config, exported_name, signature)
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
      # pytype: disable=attribute-error
      for exported_name, config in configs.items():
        # pytype: enable=attribute-error
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
