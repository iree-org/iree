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

# Test coverage across backends for e2e tests is defined directly in the BUILD
# files. A coverage table generated from this file can be viewed here:
#   https://google.github.io/iree/tf-e2e-coverage
# Updates made to test suite names should also be reflected here:
#   https://github.com/google/iree/blob/main/scripts/update_e2e_coverage.py

load(
    "//bindings/python:build_defs.oss.bzl",
    "INTREE_TENSORFLOW_PY_DEPS",
    "NUMPY_DEPS",
    "iree_py_binary",
    "iree_py_test",
)
load(
    "//integrations/tensorflow/e2e:iree_e2e_test_suite.bzl",
    "set_difference",
)
load(
    "//integrations/tensorflow/e2e:iree_e2e_cartesian_product_test_suite.bzl",
    "iree_e2e_cartesian_product_test_suite",
)

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

[
    iree_py_binary(
        name = src.replace(".py", "_manual"),
        srcs = [src],
        main = src,
        python_version = "PY3",
        deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
            "//integrations/tensorflow/bindings/python/pyiree/tf/support",
        ],
    )
    for src in glob(
        ["*_test.py"],
        exclude = ["keyword_spotting_streaming_test.py"],
    )
]

# These functions were selected using all of the funcions in the tf.math docs:
#   https://www.tensorflow.org/api_docs/python/tf/math
TF_MATH_FUNCTIONS = [
    "abs",
    "accumulate_n",
    "acos",
    "acosh",
    "add",
    "add_n",
    "angle",
    "argmax",
    "argmin",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bessel_i0",
    "bessel_i0e",
    "bessel_i1",
    "bessel_i1e",
    "betainc",
    "bincount",
    "ceil",
    "confusion_matrix",
    "conj",
    "cos",
    "cosh",
    "count_nonzero",
    "cumprod",
    "cumsum",
    "cumulative_logsumexp",
    "digamma",
    "divide",
    "divide_no_nan",
    "equal",
    "erf",
    "erfc",
    "erfinv",
    "exp",
    "expm1",
    "floor",
    "floordiv",
    "floormod",
    "greater",
    "greater_equal",
    "igamma",
    "igammac",
    "imag",
    "in_top_k",
    "invert_permutation",
    "is_finite",
    "is_inf",
    "is_nan",
    "is_non_decreasing",
    "is_strictly_increasing",
    "l2_normalize",
    "lbeta",
    "less",
    "less_equal",
    "lgamma",
    "log",
    "log1p",
    "log_sigmoid",
    "log_softmax",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "mod",
    "multiply",
    "multiply_no_nan",
    "ndtri",
    "negative",
    "nextafter",
    "not_equal",
    "polygamma",
    "polyval",
    "pow",
    "real",
    "reciprocal",
    "reciprocal_no_nan",
    "reduce_all",
    "reduce_any",
    "reduce_euclidean_norm",
    "reduce_logsumexp",
    "reduce_max",
    "reduce_mean",
    "reduce_min",
    "reduce_prod",
    "reduce_std",
    "reduce_sum",
    "reduce_variance",
    "rint",
    "round",
    "rsqrt",
    "scalar_mul",
    "segment_max",
    "segment_mean",
    "segment_min",
    "segment_prod",
    "segment_sum",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "sobol_sample",
    "softmax",
    "softplus",
    "softsign",
    "sqrt",
    "square",
    "squared_difference",
    "subtract",
    "tan",
    "tanh",
    # "top_k",  # TODO(meadowlark): Enable once list outputs are supported.
    "truediv",
    "unsorted_segment_max",
    "unsorted_segment_mean",
    "unsorted_segment_min",
    "unsorted_segment_prod",
    "unsorted_segment_sqrt_n",
    "unsorted_segment_sum",
    "xdivy",
    "xlog1py",
    "xlogy",
    "zero_fraction",
    "zeta",
]

# keep sorted
TFLITE_FAILING = [
    "abs",  # Failing for integer inputs.
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bessel_i0",
    "bessel_i0e",
    "bessel_i1",
    "bessel_i1e",
    "betainc",
    "bincount",
    "confusion_matrix",
    "cosh",
    "cumprod",
    "cumulative_logsumexp",
    "digamma",
    "divide",  # Failing for integer inputs.
    "erf",
    "erfc",
    "erfinv",
    "expm1",
    "igamma",
    "igammac",
    "in_top_k",
    "invert_permutation",
    "is_finite",
    "is_non_decreasing",
    "is_strictly_increasing",
    "l2_normalize",
    "lbeta",
    "lgamma",
    "log1p",
    "log_sigmoid",
    "ndtri",
    "nextafter",
    "polygamma",
    "polyval",
    "pow",  # Failing for integer inputs.
    "reduce_all",
    "reduce_euclidean_norm",
    "reduce_logsumexp",
    "reduce_mean",
    "reduce_std",
    "reduce_variance",
    "rint",
    "segment_max",
    "segment_mean",
    "segment_min",
    "segment_prod",
    "sign",
    "sinh",
    "sobol_sample",
    "softmax",
    "softplus",
    "softsign",
    "tan",
    "unsorted_segment_max",
    "unsorted_segment_mean",
    "unsorted_segment_min",
    "unsorted_segment_prod",
    "unsorted_segment_sqrt_n",
    "unsorted_segment_sum",
    "xdivy",
    "xlog1py",
    "xlogy",
    "zeta",
]

# Note: The VMLA_FAILING_DYNAMIC specification extends this list. Newly-passing
# functions removed from this list may need to be added to VMLA_FAILING_DYNAMIC.
# keep sorted
VMLA_FAILING = [
    "acosh",
    "argmax",
    "argmin",
    "asin",
    "asinh",
    "atan2",
    "atanh",
    "bessel_i0",
    "bessel_i0e",
    "bessel_i1",
    "bessel_i1e",
    "betainc",
    "bincount",
    "confusion_matrix",
    "cosh",
    "count_nonzero",
    "cumprod",
    "cumulative_logsumexp",
    "digamma",
    "divide",  # Failing for integer inputs because iree doesn't output 'f64'.
    "erf",
    "erfc",
    "erfinv",
    "expm1",
    "igamma",
    "igammac",
    "in_top_k",
    "invert_permutation",
    "is_nan",
    "is_non_decreasing",
    "is_strictly_increasing",
    "ndtri",
    "nextafter",
    "polygamma",
    "pow",  # Failing for integer inputs.
    "reduce_all",
    "reduce_any",
    "reduce_euclidean_norm",
    "reduce_prod",
    "rint",
    "segment_max",
    "segment_mean",
    "segment_min",
    "segment_prod",
    "segment_sum",
    "sign",
    "sobol_sample",
    "softsign",
    "unsorted_segment_max",
    "unsorted_segment_mean",
    "unsorted_segment_min",
    "unsorted_segment_prod",
    "unsorted_segment_sqrt_n",
    "unsorted_segment_sum",
    "xdivy",
    "xlog1py",
    "xlogy",
    "zeta",
]

# Note: The LLVM_FAILING_DYNAMIC specification extends this list. Newly-passing
# functions removed from this list may need to be added to LLVM_FAILING_DYNAMIC.
# keep sorted
LLVM_FAILING = [
    "acos",
    "acosh",
    "argmax",
    "argmin",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bessel_i0",
    "bessel_i0e",
    "bessel_i1",
    "bessel_i1e",
    "betainc",
    "bincount",
    "confusion_matrix",
    "cosh",
    "count_nonzero",
    "cumprod",
    "cumulative_logsumexp",
    "digamma",
    "divide",  # Failing for integer inputs because iree doesn't output 'f64'.
    "erf",
    "erfc",
    "erfinv",
    "expm1",
    "igamma",
    "igammac",
    "in_top_k",
    "invert_permutation",
    "is_nan",
    "is_non_decreasing",
    "is_strictly_increasing",
    "l2_normalize",
    "logical_or",
    "logical_xor",
    "ndtri",
    "nextafter",
    "polygamma",
    "pow",
    "reduce_all",
    "reduce_any",
    "reduce_euclidean_norm",
    "reduce_logsumexp",
    "reduce_prod",
    "reduce_std",
    "reduce_variance",
    "rint",
    "segment_max",
    "segment_mean",
    "segment_min",
    "segment_prod",
    "segment_sum",
    "sign",
    "sobol_sample",
    "softsign",
    "unsorted_segment_max",
    "unsorted_segment_mean",
    "unsorted_segment_min",
    "unsorted_segment_prod",
    "unsorted_segment_sqrt_n",
    "unsorted_segment_sum",
    "xdivy",
    "xlog1py",
    "xlogy",
    "zeta",
]

# Note: The VULKAN_FAILING_DYNAMIC specification extends this list.
# Newly-passing functions removed from this list may need to be added to
# VULKAN_FAILING_DYNAMIC.
# keep sorted
VULKAN_FAILING = [
    "acos",
    "acosh",
    "argmax",
    "argmin",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bessel_i0",
    "bessel_i0e",
    "bessel_i1",
    "bessel_i1e",
    "betainc",
    "bincount",
    "confusion_matrix",
    "cosh",
    "count_nonzero",
    "cumprod",
    "cumsum",
    "cumulative_logsumexp",
    "digamma",
    "divide",  # Failing for integer inputs because iree doesn't output 'f64'.
    "erf",
    "erfc",
    "erfinv",
    "expm1",
    "igamma",
    "igammac",
    "in_top_k",
    "invert_permutation",
    "is_nan",
    "is_non_decreasing",
    "is_strictly_increasing",
    "l2_normalize",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "ndtri",
    "nextafter",
    "polygamma",
    "pow",
    "reduce_all",
    "reduce_any",
    "reduce_euclidean_norm",
    "reduce_logsumexp",
    "reduce_prod",
    "reduce_std",
    "reduce_variance",
    "rint",
    "segment_max",
    "segment_mean",
    "segment_min",
    "segment_prod",
    "segment_sum",
    "sign",
    "sobol_sample",
    "softsign",
    "unsorted_segment_max",
    "unsorted_segment_mean",
    "unsorted_segment_min",
    "unsorted_segment_prod",
    "unsorted_segment_sqrt_n",
    "unsorted_segment_sum",
    "xdivy",
    "xlog1py",
    "xlogy",
    "zeta",
]

# ---- INDIVIDUAL STATIC TESTS ----------------------------------------------- #

# These tests allow us to generate coverage tables and give a finer-grained view
# of the coverage, but are very slow due to bazel overhead, so they are not
# run on the internal or OSS CI.
iree_e2e_cartesian_product_test_suite(
    name = "math_tests",
    srcs = ["math_test.py"],
    failing_configurations = [
        {
            # Failing on TFLite.
            "functions": TFLITE_FAILING,
            "target_backends": "tflite",
        },
        {
            # Failing on vmla.
            "functions": VMLA_FAILING,
            "target_backends": "iree_vmla",
        },
        {
            # Failing on llvm.
            "functions": LLVM_FAILING,
            "target_backends": "iree_llvmjit",
        },
        {
            # Failing on vulkan.
            "functions": VULKAN_FAILING,
            "target_backends": "iree_vulkan",
        },
    ],
    flags_to_values = {
        "reference_backend": "tf",
        "functions": TF_MATH_FUNCTIONS,
        "dynamic_dims": False,
        "test_complex": False,
        "target_backends": [
            "tf",
            "tflite",
            "iree_vmla",
            "iree_llvmjit",
            "iree_vulkan",
        ],
    },
    main = "math_test.py",
    tags = [
        "manual",
        "nokokoro",
        "notap",
    ],
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)

# ---- MULTIPLE STATIC TESTS ------------------------------------------------ #

# These tests compile all functions in tf.math at once for testing so that
# we can run them on the CI with 5 additional targets instead of 640.

# Failing on Kokoro.
TFLITE_FAILING_KOKORO = TFLITE_FAILING + ["angle"]

# TODO(#3810) 'multiply' outputs all zeros when compiled with other functions.
VMLA_FAILING_MULTIPLE = VMLA_FAILING + ["multiply"]

# TODO(#3810) Including 'square' causes error: Recieved signal 11.
LLVM_FAILING_MULTIPLE = LLVM_FAILING + ["square"]

# TODO(#3810) Including 'square' causes error: Recieved signal 11.
VULKAN_FAILING_MULTIPLE = VULKAN_FAILING + ["square"]

[
    iree_py_test(
        name = "math_tests_multiple__{}".format(target_backend),
        srcs = ["math_test.py"],
        args = [
            "--reference_backend=tf",
            "--target_backends={}".format(target_backend),
            "--functions={}".format(",".join(functions)),
            "--dynamic_dims=False",
        ],
        main = "math_test.py",
        deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
            "//integrations/tensorflow/bindings/python/pyiree/tf/support",
        ],
    )
    for target_backend, functions in dict(
        iree_llvmjit = set_difference(TF_MATH_FUNCTIONS, LLVM_FAILING_MULTIPLE),
        iree_vmla = set_difference(TF_MATH_FUNCTIONS, VMLA_FAILING_MULTIPLE),
        iree_vulkan = set_difference(TF_MATH_FUNCTIONS, VULKAN_FAILING_MULTIPLE),
        tf = TF_MATH_FUNCTIONS,
        tflite = set_difference(TF_MATH_FUNCTIONS, TFLITE_FAILING_KOKORO),
    ).items()
]

# ---- INDIVIDUAL DYNAMIC TESTS ---------------------------------------------- #

# keep sorted
VMLA_FAILING_DYNAMIC = VMLA_FAILING + [
    "angle",
    "ceil",
    "cumsum",
    "divide_no_nan",
    "equal",
    "floormod",
    "imag",
    "lbeta",
    "lgamma",
    "log_sigmoid",
    "log1p",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "mod",
    "multiply_no_nan",
    "not_equal",
    "reciprocal_no_nan",
    "reduce_mean",
    "reduce_std",
    "reduce_variance",
    "softplus",
    "zero_fraction",
]

# keep sorted
LLVM_FAILING_DYNAMIC = LLVM_FAILING + [
    "accumulate_n",
    "add",
    "add_n",
    "angle",
    "cumsum",
    "divide",
    "divide_no_nan",
    "equal",
    "floordiv",
    "floormod",
    "greater",
    "greater_equal",
    "imag",
    "is_finite",
    "is_inf",
    "lbeta",
    "less",
    "less_equal",
    "lgamma",
    "log_sigmoid",
    "log_softmax",
    "log1p",
    "logical_and",
    "logical_not",
    "maximum",
    "minimum",
    "mod",
    "multiply",
    "multiply_no_nan",
    "not_equal",
    "polyval",
    "reciprocal",
    "reciprocal_no_nan",
    "reduce_mean",
    "reduce_sum",
    "scalar_mul",
    "sigmoid",
    "sinh",
    "softmax",
    "softplus",
    "square",
    "squared_difference",
    "subtract",
    "tan",
    "truediv",
    "zero_fraction",
]

# keep sorted
VULKAN_FAILING_DYNAMIC = VULKAN_FAILING + [
    "abs",
    "accumulate_n",
    "add",
    "add_n",
    "angle",
    "ceil",
    "cos",
    "divide",
    "divide_no_nan",
    "equal",
    "exp",
    "floor",
    "floordiv",
    "floormod",
    "greater",
    "greater_equal",
    "imag",
    "is_finite",
    "is_inf",
    "lbeta",
    "less",
    "less_equal",
    "lgamma",
    "log",
    "log_sigmoid",
    "log_softmax",
    "log1p",
    "maximum",
    "minimum",
    "mod",
    "multiply",
    "multiply_no_nan",
    "negative",
    "not_equal",
    "polyval",
    "reciprocal",
    "reciprocal_no_nan",
    "reduce_max",
    "reduce_mean",
    "reduce_min",
    "reduce_sum",
    "rsqrt",
    "scalar_mul",
    "sigmoid",
    "sin",
    "sinh",
    "softmax",
    "softplus",
    "sqrt",
    "square",
    "squared_difference",
    "subtract",
    "tan",
    "tanh",
    "truediv",
    "zero_fraction",
]

# These tests allow us to generate coverage tables and give a finer-grained view
# of the coverage, but are very slow due to bazel overhead, so they are not
# run on the internal or OSS CI.
iree_e2e_cartesian_product_test_suite(
    name = "math_dynamic_dims_tests",
    srcs = ["math_test.py"],
    failing_configurations = [
        {
            # TFLite does not support dynamic shapes.
            "target_backends": "tflite",
        },
        {
            # Failing on vmla.
            "functions": VMLA_FAILING_DYNAMIC,
            "target_backends": "iree_vmla",
        },
        {
            # Failing on llvm.
            "functions": LLVM_FAILING_DYNAMIC,
            "target_backends": "iree_llvmjit",
        },
        {
            # Failing on vulkan.
            "functions": VULKAN_FAILING_DYNAMIC,
            "target_backends": "iree_vulkan",
        },
    ],
    flags_to_values = {
        "reference_backend": "tf",
        "functions": TF_MATH_FUNCTIONS,
        "dynamic_dims": True,
        "test_complex": False,
        "target_backends": [
            "tf",
            "tflite",
            "iree_vmla",
            "iree_llvmjit",
            "iree_vulkan",
        ],
    },
    main = "math_test.py",
    tags = [
        "manual",
        "nokokoro",
        "notap",
    ],
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)

# ---- MULTIPLE DYNAMIC TESTS ----------------------------------------------- #

# These tests compile all functions in tf.math at once for testing so that
# we can run them on the CI with 4 additional targets instead of 512.

# TODO(#3810) 'multiply' outputs all zeros when compiled with other functions.
VMLA_FAILING_DYNAMIC_MULTIPLE = VMLA_FAILING_DYNAMIC + ["multiply"]

[
    iree_py_test(
        name = "math_dynamic_dims_tests_multiple__{}".format(target_backend),
        srcs = ["math_test.py"],
        args = [
            "--reference_backend=tf",
            "--target_backends={}".format(target_backend),
            "--functions={}".format(",".join(functions)),
            "--dynamic_dims=False",
        ],
        main = "math_test.py",
        deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
            "//integrations/tensorflow/bindings/python/pyiree/tf/support",
        ],
    )
    for target_backend, functions in dict(
        iree_llvmjit = set_difference(TF_MATH_FUNCTIONS, LLVM_FAILING_DYNAMIC),
        iree_vmla = set_difference(TF_MATH_FUNCTIONS, VMLA_FAILING_DYNAMIC_MULTIPLE),
        iree_vulkan = set_difference(TF_MATH_FUNCTIONS, VULKAN_FAILING_DYNAMIC),
        tf = TF_MATH_FUNCTIONS,
    ).items()
]

# ---- INDIVIDUAL COMPLEX TESTS ---------------------------------------------- #

# This list was generated by running:
#   bazel run integrations/tensorflow/e2e/math:math_test_manual -- --list_functions_with_complex_tests
COMPLEX_FUNCTIONS = [
    "abs",
    "add",
    "angle",
    "asinh",
    "atanh",
    "conj",
    "cos",
    "cosh",
    "count_nonzero",
    "cumprod",
    "cumsum",
    "divide",
    "divide_no_nan",
    "exp",
    "expm1",
    "imag",
    "l2_normalize",
    "log",
    "log1p",
    "multiply",
    "multiply_no_nan",
    "negative",
    "pow",
    "real",
    "reciprocal",
    "reciprocal_no_nan",
    "reduce_euclidean_norm",
    "reduce_std",
    "reduce_variance",
    "rsqrt",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "squared_difference",
    "subtract",
    "tan",
    "tanh",
    "truediv",
    "xdivy",
    "xlog1py",
    "xlogy",
    "zero_fraction",
]

# keep sorted
FAILING_COMPLEX = [
    "angle",
    "cos",
    "cumsum",
    "divide_no_nan",
    "log",
    "log1p",
    "multiply_no_nan",
    "negative",
    "reciprocal",
    "reciprocal_no_nan",
    "reduce_std",
    "reduce_variance",
    "rsqrt",
    "sigmoid",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "zero_fraction",
]

VMLA_FAILING_COMPLEX = VMLA_FAILING + FAILING_COMPLEX

LLVM_FAILING_COMPLEX = LLVM_FAILING + FAILING_COMPLEX

VULKAN_FAILING_COMPLEX = VULKAN_FAILING + FAILING_COMPLEX

# These tests allow us to generate coverage tables and give a finer-grained view
# of the coverage, but are very slow due to bazel overhead, so they are not
# run on the internal or OSS CI.
iree_e2e_cartesian_product_test_suite(
    name = "math_complex_tests",
    srcs = ["math_test.py"],
    failing_configurations = [
        {
            # TFLite does not support complex numbers.
            "target_backends": "tflite",
        },
        {
            # Failing on vmla.
            "functions": VMLA_FAILING_COMPLEX,
            "target_backends": "iree_vmla",
        },
        {
            # Failing on llvm.
            "functions": LLVM_FAILING_COMPLEX,
            "target_backends": "iree_llvmjit",
        },
        {
            # Failing on vulkan.
            "functions": VULKAN_FAILING_COMPLEX,
            "target_backends": "iree_vulkan",
        },
    ],
    flags_to_values = {
        "reference_backend": "tf",
        "functions": COMPLEX_FUNCTIONS,
        "dynamic_dims": False,
        "test_complex": True,
        "target_backends": [
            "tf",
            "tflite",
            "iree_vmla",
            "iree_llvmjit",
            "iree_vulkan",
        ],
    },
    main = "math_test.py",
    tags = [
        "manual",
        "nokokoro",
        "notap",
    ],
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)

# ---- MULTIPLE COMPLEX TESTS ----------------------------------------------- #

# These tests compile all functions in tf.math at once for testing so that
# we can run them on the CI with 4 additional targets instead of 512.

# TODO(#3810) 'multiply' outputs all zeros when compiled with other functions.
VMLA_FAILING_COMPLEX_MULTIPLE = VMLA_FAILING_COMPLEX + ["multiply"]

# TODO(#3810) Including 'square' causes error: Recieved signal 11.
LLVM_FAILING_COMPLEX_MULTIPLE = LLVM_FAILING_COMPLEX + ["square"]

# TODO(#3810) Including 'square' causes error: Recieved signal 11.
VULKAN_FAILING_COMPLEX_MULTIPLE = VULKAN_FAILING_COMPLEX + ["square"]

[
    iree_py_test(
        name = "math_complex_tests_multiple__{}".format(target_backend),
        srcs = ["math_test.py"],
        args = [
            "--reference_backend=tf",
            "--target_backends={}".format(target_backend),
            "--functions={}".format(",".join(functions)),
            "--dynamic_dims=False",
        ],
        main = "math_test.py",
        deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
            "//integrations/tensorflow/bindings/python/pyiree/tf/support",
        ],
    )
    for target_backend, functions in dict(
        iree_llvmjit = set_difference(TF_MATH_FUNCTIONS, LLVM_FAILING_COMPLEX_MULTIPLE),
        iree_vmla = set_difference(TF_MATH_FUNCTIONS, VMLA_FAILING_COMPLEX_MULTIPLE),
        iree_vulkan = set_difference(TF_MATH_FUNCTIONS, VULKAN_FAILING_COMPLEX_MULTIPLE),
        tf = TF_MATH_FUNCTIONS,
    ).items()
]
