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

load(
    "//bindings/python:build_defs.oss.bzl",
    "INTREE_TENSORFLOW_PY_DEPS",
    "NUMPY_DEPS",
    "PYBIND_COPTS",
    "PYBIND_EXTENSION_COPTS",
    "PYBIND_FEATURES",
    "iree_py_extension",
    "iree_py_library",
    "iree_py_test",
    "pybind_cc_library",
)

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

config_setting(
    name = "disable_kernels",
    define_values = {"PYIREE_TF_DISABLE_KERNELS": "1"},
)

# Runtime deps needed to compile the tensorflow compiler.
# As of 2020-04-13, robust constant folding is dependent on the legacy
# TensorFlow executor, which requires kernels to operate on/simplify
# the graph. This should become less necessary as more robust support
# is implemented as part of the MLIR-based tf2xla bridge. This adds
# about ~350 files to the system, many of which are quite heavy to
# compile. Excluding them disables TensorFlow constant propagation,
# which can cause non-optimized binaries (and tickle bugs and unimplemented
# features). However, it is allowed, especially for development because it
# is such a burden to build them. Disable kernels with this command line
# options:
#   --define=PYIREE_TF_DISABLE_KERNELS=1
# See: https://github.com/google/iree/issues/1506
SAVED_MODEL_TF_RUNTIME_DEPS = [
    "@org_tensorflow//tensorflow/core:ops",
] + select({
    ":disable_kernels": [],
    "//conditions:default": [
        "@org_tensorflow//tensorflow/core/kernels:array",
        "@org_tensorflow//tensorflow/core/kernels:math",
    ],
})

# TODO: Isolate SignatureDef SavedModel support into its own library to decrease buildtime cost.
# While it would be nice to simply depend on TensorFlow, manually paring
# down the dependencies significantly reduces the build time that this adds.
#
# Baseline: 449s
# SignatureDef SavedModels: 546s – 22% increase in build time.
# SignatureDef SavedModels + Deps for MobileBert: 572s – 27% increase in build time.
# TF OpenSource: 664s – 49% increase in build time.
SIGNATURE_DEF_SAVED_MODEL_TF_RUNTIME_DEPS = [
    # Deps for SignatureDef SavedModels:
    "@org_tensorflow//tensorflow/core:direct_session",
    "@org_tensorflow//tensorflow/core/kernels:resource_variable_ops",  #  VarHandleOp
    "@org_tensorflow//tensorflow/core/kernels:regex_full_match_op",  # StaticRegexFullMatch
    "@org_tensorflow//tensorflow/core/kernels:string_join_op",  # StringJoin
    "@org_tensorflow//tensorflow/core/kernels:save_op",  # SharedFilename
    "@org_tensorflow//tensorflow/core/kernels:save_restore_v2_ops",  # SaveV2

    # Deps for MobileBert:
    "@org_tensorflow//tensorflow/core/kernels:parameterized_truncated_normal_op",  # TruncatedNormal
    "@org_tensorflow//tensorflow/core/kernels:state",  # Assign.
    "@org_tensorflow//tensorflow/core/kernels:logging_ops",  # Assert
    "@org_tensorflow//tensorflow/core/kernels:bias_op",  # BiasAdd
    "@org_tensorflow//tensorflow/core/kernels:softmax_op",  # Softmax
    "@org_tensorflow//tensorflow/core/kernels:relu_op",  # Relu
]

TF_XLA_PASS_DEPS = [
    "//integrations/tensorflow/compiler:tensorflow",
    "@org_tensorflow//tensorflow/compiler/mlir/xla:xla_legalize_tf",
]

iree_py_library(
    name = "compiler",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":binding",
        "//bindings/python:pathsetup",  # build_cleaner: keep
        "//integrations/tensorflow/bindings/python:pathsetup",  # build_cleaner: keep
    ],
)

iree_py_extension(
    name = "binding",
    srcs = [
        "initialize_module.cc",
    ],
    copts = PYBIND_COPTS + PYBIND_EXTENSION_COPTS,
    features = PYBIND_FEATURES,
    linkstatic = 1,
    win_def_file = "export.def",
    deps = [
        ":compiler_library",
        "//bindings/python/pyiree/common",
        "//bindings/python/pyiree/compiler:compiler_library",
    ],
)

pybind_cc_library(
    name = "compiler_library",
    srcs = [
        "register_tensorflow.cc",
    ],
    hdrs = [
        "register_tensorflow.h",
    ],
    deps = SAVED_MODEL_TF_RUNTIME_DEPS + TF_XLA_PASS_DEPS + SIGNATURE_DEF_SAVED_MODEL_TF_RUNTIME_DEPS + [
        "//bindings/python/pyiree/common",
        "//bindings/python/pyiree/compiler:compiler_library",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@org_tensorflow//tensorflow/cc/saved_model:loader_lite",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tf_saved_model_passes",
        "@org_tensorflow//tensorflow/core:core_cpu",
    ],
)

iree_py_test(
    name = "saved_model_test",
    srcs = ["saved_model_test.py"],
    python_version = "PY3",
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "@absl_py//absl/testing:absltest",
        "//integrations/tensorflow/bindings/python/pyiree/tf/compiler",
    ],
)

iree_py_test(
    name = "signature_def_saved_model_test",
    srcs = ["signature_def_saved_model_test.py"],
    python_version = "PY3",
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "@absl_py//absl/testing:absltest",
        "//integrations/tensorflow/bindings/python/pyiree/tf/compiler",
    ],
)
