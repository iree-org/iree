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

load(
    "//bindings/python:build_defs.oss.bzl",
    "NUMPY_DEPS",
    "PYBIND_COPTS",
    "PYBIND_EXTENSION_COPTS",
    "PYBIND_FEATURES",
    "PYBIND_REGISTER_MLIR_PASSES",
    "PYTHON_CPP_EXTRA_DEPS",
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

COMPILER_DEPS = [
    # Transforms.
    "//iree/compiler/Dialect/Flow/Transforms",
    "//iree/compiler/Dialect/HAL/Target",
    "//iree/compiler/Dialect/HAL/Transforms",
    "//iree/compiler/Dialect/Shape/IR",
    "//iree/compiler/Dialect/Shape/Transforms",
    "//iree/compiler/Dialect/VM/Transforms",

    # Targets.
    "//iree/compiler/Dialect/HAL/Target/VMLA",
    "//iree/compiler/Dialect/HAL/Target/LLVM:LLVMAOT",
    "//iree/compiler/Dialect/HAL/Target/LLVM:LLVMIR",
    "//iree/compiler/Dialect/HAL/Target/VulkanSPIRV",
    "//iree/compiler/Dialect/VM/Target:init_targets",
    "//iree/compiler/Dialect/VM/Target/Bytecode",
]

iree_py_library(
    name = "compiler",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = PYTHON_CPP_EXTRA_DEPS + [
        ":binding",
        "//bindings/python:pathsetup",  # build_cleaner: keep
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
    ],
)

pybind_cc_library(
    name = "compiler_library",
    srcs = [
        "compiler.cc",
    ],
    hdrs = [
        "compiler.h",
    ],
    copts = PYBIND_REGISTER_MLIR_PASSES,
    deps = COMPILER_DEPS + [
        "//bindings/python/pyiree/common",
        "//iree/tools:init_compiler_modules",
        "//iree/tools:init_iree_passes_and_dialects",
        "//iree/tools:init_mlir_passes_and_dialects",
        "//iree/base:localfile",
        "//iree/tools:init_targets",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
    ],
)

iree_py_test(
    name = "compiler_test",
    srcs = ["compiler_test.py"],
    python_version = "PY3",
    deps = NUMPY_DEPS + [
        "//bindings/python:pathsetup",  # build_cleaner: keep
        "@absl_py//absl/testing:absltest",
        "//bindings/python/pyiree/compiler",
    ],
)
