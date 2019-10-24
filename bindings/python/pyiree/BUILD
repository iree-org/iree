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

load("//iree:build_defs.bzl", "NUMPY_DEPS", "iree_py_extension")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

COMPILER_DEPS = [
    "//iree/compiler/Translation/Sequencer",
    "//iree/compiler/Translation/Interpreter",
    "//iree/compiler/Translation/SPIRV",
]

DRIVER_DEPS = [
    "//iree/hal/interpreter:interpreter_driver_module",
    "//iree/hal/vulkan:vulkan_driver_module",
]

py_binary(
    name = "everything_for_colab",
    srcs = ["dummy.py"],
    main = "dummy.py",
    python_version = "PY3",
    deps = [
        ":pyiree",  # build_cleaner: keep
        "//bindings/python:pathsetup",  # build_cleaner: keep
    ],
)

py_library(
    name = "pyiree",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":binding",
    ],
)

iree_py_extension(
    name = "binding",
    srcs = [
        "binding.cc",
        "binding.h",
        "compiler.cc",
        "compiler.h",
        "hal.cc",
        "hal.h",
        "initialize.cc",
        "initialize.h",
        "rt.cc",
        "rt.h",
        "status_utils.cc",
        "status_utils.h",
        "vm.cc",
        "vm.h",
    ],
    copts = [
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "//iree/base:api",
        "//iree/base:init",
        "//iree/base:status",
        "//iree/hal:api",
        "//iree/rt:api",
        "//iree/schemas",
        "//iree/vm:api",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Parser",
        "@iree_pybind11//:pybind11",
    ] + COMPILER_DEPS + DRIVER_DEPS,
)

py_test(
    name = "compiler_test",
    srcs = ["compiler_test.py"],
    python_version = "PY3",
    deps = [
        ":binding",
        "//bindings/python:pathsetup",  # build_cleaner: keep
        "@absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "hal_test",
    srcs = ["hal_test.py"],
    python_version = "PY3",
    deps = [
        ":binding",
        "//bindings/python:pathsetup",  # build_cleaner: keep
        "@absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "runtime_test",
    srcs = ["runtime_test.py"],
    python_version = "PY3",
    deps = NUMPY_DEPS + [
        ":binding",
        "//bindings/python:pathsetup",  # build_cleaner: keep
        "@absl_py//absl/testing:absltest",
    ],
)
