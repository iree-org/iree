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
    "//iree:build_defs.bzl",
    "NUMPY_DEPS",
    "PYTHON_HEADERS_DEPS",
    "iree_py_extension",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

DEFAULT_COPTS = [
    "-fexceptions",
]

DEFAULT_FEATURES = [
    "-use_header_modules",  # Incompatible with exceptions builds.
]

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
        ":compiler",
        "//bindings/python:pathsetup",  # build_cleaner: keep
    ] + select({
        "//iree:enable_tensorflow": [
            "//bindings/python/pyiree/tf_interop:test_utils",
            "//bindings/python/pyiree/tf_interop:tf_test_driver",
        ],
        "//conditions:default": [
        ],
    }),
)

py_library(
    name = "compiler",
    srcs = ["compiler.py"],
    srcs_version = "PY3",
    deps = [
        ":binding",
        "//bindings/python:pathsetup",  # build_cleaner: keep
    ],
)

cc_library(
    name = "base",
    srcs = [
        "compiler.cc",
        "hal.cc",
        "rt.cc",
        "status_utils.cc",
        "vm.cc",
    ],
    hdrs = [
        "binding.h",
        "compiler.h",
        "hal.h",
        "rt.h",
        "status_utils.h",
        "vm.h",
    ],
    copts = DEFAULT_COPTS,
    features = DEFAULT_FEATURES,
    deps = [
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "//iree/compiler/Utils",
        "@local_config_mlir//:IR",
        "//iree/base:api",
        "//iree/base:status",
        "//iree/hal:api",
        "//iree/rt:api",
        "//iree/schemas",
        "//iree/vm:api",
        "@llvm//:support",
        "@local_config_mlir//:Parser",
        "@local_config_mlir//:Pass",
        "@iree_pybind11//:pybind11",
    ] + COMPILER_DEPS + DRIVER_DEPS + PYTHON_HEADERS_DEPS,
)

iree_py_extension(
    name = "binding",
    srcs = [
        "initialize_module.cc",
    ],
    copts = DEFAULT_COPTS,
    features = DEFAULT_FEATURES,
    linkstatic = 1,
    win_def_file = "export.def",
    deps = [
        ":base",
        "//bindings/python/pyiree/tf_interop",
        "//iree/base:initializer",
        "//iree/base:tracing",
        "@com_google_tracing_framework_cpp//:tracing_framework_bindings_cpp",
    ],
)

py_test(
    name = "compiler_test",
    srcs = ["compiler_test.py"],
    python_version = "PY3",
    deps = [
        "//bindings/python:pathsetup",  # build_cleaner: keep
        "//bindings/python/pyiree",
        "@absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "hal_test",
    srcs = ["hal_test.py"],
    python_version = "PY3",
    deps = [
        "//bindings/python:pathsetup",  # build_cleaner: keep
        "//bindings/python/pyiree",
        "@absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "runtime_test",
    srcs = ["runtime_test.py"],
    python_version = "PY3",
    deps = NUMPY_DEPS + [
        "//bindings/python:pathsetup",  # build_cleaner: keep
        "@absl_py//absl/testing:absltest",
        "//bindings/python/pyiree",
    ],
)
