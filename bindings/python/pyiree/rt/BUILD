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
    "PLATFORM_VULKAN_DEPS",
    "PYBIND_COPTS",
    "PYBIND_EXTENSION_COPTS",
    "PYBIND_FEATURES",
    "iree_py_extension",
    "pybind_cc_library",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

DRIVER_DEPS = PLATFORM_VULKAN_DEPS + [
    "//iree/hal/interpreter:interpreter_driver_module",
    "//iree/hal/vulkan:vulkan_driver_module",
]

py_library(
    name = "rt",
    srcs = [
        "__init__.py",
        "system_api.py",
    ],
    srcs_version = "PY3",
    tags = ["nokokoro"],
    deps = [
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
    # TODO(b/145815906) Get this running in OSS CI.
    tags = ["nokokoro"],
    win_def_file = "export.def",
    deps = DRIVER_DEPS + [
        ":rt_library",
        "//bindings/python/pyiree/common",
        "//iree/base:initializer",
        "//iree/base:tracing",
        "@com_google_tracing_framework_cpp//:tracing_framework_bindings_cpp",
    ],
)

pybind_cc_library(
    name = "rt_library",
    srcs = [
        "function_abi.cc",
        "hal.cc",
        "host_types.cc",
        "vm.cc",
    ],
    hdrs = [
        "function_abi.h",
        "hal.h",
        "host_types.h",
        "vm.h",
    ],
    tags = ["nokokoro"],
    deps = [
        "//bindings/python/pyiree/common",
        "//iree/base:api",
        "//iree/base:signature_mangle",
        "//iree/hal:api",
        "//iree/modules/hal",
        "//iree/vm",
        "//iree/vm:bytecode_module",
        "//iree/vm:invocation",
        "//iree/vm:module",
        "//iree/vm:ref",
        "//iree/vm:variant_list",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

py_library(
    name = "system_api",
    srcs = ["system_api.py"],
    srcs_version = "PY3",
    # TODO(b/145815906) Get this running in OSS CI.
    tags = ["nokokoro"],
    deps = [
        ":binding",
        "//bindings/python:pathsetup",  # build_cleaner: keep
    ],
)

py_test(
    name = "function_abi_test",
    srcs = ["function_abi_test.py"],
    python_version = "PY3",
    # TODO(laurenzo): Enable once test does not depend on a real vulkan device.
    tags = [
        "noga",
        "nokokoro",
        "notap",
    ],
    deps = NUMPY_DEPS + [
        "//bindings/python:pathsetup",  # build_cleaner: keep
        "@absl_py//absl/testing:absltest",
        "//bindings/python/pyiree/rt",
    ],
    # TODO(b/145815906) Get this running in OSS CI.
)

py_test(
    name = "hal_test",
    srcs = ["hal_test.py"],
    python_version = "PY3",
    # TODO(b/145815906) Get this running in OSS CI.
    tags = ["nokokoro"],
    deps = NUMPY_DEPS + [
        "//bindings/python:pathsetup",  # build_cleaner: keep
        "@absl_py//absl/testing:absltest",
        "//bindings/python/pyiree/rt",
    ],
)

py_test(
    name = "system_api_test",
    srcs = ["system_api_test.py"],
    python_version = "PY3",
    tags = [
        # TODO(laurenzo): Enable once test does not depend on a real vulkan device.
        "notap",
        # TODO(b/145815906) Get this running in OSS CI.
        "noga",
        "nokokoro",
    ],
    deps = NUMPY_DEPS + [
        ":system_api",
        "//bindings/python:pathsetup",  # build_cleaner: keep
        "@absl_py//absl/testing:absltest",
        "//bindings/python/pyiree/compiler",
        "//bindings/python/pyiree/rt",
    ],
)

py_test(
    name = "vm_test",
    srcs = ["vm_test.py"],
    python_version = "PY3",
    # TODO(laurenzo): Enable once test does not depend on a real vulkan device.
    tags = [
        "noga",
        "nokokoro",
        "notap",
    ],
    deps = NUMPY_DEPS + [
        "//bindings/python:pathsetup",  # build_cleaner: keep
        "@absl_py//absl/testing:absltest",
        "//bindings/python/pyiree/compiler",
        "//bindings/python/pyiree/rt",
    ],
)

# py_library(
#     name = "pyiree",
#     srcs = [
#         "__init__.py",
#     ],
#     srcs_version = "PY3",
#     # TODO(b/145815906) Get this running in OSS CI.
#     tags = ["nokokoro"],
#     deps = [
#         ":binding",
#         ":compiler",
#         ":system_api",
#         "//bindings/python:pathsetup",  # build_cleaner: keep
#     ] + select({
#         "//iree:enable_tensorflow": [
#             "//bindings/python/pyiree/tf_interop:test_utils",
#             "//bindings/python/pyiree/tf_interop:tf_test_driver",
#         ],
#         "//conditions:default": [
#         ],
#     }),
# )
#
#
# cc_library(
#     name = "base",
#     srcs = [
#         "compiler.cc",
#         "function_abi.cc",
#         "hal.cc",
#         "host_types.cc",
#         "status_utils.cc",
#         "vm.cc",
#     ],
#     hdrs = [
#         "binding.h",
#         "compiler.h",
#         "function_abi.h",
#         "hal.h",
#         "host_types.h",
#         "status_utils.h",
#         "vm.h",
#     ],
#     copts = DEFAULT_COPTS,
#     features = DEFAULT_FEATURES,
#     # TODO(b/145815906) Get this running in OSS CI.
#     tags = ["nokokoro"],
#     deps = [
#         "@com_google_absl//absl/container:inlined_vector",
#         "@com_google_absl//absl/memory",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/types:optional",
#         "@com_google_absl//absl/types:span",
#         "@com_google_absl//absl/types:variant",
#         "//iree/compiler/Utils",
#         "//iree/modules/hal",
#         "//iree/vm",
#         "//iree/vm:bytecode_module",
#         "//iree/vm:invocation",
#         "//iree/vm:module",
#         "//iree/vm:ref",
#         "//iree/vm:variant_list",
#         "//third_party/llvm/llvm/projects/google_mlir:IR",
#         "//iree/base:api",
#         "//iree/base:status",
#         "//iree/base:signature_mangle",
#         "//iree/hal:api",
#         "//iree/vm:api",
#         "@llvm-project//llvm:support",
#         "//third_party/llvm/llvm/projects/google_mlir:Parser",
#         "//third_party/llvm/llvm/projects/google_mlir:Pass",
#         "@iree_pybind11//:pybind11",
#     ] + COMPILER_DEPS + DRIVER_DEPS + PYTHON_HEADERS_DEPS,
# )
#
# iree_py_extension(
#     name = "binding",
#     srcs = [
#         "initialize_module.cc",
#     ],
#     copts = DEFAULT_COPTS,
#     features = DEFAULT_FEATURES,
#     linkstatic = 1,
#     # TODO(b/145815906) Get this running in OSS CI.
#     tags = ["nokokoro"],
#     win_def_file = "export.def",
#     deps = [
#         ":base",
#         "//bindings/python/pyiree/tf_interop",
#         "//iree/base:initializer",
#         "//iree/base:tracing",
#         "@com_google_tracing_framework_cpp//:tracing_framework_bindings_cpp",
#     ],
# )
#
# py_test(
#     name = "compiler_test",
#     srcs = ["compiler_test.py"],
#     python_version = "PY3",
#     # TODO(b/145815906) Get this running in OSS CI.
#     tags = ["nokokoro"],
#     deps = [
#         "//bindings/python:pathsetup",  # build_cleaner: keep
#         "@absl_py//absl/testing:absltest",
#         "//bindings/python/pyiree",
#     ],
# )
#
