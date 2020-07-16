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
    "iree_py_library",
    "iree_py_test",
    "pybind_cc_library",
)

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

DRIVER_DEPS = PLATFORM_VULKAN_DEPS + [
    "//iree/hal/vulkan:vulkan_driver_module",
    "//iree/hal/llvmjit:llvmjit_driver_module",
    "//iree/hal/vmla:vmla_driver_module",
]

iree_py_library(
    name = "rt",
    srcs = [
        "__init__.py",
        "system_api.py",
    ],
    srcs_version = "PY3",
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
    win_def_file = "export.def",
    deps = DRIVER_DEPS + [
        ":rt_library",
        "//bindings/python/pyiree/common",
        "//iree/base:initializer",
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
    deps = [
        "//bindings/python/pyiree/common",
        "//iree/base:api",
        "//iree/base:localfile",
        "//iree/base:signature_mangle",
        "//iree/hal:api",
        "//iree/modules/hal",
        "//iree/modules/strings:strings_module",
        "//iree/modules/tensorlist:native_module",
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

iree_py_library(
    name = "system_api",
    srcs = ["system_api.py"],
    srcs_version = "PY3",
    deps = [
        ":binding",
        "//bindings/python:pathsetup",  # build_cleaner: keep
    ],
)

iree_py_test(
    name = "function_abi_test",
    srcs = ["function_abi_test.py"],
    python_version = "PY3",
    # TODO(laurenzo): Enable once test does not depend on a real vulkan device.
    tags = [
        "nokokoro",
    ],
    deps = NUMPY_DEPS + [
        "//bindings/python:pathsetup",  # build_cleaner: keep
        "@absl_py//absl/testing:absltest",
        "//bindings/python/pyiree/rt",
    ],
)

iree_py_test(
    name = "hal_test",
    srcs = ["hal_test.py"],
    python_version = "PY3",
    deps = NUMPY_DEPS + [
        "//bindings/python:pathsetup",  # build_cleaner: keep
        "@absl_py//absl/testing:absltest",
        "//bindings/python/pyiree/rt",
    ],
)

iree_py_test(
    name = "system_api_test",
    srcs = ["system_api_test.py"],
    python_version = "PY3",
    deps = NUMPY_DEPS + [
        ":system_api",
        "//bindings/python:pathsetup",  # build_cleaner: keep
        "@absl_py//absl/testing:absltest",
        "//bindings/python/pyiree/compiler",
        "//bindings/python/pyiree/rt",
    ],
)

iree_py_test(
    name = "vm_test",
    srcs = ["vm_test.py"],
    python_version = "PY3",
    deps = NUMPY_DEPS + [
        "//bindings/python:pathsetup",  # build_cleaner: keep
        "@absl_py//absl/testing:absltest",
        "//bindings/python/pyiree/compiler",
        "//bindings/python/pyiree/rt",
    ],
)
