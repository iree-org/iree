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

"""Macros for building IREE python extensions."""

load("@iree_native_python//:build_defs.bzl", "py_extension")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("//iree:build_defs.oss.bzl", _PLATFORM_VULKAN_DEPS = "PLATFORM_VULKAN_DEPS")

NUMPY_DEPS = []
PLATFORM_VULKAN_DEPS = _PLATFORM_VULKAN_DEPS
PYTHON_HEADERS_DEPS = ["@iree_native_python//:python_headers"]
PYTHON_CPP_EXTRA_DEPS = []

PYBIND_COPTS = [
    "-fexceptions",
]

PYBIND_FEATURES = [
    "-use_header_modules",  # Incompatible with exceptions builds.
]

PYBIND_EXTENSION_COPTS = [
    "-fvisibility=hidden",
]

PYBIND_REGISTER_MLIR_PASSES = [
    "-DIREE_REGISTER_MLIR_PASSES",
]

# Optional deps to enable an intree TensorFlow python. This build configuration
# defaults to getting TensorFlow from the python environment (empty).
INTREE_TENSORFLOW_PY_DEPS = []

def pybind_cc_library(
        name,
        copts = [],
        features = [],
        deps = [],
        **kwargs):
    """Wrapper cc_library for deps that are part of the python bindings."""
    cc_library(
        name = name,
        copts = copts + PYBIND_COPTS,
        features = PYBIND_FEATURES,
        deps = [
            "@iree_pybind11//:pybind11",
        ] + deps + PYTHON_HEADERS_DEPS,
        **kwargs
    )

def iree_py_library(**kwargs):
    """Compatibility py_library which has bazel compatible args."""

    # This is used when args are needed that are incompatible with upstream.
    # Presently, this includes:
    #   imports
    py_library(**kwargs)

def iree_py_binary(**kwargs):
    """Compatibility py_binary which has bazel specific args."""

    # See: https://github.com/google/iree/issues/2405
    py_binary(legacy_create_init = False, **kwargs)

def iree_py_test(**kwargs):
    """Compatibility py_test which has bazel compatible args."""

    # See: https://github.com/google/iree/issues/2405
    py_test(legacy_create_init = False, **kwargs)

def iree_py_extension(**kwargs):
    """Delegates to the real py_extension."""
    py_extension(**kwargs)
