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
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//iree:build_defs.oss.bzl", _PLATFORM_VULKAN_DEPS = "PLATFORM_VULKAN_DEPS")

NUMPY_DEPS = []
PLATFORM_VULKAN_DEPS = _PLATFORM_VULKAN_DEPS
PYTHON_HEADERS_DEPS = ["@iree_native_python//:python_headers"]

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
        deps = [
        ],
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

def iree_py_test(**kwargs):
    """Compatibility py_test which has bazel compatible args."""

    # This is used when args are needed that are incompatible with upstream.
    # Presently, this includes:
    #   imports
    py_test(legacy_create_init = False, **kwargs)

def iree_py_test_suite(
        name,
        srcs,
        deps = None,
        tags = None,
        size = None,
        python_version = "PY3",
        **kwargs):
    """Creates one iree_py_test per source file and a test suite that bundles them.

    Args:
      name: name of the generated test suite.
      srcs: test file sources.
      deps: test dependencies.
      tags: tags to apply to the test. Note that as in standard test suites,
            manual is treated specially and will also apply to the test suite
            itself.
      size: size of the tests.
      python_version: the python version to run the tests with. Uses python3
                      by default.
      **kwargs: Any additional arguments that will be passed to the underlying
                tests and test_suite.
    """
    tests = []
    for src in srcs:
        test_name = "{}_{}".format(name, src)
        iree_py_test(
            name = test_name,
            main = src,
            srcs = [src],
            deps = deps,
            tags = tags,
            size = size,
            python_version = python_version,
            **kwargs
        )
        tests.append(test_name)

    native.test_suite(
        name = name,
        tests = tests,
        # Note that only the manual tag really has any effect here. Others are
        # used for test suite filtering, but all tests are passed the same tags.
        tags = tags,
        # If there are kwargs that need to be passed here which only apply to
        # the generated tests and not to test_suite, they should be extracted
        # into separate named arguments.
        **kwargs
    )

def iree_py_extension(**kwargs):
    """Delegates to the real py_extension."""
    py_extension(**kwargs)
