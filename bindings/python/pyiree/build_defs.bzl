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

"""Macros for building IREE python extensions."""

# Re-export various top-level things.
# We do this to keep the python bindings fairly self-contained and
# able to move to a dedicated repo more easily.

load(
    "//iree:build_defs.bzl",
    "cc_library",
    _NUMPY_DEPS = "NUMPY_DEPS",
    _PLATFORM_VULKAN_DEPS = "PLATFORM_VULKAN_DEPS",
    _PYTHON_HEADER_DEPS = "PYTHON_HEADERS_DEPS",
    _iree_py_extension = "iree_py_extension",
)

NUMPY_DEPS = _NUMPY_DEPS
PLATFORM_VULKAN_DEPS = _PLATFORM_VULKAN_DEPS
PYTHON_HEADER_DEPS = _PYTHON_HEADER_DEPS
iree_py_extension = _iree_py_extension

PYBIND_COPTS = [
    "-fexceptions",
]

PYBIND_FEATURES = [
    "-use_header_modules",  # Incompatible with exceptions builds.
]

PYBIND_EXTENSION_COPTS = [
    "-fvisibility=hidden",
]

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
        copts = PYBIND_COPTS,
        features = PYBIND_FEATURES,
        deps = [
            "@iree_pybind11//:pybind11",
        ] + deps + PYTHON_HEADER_DEPS,
        **kwargs
    )
