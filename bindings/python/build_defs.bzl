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

load("@rules_python//python:defs.bzl", "py_library")

def iree_py_library(**kwargs):
    """Compatibility py_library which has bazel compatible args."""

    # This is used when args are needed that are incompatible with upstream.
    # Presently, this includes:
    #   imports
    py_library(**kwargs)
