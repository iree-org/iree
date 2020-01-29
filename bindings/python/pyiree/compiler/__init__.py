"""Module init for the python bindings."""

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

# pylint: disable=g-multiple-import
# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
# pylint: disable=wildcard-import

from . import binding as binding

# Native aliases.
Context = binding.CompilerContext
Module = binding.CompilerModule
CompileOptions = binding.CompileOptions
OutputFormat = binding.OutputFormat

# Conditionally import TensorFlow interop aliases.
HAS_TENSORFLOW = hasattr(binding, "tf_interop")
if HAS_TENSORFLOW:
  from .conditional_tensorflow import *
