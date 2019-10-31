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

# pylint: disable=invalid-name
# pylint: disable=g-import-not-at-top

# Always make the low-level native bindings accessible.
from . import binding

# Alias public compiler symbols.
from .binding.compiler import CompilerContext
from .binding.compiler import CompilerModule

# Alias symbols from the native tf_interop module.
if hasattr(binding, "tf_interop"):
  from .binding.tf_interop import load_saved_model as tf_load_saved_model

### Load non-native py_library deps here ###
### Order matters because these typically have a back-reference on this
### module (being constructed). Issues should fail-fast and be easy to see.

# Alias the tf_test_driver if available (optional - only if testing is linked
# in).
try:
  from .tf_interop import tf_test_driver
except ImportError:
  pass
