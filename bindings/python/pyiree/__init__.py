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
# pylint: disable=g-bad-import-order

# Top-level modules that are imported verbatim.
from . import binding
from . import compiler
from .binding import tracing

# Alias public compiler symbols.
# TODO(laurenzo): Remove these aliases.
from .binding.compiler import CompilerContext
from .binding.compiler import CompilerModule

# Alias specific native functions.
from .binding.vm import create_module_from_blob

### Load non-native py_library deps here ###
### Order matters because these typically have a back-reference on this
### module (being constructed). Issues should fail-fast and be easy to see.

# Alias the tf_test_driver if available (optional - only if testing is linked
# in).
try:
  from .tf_interop import tf_test_driver
  from .tf_interop import test_utils as tf_test_utils
except ImportError:
  pass
