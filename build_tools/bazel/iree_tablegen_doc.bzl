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

load("//build_tools/bazel:tblgen.bzl", "gentbl")

def iree_tablegen_doc(*args, **kwargs):
    """iree_tablegen_doc() generates documentation from a table definition file.

    This is a simple wrapper over gentbl() so we can differentiate between
    documentation and others. See gentbl() for details regarding arguments.
    """
    gentbl(*args, **kwargs)
