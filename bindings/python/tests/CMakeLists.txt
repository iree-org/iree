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

iree_py_test(
  NAME
    compiler_core_test
  SRCS
    "compiler_core_test.py"
)

iree_py_test(
  NAME
    compiler_tf_test
  SRCS
    "compiler_tf_test.py"
)

iree_py_test(
  NAME
    compiler_tflite_test
  SRCS
    "compiler_tflite_test.py"
)

iree_py_test(
  NAME
    compiler_xla_test
  SRCS
    "compiler_xla_test.py"
)
