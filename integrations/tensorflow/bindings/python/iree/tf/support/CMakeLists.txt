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

iree_py_library(
  NAME
    support
  SRCS
    "__init__.py"
    "module_utils.py"
    "tf_test_driver.py"
    "tf_test_utils.py"
    "tf_utils.py"
    "trace_utils.py"
)

# TODO(#5147): Replace the hack below with something more thought out.
# Make the base dir that run_test.sh expects to exist. This dir doesn't exist
# by default because we symlink these python files into bindings/python/iree.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/integrations/tensorflow/bindings/python/iree/tf/support")

iree_py_test(
  NAME
    module_utils_test
  SRCS
    "module_utils_test.py"
)

iree_py_test(
  NAME
    tf_test_utils_test
  SRCS
    "tf_test_utils_test.py"
)

iree_py_test(
  NAME
    tf_utils_test
  SRCS
    "tf_utils_test.py"
)

iree_py_test(
  NAME
    trace_utils_test
  SRCS
    "trace_utils_test.py"
)

iree_py_install_package(
  AUGMENT_EXISTING_PACKAGE
  COMPONENT IreePythonPackage-tools-tf
  PACKAGE_NAME iree_tools_tf
  MODULE_PATH iree/tf/support
)
