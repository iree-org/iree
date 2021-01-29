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
    tf
  SRCS
    "__init__.py"
  DEPS
    integrations_iree_tensorflow_importers
)

iree_symlink_tool(
  TARGET tf
  FROM_TOOL_TARGET iree_tf_compiler_iree-tf-import
  TO_EXE_NAME iree-tf-import
)

iree_py_install_package(
  COMPONENT IreePythonPackage-tools-tf
  PACKAGE_NAME iree_tools_tf
  MODULE_PATH pyiree/tools/tf
  DEPS
    integrations_iree_tensorflow_importers
)

# Since imported, need to resolve the TARGET_FILE ourselves instead of
# install TARGETS form.
install(
  PROGRAMS "$<TARGET_FILE:iree_tf_compiler_iree-tf-import>"
  DESTINATION "${PY_INSTALL_MODULE_DIR}"
  COMPONENT "${PY_INSTALL_COMPONENT}"
)
