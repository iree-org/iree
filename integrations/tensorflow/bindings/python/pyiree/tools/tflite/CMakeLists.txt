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
    tflite
  SRCS
    "__init__.py"
  DEPS
    integrations_iree_tensorflow_importers
)

iree_symlink_tool(
  TARGET tflite
  FROM_TOOL_TARGET iree_tf_compiler_iree-import-tflite
  TO_EXE_NAME iree-import-tflite
)

iree_py_install_package(
  COMPONENT IreePythonPackage-tools-tflite
  PACKAGE_NAME iree_tools_tflite
  MODULE_PATH pyiree/tools/tflite
  DEPS
    integrations_iree_tensorflow_importers
)

# Since imported, need to resolve the TARGET_FILE ourselves instead of
# install TARGETS form.
install(
  PROGRAMS "$<TARGET_FILE:iree_tf_compiler_iree-import-tflite>"
  DESTINATION "${PY_INSTALL_MODULE_DIR}"
  COMPONENT "${PY_INSTALL_COMPONENT}"
)
