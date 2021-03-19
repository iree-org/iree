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
    core
  SRCS
    "__init__.py"
)

if(${IREE_BUILD_COMPILER})
  iree_symlink_tool(
    TARGET core
    FROM_TOOL_TARGET iree_tools_iree-translate
    TO_EXE_NAME iree-translate
  )
endif()

iree_py_install_package(
  AUGMENT_EXISTING_PACKAGE
  COMPONENT IreePythonPackage-compiler
  PACKAGE_NAME iree_compiler
  MODULE_PATH iree/tools/core
  DEPS
    iree_tools_iree-translate
)

install(
  TARGETS iree_tools_iree-translate
  COMPONENT ${PY_INSTALL_COMPONENT}
  DESTINATION "${PY_INSTALL_MODULE_DIR}"
)
