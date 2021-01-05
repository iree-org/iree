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
    compiler2
  SRCS
    "__init__.py"
    "core.py"
    "tf.py"
    "tflite.py"
    "tools.py"
    "xla.py"
)

# Installation.
configure_file(setup.py.in setup.py)
configure_file(version.py.in version.py)

set(_install_component IreePythonCompiler)
set(_install_package_dir "${CMAKE_INSTALL_PREFIX}/python_packages/iree_compiler")
set(_install_destination "${_install_package_dir}/pyiree/compiler2")

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py
  COMPONENT ${_install_component}
  DESTINATION "${_install_package_dir}"
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/version.py
  COMPONENT ${_install_component}
  DESTINATION "${_install_destination}"
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  COMPONENT ${_install_component}
  DESTINATION "${_install_destination}"
  FILES_MATCHING
    PATTERN "*.py"
)
