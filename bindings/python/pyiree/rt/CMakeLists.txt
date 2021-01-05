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

iree_pyext_module(
  NAME
    PyExtRt
  MODULE_NAME binding
  SRCS
    "initialize_module.cc"
    "binding.h"
    "function_abi.h"
    "hal.h"
    "host_types.h"
    "vm.h"
    "function_abi.cc"
    "hal.cc"
    "host_types.cc"
    "status_utils.cc"
    "status_utils.h"
    "vm.cc"
  UNIX_LINKER_SCRIPT
    "unix_version.lds"
  INCLUDES
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
  DEPS
    iree::base::api
    iree::base::signature_mangle
    iree::hal::api
    iree::hal::drivers
    iree::modules::hal
    iree::modules::strings::strings_module
    iree::modules::tensorlist::native_module
    iree::vm
    iree::vm::bytecode_module
    absl::inlined_vector
    absl::memory
    absl::strings
    absl::optional
    absl::span
)

iree_py_library(
  NAME
    rt
  SRCS
    "__init__.py"
    "system_api.py"
  PYEXT_DEPS
    ::PyExtRt
)

iree_py_test(
  NAME
    function_abi_test
  SRCS
    "function_abi_test.py"
  LABELS
    "nokokoro"
)

iree_py_test(
  NAME
    hal_test
  SRCS
    "hal_test.py"
)

iree_py_test(
  NAME
    system_api_test
  SRCS
    "system_api_test.py"
)

iree_py_test(
  NAME
    vm_test
  SRCS
    "vm_test.py"
)


# Installation.
configure_file(setup.py.in setup.py)

set(_install_component IreePythonRt)
set(_install_package_dir "${CMAKE_INSTALL_PREFIX}/python_packages/iree_rt")
set(_install_destination "${_install_package_dir}/pyiree/rt")

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/setup.py
  COMPONENT ${_install_component}
  DESTINATION "${_install_package_dir}"
)

install(
  TARGETS bindings_python_pyiree_rt_PyExtRt
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
