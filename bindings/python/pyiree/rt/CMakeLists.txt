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
    rt
  SRCS
    "__init__.py"
    # "system_api.py"
  DEPS
    ::binding
)

iree_py_extension(
  NAME
    binding
  SRCS
    "initialize_module.cc"
  DEPS
    iree::hal::vulkan::vulkan_driver_module
    iree::hal::llvmjit::llvmjit_driver_module
    ::rt_library
    bindings::python::pyiree::common
    iree::base::initializer
    iree::base::tracing
    # TODO(marbre): Add dependency
    # "@com_google_tracing_framework_cpp//:tracing_framework_bindings_cpp",
  COPTS
    ${PYBIND_COPTS}
    ${PYBIND_EXTENSION_COPTS}
  TYPE
    SHARED
)

iree_pybind_cc_library(
  NAME
    rt_library
  HDRS
    "function_abi.h"
    "hal.h"
    "host_types.h"
    "vm.h"
  SRCS
    "function_abi.cc"
    "hal.cc"
    "host_types.cc"
    "vm.cc"
  DEPS
    bindings::python::pyiree::common
    iree::base::api
    iree::base::signature_mangle
    iree::hal::api
    iree::modules::hal
    iree::modules::tensorlist::native_module
    iree::vm
    iree::vm::bytecode_module
    iree::vm::invocation
    iree::vm::ref
    iree::vm::variant_list
    absl::inlined_vector
    absl::memory
    absl::strings
    absl::optional
    absl::span
  TYPE
    STATIC
)

iree_py_library(
  NAME
    system_api
  SRCS
    "system_api.py"
  DEPS
    ::binding
)

iree_py_test(
  NAME
    function_abi_test
  SRCS
    "function_abi_test.py"
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
