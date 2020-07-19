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

iree_add_pyext_library(IREEBindingsPythonRtLib
  SRCS
    "function_abi.h"
    "hal.h"
    "host_types.h"
    "vm.h"
    "function_abi.cc"
    "hal.cc"
    "host_types.cc"
    "vm.cc"
  PYEXT_DEPS
    IREEBindingsPythonCommonLib
  DEPS
    iree_base_api
    iree_base_signature_mangle
    iree_hal_api
    iree_modules_hal_hal
    iree_modules_strings_strings_module
    iree_modules_tensorlist_native_module
    iree_vm_vm
    iree_vm_bytecode_module
    iree_vm_invocation
    iree_vm_ref
    iree_vm_variant_list
    absl_inlined_vector
    absl_memory
    absl_strings
    absl_optional
    absl_span
)

iree_add_pyext_module(IREEBindingsPythonRt
  MODULE_NAME binding
  SRCS
    "initialize_module.cc"
  PYEXT_DEPS
    IREEBindingsPythonCommonLib
    IREEBindingsPythonRtLib
  DEPS
    iree_hal_vulkan_vulkan_driver_module
    iree_hal_llvmjit_llvmjit_driver_module
    iree_hal_vmla_vmla_driver_module
    iree_base_initializer
)

iree_py_library(
  NAME
    rt
  SRCS
    "__init__.py"
    "system_api.py"
  DEPS
    IREEBindingsPythonRt
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
