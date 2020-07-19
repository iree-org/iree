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
    compiler
  SRCS
    "__init__.py"
  DEPS
    IREEBindingsPythonCompiler
)

iree_add_pyext_module(IREEBindingsPythonCompiler
  MODULE_NAME
    binding
  SRCS
    "initialize_module.cc"
  PYEXT_DEPS
    IREEBindingsPythonCommonLib
    IREEBindingsPythonCompilerLib
)

iree_add_pyext_library(IREEBindingsPythonCompilerLib
  SRCS
    "compiler.h"
    "compiler.cc"
  PYEXT_DEPS
    IREEBindingsPythonCommonLib
  COPTS
    ${PYBIND_REGISTER_MLIR_PASSES}
  DEPS
    # Transforms. Adopted from the Bazel variable COMPILER_DEPS.
    iree_compiler_Dialect_Flow_Transforms_Transforms
    iree_compiler_Dialect_HAL_Transforms_Transforms
    iree_compiler_Dialect_HAL_Target_Target
    iree_compiler_Dialect_Shape_IR_IR
    iree_compiler_Dialect_Shape_Transforms_Transforms
    iree_compiler_Dialect_VM_Transforms_Transforms
    # Targets. Adopted from the Bazel variable COMPILER_DEPS.
    iree_compiler_Dialect_HAL_Target_VMLA_VMLA
    iree_compiler_Dialect_HAL_Target_LLVM_LLVMAOT
    iree_compiler_Dialect_HAL_Target_LLVM_LLVMIR
    iree_compiler_Dialect_HAL_Target_VulkanSPIRV_VulkanSPIRV
    iree_compiler_Dialect_VM_Target_Bytecode_Bytecode
    iree_compiler_Dialect_VM_Target_init_targets
    iree_tools_init_compiler_modules
    iree_tools_init_iree_passes_and_dialects
    iree_tools_init_mlir_passes_and_dialects
    iree_tools_init_targets
    LLVMSupport
    MLIRIR
    MLIRSCFTransforms
    MLIRParser
    MLIRPass
)

iree_py_test(
  NAME
    compiler_test
  SRCS
    "compiler_test.py"
)
