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
  PYEXT_DEPS
    ::PyExtCompiler
)

iree_pyext_module(
  NAME
    PyExtCompiler
  MODULE_NAME
    binding
  UNIX_LINKER_SCRIPT
    "unix_version.lds"
  SRCS
    "initialize_module.cc"
  PYEXT_DEPS
    ::PyExtCompilerLib
    bindings::python::pyiree::common::PyextCommonLib
)

iree_pyext_library(
  NAME
    PyExtCompilerLib
  SRCS
    "compiler.h"
    "compiler.cc"
  PYEXT_DEPS
    bindings::python::pyiree::common::PyextCommonLib
  COPTS
    ${PYBIND_REGISTER_MLIR_PASSES}
  DEPS
    # Transforms. Adopted from the Bazel variable COMPILER::DEPS.
    iree::compiler::Dialect::Flow::Transforms
    iree::compiler::Dialect::HAL::Transforms
    iree::compiler::Dialect::HAL::Target
    iree::compiler::Dialect::Shape::IR
    iree::compiler::Dialect::Shape::Transforms
    iree::compiler::Dialect::VM::Transforms
    # Targets. Adopted from the Bazel variable COMPILER::DEPS.
    iree::compiler::Dialect::HAL::Target::VMLA
    iree::compiler::Dialect::HAL::Target::LLVM::LLVMAOT
    iree::compiler::Dialect::HAL::Target::LLVM::LLVMIR
    iree::compiler::Dialect::HAL::Target::VulkanSPIRV
    iree::compiler::Dialect::VM::Target::Bytecode
    iree::compiler::Dialect::VM::Target::init_targets
    iree::tools::init_compiler_modules
    iree::tools::init_iree_passes_and_dialects
    iree::tools::init_mlir_passes_and_dialects
    iree::tools::init_targets
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
