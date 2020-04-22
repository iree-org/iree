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
    ::binding
)

iree_py_extension(
  NAME
    binding
  SRCS
    "initialize_module.cc"
  DEPS
    ::compiler_library
    bindings::python::pyiree::common
  COPTS
    ${PYBIND_COPTS}
    ${PYBIND_EXTENSION_COPTS}
  TYPE
    SHARED
)

iree_pybind_cc_library(
  NAME
    compiler_library
  HDRS
    "compiler.h"
  SRCS
    "compiler.cc"
  DEPS
    # Transforms. Adopted from the Bazel variable COMPILER_DEPS.
    iree::compiler::Dialect::Flow::Transforms
    iree::compiler::Dialect::HAL::Transforms
    iree::compiler::Dialect::HAL::Target
    iree::compiler::Dialect::Shape::IR
    iree::compiler::Dialect::Shape::Transforms
    iree::compiler::Dialect::VM::Transforms
    # Targets. Adopted from the Bazel variable COMPILER_DEPS.
    iree::compiler::Dialect::HAL::Target::VMLA
    iree::compiler::Dialect::HAL::Target::LLVM
    iree::compiler::Dialect::HAL::Target::VulkanSPIRV
    iree::compiler::Dialect::VM::Target::Bytecode
    iree::tools::init_compiler_modules
    iree::tools::init_passes_and_dialects
    iree::tools::init_targets
    bindings::python::pyiree::common
    LLVMSupport
    MLIRIR
    MLIRLoopOpsTransforms
    MLIRParser
    MLIRPass
  COPTS
    ${PYBIND_REGISTER_MLIR_PASSES}
  TYPE
    STATIC
)

iree_py_test(
  NAME
    compiler_test
  SRCS
    "compiler_test.py"
)
