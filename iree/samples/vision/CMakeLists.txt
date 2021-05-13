# Copyright 2021 Google LLC
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

if(NOT ${IREE_TARGET_BACKEND_DYLIB-LLVM-AOT} OR NOT ${IREE_HAL_DRIVER_DYLIB})
  return()
endif()

# Build the bytecode from the mnist.mlir in iree/samples/models.
iree_bytecode_module(
  NAME
    mnist_bytecode_module
  SRC
    "mnist.mlir"
  C_IDENTIFIER
    "mnist_bytecode_module_c"
  FLAGS
    "-iree-mlir-to-vm-bytecode-module"
    "-iree-hal-target-backends=dylib-llvm-aot"
  PUBLIC
)

iree_cc_binary(
  NAME
    iree-run-mnist-module
  OUT
    iree-run-mnist-module
  SRCS
    "iree-run-mnist-module.c"
  DATA
    "iree::samples::vision::mnist_test.png"
  DEPS
    ::mnist_bytecode_module_c
    iree::runtime
    iree::tools::utils::image_util
)
