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

if(${IREE_BUILD_COMPILER})
  iree_bytecode_module(
    NAME
      simple_mul_bytecode_module
    SRC
      "simple_mul.mlir"
    C_OUTPUT
    FLAGS
      "-iree-mlir-to-vm-bytecode-module"
      "-iree-hal-target-backends=vmla"
  )

  iree_cc_binary(
    NAME
      integration_test
     SRCS
      "integration_test.cc"
    DEPS
       experimental::bindings::java::com::google::iree::native::cc_wrappers
       experimental::bindings::java::com::google::iree::tests::simple_mul_bytecode_module_c
       iree::base::status
  )
endif()
