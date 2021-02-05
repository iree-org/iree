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

if(${IREE_ENABLE_EMITC})
  # TODO(simon-camp): We are only interested in the output of iree-translate, i.e. add_module.vmfb
  # TODO(marbre): Use a custom command or rather implement a new CMake function.
  iree_bytecode_module(
    NAME
      add_module
    SRC
      "add.mlir"
    CC_NAMESPACE
      "iree::samples::custom_modules"
    FLAGS
      "-iree-vm-ir-to-c-module"
    PUBLIC
  )

  iree_cc_test(
    NAME
      add_module_test
    SRCS
      "add_module_test.cc"
    DEPS
      ::add_module_cc
      iree::base::api
      iree::base::status
      iree::testing::gtest
      iree::testing::gtest_main
      iree::vm
      iree::vm::cc
      iree::vm::ops
      iree::vm::shims
  )
endif()
