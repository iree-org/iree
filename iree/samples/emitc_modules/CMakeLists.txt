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
  iree_cc_test(
  NAME
      add_module_test
  SRCS
      "add_module_test.cc"
  DEPS
      iree::vm::context
      iree::vm::instance
      iree::vm::invocation
      iree::vm::list
      ::add_module_test_hdrs
      iree::vm::ref_cc
      iree::base::api
      iree::base::status
      iree::testing::gtest
      iree::testing::gtest_main
  )
  
  iree_cc_library(
  NAME
      add_module_test_hdrs
  HDRS
      "add_module_test.h"
      "add_mlir_generated.h"
  DEPS
      iree::vm::context
      iree::vm::instance
      iree::vm::native_module
      iree::vm::ref
      iree::vm::stack
      iree::base::api
      iree::compiler::Dialect::VM::Target::C::VMCFuncs
  PUBLIC
  )
endif()