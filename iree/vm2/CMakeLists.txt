# Copyright 2019 Google LLC
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

iree_bytecode_module(
  NAME
    bytecode_dispatch_test_module
  SRC
    "bytecode_dispatch_test.mlir"
  CC_NAMESPACE
    "iree::vm"
  TRANSLATION
    "-iree-vm-ir-to-bytecode-module"
)

iree_cc_library(
  NAME
    bytecode_module
  HDRS
    "bytecode_module.h"
  SRCS
    "bytecode_dispatch.c"
    "bytecode_module.cc"
    "bytecode_module_impl.h"
    "bytecode_op_table.h"
  DEPS
    iree::vm2::bytecode_op_table_gen
    iree::vm2::module
    iree::vm2::ref
    iree::vm2::stack
    iree::vm2::types
    iree::vm2::value
    iree::base::api
    iree::base::flatbuffer_util
    iree::base::target_platform
    iree::schemas::bytecode_module_def_cc_fbs
    flatbuffers
  PUBLIC
)

#TODO(marbre): 'benchmark/benchmark.h' file not found by bytecode_module_benchmark.cc
#
#iree_cc_test(
#  NAME
#    bytecode_module_benchmark
#  SRCS
#    "bytecode_module_benchmark.cc"
#  DEPS
#    iree::vm2::bytecode_module
#    iree::vm2::bytecode_module_benchmark_module_cc
#    iree::vm2::module
#    iree::vm2::stack
#    iree::base::api
#    iree::base::logging
#    #TODO(marbre): Add dep iree::testing::benchmark_main
#    absl::container
#    absl::strings
#    #TODO(marbre): Add depp "@com_google_benchmark//:benchmark"
#)

iree_bytecode_module(
  NAME
    bytecode_module_benchmark_module
  SRC
    "bytecode_module_benchmark_module.mlir"
  CC_NAMESPACE
    "iree::vm"
  TRANSLATION
    "-iree-vm-ir-to-bytecode-module"
)

iree_cc_test(
  NAME
    bytecode_module_test
  SRCS
    "bytecode_module_test.cc"
  DEPS
    iree::vm2::bytecode_module
    gtest_main
)

iree_tablegen_library(
  NAME
    bytecode_op_table_gen
  SRCS
    "${IREE_ROOT_DIR}/iree/compiler/Dialect/VM/IR/VMOps.td"
  OUTS
    -gen-iree-vm-op-table-defs bytecode_op_table.h
  TBLGEN
    IREE
 )

iree_cc_library(
  NAME
    context
  HDRS
    "context.h"
  SRCS
    "context.c"
  DEPS
    iree::vm2::instance
    iree::vm2::module
    iree::vm2::stack
    iree::base::api
)

iree_cc_library(
  NAME
    instance
  HDRS
    "instance.h"
  SRCS
    "instance.c"
  DEPS
    iree::vm2::types
    iree::base::api
)

iree_cc_library(
  NAME
    invocation
  HDRS
    "invocation.h"
  SRCS
    "invocation.c"
  DEPS
    iree::vm2::context
    iree::vm2::module
    iree::vm2::variant_list
    iree::base::api
)

iree_cc_library(
  NAME
    module
  HDRS
    "module.h"
  SRCS
    "module.c"
  DEPS
    iree::base::api
)

iree_cc_library(
  NAME
    ref
  HDRS
    "ref.h"
  SRCS
    "ref.c"
  DEPS
    iree::base::api
)

iree_cc_test(
  NAME
    ref_test
  SRCS
    "ref_test.cc"
  DEPS
    iree::vm2::ref
    iree::base::api
    iree::base::ref_ptr
    gtest_main
)

iree_cc_library(
  NAME
    stack
  HDRS
    "stack.h"
  SRCS
    "stack.c"
  DEPS
    iree::vm2::module
    iree::vm2::ref
    iree::base::api
)

iree_cc_test(
  NAME
    stack_test
  SRCS
    "stack_test.cc"
  DEPS
    iree::vm2::ref
    iree::vm2::stack
    iree::base::api
    iree::base::ref_ptr
    gtest_main
)

iree_cc_library(
  NAME
    types
  HDRS
    "types.h"
  SRCS
    "types.c"
  DEPS
    iree::vm2::ref
    iree::base::api
)

iree_cc_library(
  NAME
    value
  HDRS
    "value.h"
)

iree_cc_library(
  NAME
    variant_list
  HDRS
    "variant_list.h"
  SRCS
    "variant_list.c"
  DEPS
    iree::vm2::ref
    iree::vm2::value
)

iree_cc_library(
  NAME
    vm2
  HDRS
    "api.h"
  DEPS
    iree::vm2::context
    iree::vm2::instance
    iree::vm2::invocation
    iree::vm2::module
    iree::vm2::ref
    iree::vm2::stack
    iree::vm2::types
    iree::vm2::value
    iree::vm2::variant_list
    iree::base::api
)
