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

iree_add_all_subdirs()

iree_cc_library(
  NAME
    builtin_types
  HDRS
    "builtin_types.h"
  SRCS
    "builtin_types.c"
  DEPS
    ::list
    ::ref
    iree::base::api
  PUBLIC
)

iree_cc_test(
  NAME
    bytecode_dispatch_test
  SRCS
    "bytecode_dispatch_test.cc"
  DEPS
    ::builtin_types
    ::bytecode_module
    ::context
    ::instance
    ::invocation
    ::module
    absl::strings
    iree::base::logging
    iree::testing::gtest_main
    iree::vm::test::all_bytecode_modules_cc
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
    ::builtin_types
    ::list
    ::module
    ::ref
    ::stack
    ::type_def
    ::value
    absl::strings
    flatbuffers
    iree::base::alignment
    iree::base::api
    iree::base::flatbuffer_util
    iree::base::target_platform
    iree::schemas::bytecode_module_def_cc_fbs
  PUBLIC
)

iree_cc_test(
  NAME
    bytecode_module_benchmark
  SRCS
    "bytecode_module_benchmark.cc"
  DEPS
    ::bytecode_module
    ::bytecode_module_benchmark_module_cc
    ::module
    ::stack
    absl::inlined_vector
    absl::strings
    benchmark
    iree::base::api
    iree::base::logging
    iree::testing::benchmark_main
)

iree_bytecode_module(
  NAME
    bytecode_module_benchmark_module
  SRC
    "bytecode_module_benchmark.mlir"
  CC_NAMESPACE
    "iree::vm"
  FLAGS
    "-iree-vm-ir-to-bytecode-module"
  PUBLIC
)

iree_cc_test(
  NAME
    bytecode_module_test
  SRCS
    "bytecode_module_test.cc"
  DEPS
    ::bytecode_module
    iree::testing::gtest_main
)

iree_tablegen_library(
  NAME
    bytecode_op_table_gen
  TD_FILE
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
    ::instance
    ::module
    ::stack
    iree::base::api
    iree::base::atomics
    iree::base::tracing
  PUBLIC
)

iree_cc_library(
  NAME
    instance
  HDRS
    "instance.h"
  SRCS
    "instance.c"
  DEPS
    ::builtin_types
    iree::base::api
    iree::base::atomics
  PUBLIC
)

iree_cc_library(
  NAME
    invocation
  HDRS
    "invocation.h"
  SRCS
    "invocation.c"
  DEPS
    ::context
    ::module
    ::variant_list
    iree::base::api
    iree::base::tracing
  PUBLIC
)

iree_cc_library(
  NAME
    list
  HDRS
    "list.h"
  SRCS
    "list.c"
  DEPS
    ::ref
    ::type_def
    ::value
    iree::base::alignment
    iree::base::api
  PUBLIC
)

iree_cc_test(
  NAME
    list_test
  SRCS
    "list_test.cc"
  DEPS
    ::builtin_types
    ::list
    iree::base::api
    iree::base::ref_ptr
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    module
  HDRS
    "module.h"
  SRCS
    "module.c"
  DEPS
    iree::base::alignment
    iree::base::api
    iree::base::atomics
  PUBLIC
)

iree_cc_library(
  NAME
    module_abi_cc
  HDRS
    "module_abi_cc.h"
    "module_abi_packing.h"
  DEPS
    ::builtin_types
    ::module
    ::ref
    ::ref_cc
    ::stack
    absl::span
    absl::strings
    iree::base::api
    iree::base::api_util
    iree::base::ref_ptr
    iree::base::status
  PUBLIC
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
    iree::base::atomics
  PUBLIC
)

iree_cc_test(
  NAME
    ref_test
  SRCS
    "ref_test.cc"
  DEPS
    ::ref
    iree::base::api
    iree::base::ref_ptr
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    ref_cc
  HDRS
    "ref_cc.h"
  DEPS
    ::ref
    absl::core_headers
    iree::base::api
  PUBLIC
)

iree_cc_library(
  NAME
    stack
  HDRS
    "stack.h"
  SRCS
    "stack.c"
  DEPS
    ::module
    ::ref
    ::variant_list
    iree::base::alignment
    iree::base::api
    iree::base::tracing
  PUBLIC
)

iree_cc_test(
  NAME
    stack_test
  SRCS
    "stack_test.cc"
  DEPS
    ::ref
    ::stack
    iree::base::api
    iree::base::ref_ptr
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    type_def
  HDRS
    "type_def.h"
  DEPS
    ::ref
    ::value
  PUBLIC
)

iree_cc_library(
  NAME
    value
  HDRS
    "value.h"
  PUBLIC
)

iree_cc_library(
  NAME
    variant_list
  HDRS
    "variant_list.h"
  SRCS
    "variant_list.c"
  DEPS
    ::ref
    ::value
  PUBLIC
)

iree_cc_library(
  NAME
    vm
  HDRS
    "api.h"
  DEPS
    ::builtin_types
    ::context
    ::instance
    ::invocation
    ::list
    ::module
    ::ref
    ::stack
    ::type_def
    ::value
    ::variant_list
    iree::base::api
  PUBLIC
)
