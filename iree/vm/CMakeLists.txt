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
    vm
  HDRS
    "api.h"
  DEPS
    ::impl
    iree::base::api
  PUBLIC
)

iree_cc_library(
  NAME
    cc
  HDRS
    "module_abi_packing.h"
    "native_module_cc.h"
    "ref_cc.h"
  DEPS
    ::vm
    absl::core_headers
    absl::inlined_vector
    absl::optional
    absl::span
    absl::strings
    iree::base::api
    iree::base::status
  PUBLIC
)

iree_cc_library(
  NAME
    impl
  HDRS
    "builtin_types.h"
    "context.h"
    "instance.h"
    "invocation.h"
    "list.h"
    "module.h"
    "native_module.h"
    "ref.h"
    "stack.h"
    "type_def.h"
    "value.h"
  SRCS
    "builtin_types.c"
    "context.c"
    "instance.c"
    "invocation.c"
    "list.c"
    "module.c"
    "native_module.c"
    "ref.c"
    "stack.c"
  DEPS
    iree::base::api
    iree::base::core_headers
    iree::base::internal
    iree::base::tracing
  PUBLIC
)

iree_cc_test(
  NAME
    list_test
  SRCS
    "list_test.cc"
  DEPS
    ::cc
    ::impl
    iree::base::api
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    native_module_test
  SRCS
    "native_module_test.cc"
  DEPS
    ::cc
    ::impl
    ::native_module_test_hdrs
    iree::base::api
    iree::base::status
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    native_module_test_hdrs
  HDRS
    "native_module_test.h"
  DEPS
    ::impl
    iree::base::api
  PUBLIC
)

iree_cc_test(
  NAME
    native_module_benchmark
  SRCS
    "native_module_benchmark.cc"
  DEPS
    ::impl
    ::native_module_test_hdrs
    benchmark
    iree::base::api
    iree::base::logging
    iree::testing::benchmark_main
)

iree_cc_test(
  NAME
    ref_test
  SRCS
    "ref_test.cc"
  DEPS
    ::cc
    ::impl
    iree::base::api
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    stack_test
  SRCS
    "stack_test.cc"
  DEPS
    ::impl
    iree::base::api
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    bytecode_module
  HDRS
    "bytecode_module.h"
  SRCS
    "bytecode_dispatch.c"
    "bytecode_dispatch_util.h"
    "bytecode_module.c"
    "bytecode_module_impl.h"
    "generated/bytecode_op_table.h"
  DEPS
    ::ops
    ::vm
    iree::base::api
    iree::base::core_headers
    iree::base::flatcc
    iree::base::internal
    iree::base::tracing
    iree::schemas::bytecode_module_def_c_fbs
  PUBLIC
)

if(${IREE_BUILD_COMPILER})

iree_cc_test(
  NAME
    bytecode_module_test
  SRCS
    "bytecode_dispatch_test.cc"
    "bytecode_module_test.cc"
  DEPS
    ::bytecode_module
    ::vm
    absl::strings
    iree::base::logging
    iree::base::status
    iree::testing::gtest
    iree::testing::gtest_main
    iree::vm::test::all_bytecode_modules_cc
)

iree_cc_binary(
  NAME
    bytecode_module_benchmark
  SRCS
    "bytecode_module_benchmark.cc"
  DEPS
    ::bytecode_module
    ::bytecode_module_benchmark_module_cc
    ::vm
    absl::inlined_vector
    absl::strings
    benchmark
    iree::base::api
    iree::base::logging
    iree::testing::benchmark_main
  TESTONLY
)

iree_run_binary_test(
  NAME
    "bytecode_module_benchmark_test"
  ARGS
    "--benchmark_min_time=0"
  TEST_BINARY
    ::bytecode_module_benchmark
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
  TESTONLY
  PUBLIC
)

iree_cc_test(
  NAME
    bytecode_module_size_benchmark
  SRCS
    "bytecode_module_size_benchmark.cc"
  DEPS
    ::bytecode_module
    ::bytecode_module_size_benchmark_module_cc
    ::vm
    iree::base::api
)

iree_bytecode_module(
  NAME
    bytecode_module_size_benchmark_module
  SRC
    "bytecode_module_size_benchmark.mlir"
  CC_NAMESPACE
    "iree::vm"
  FLAGS
    "-iree-vm-ir-to-bytecode-module"
  TESTONLY
  PUBLIC
)

endif()

iree_cc_library(
  NAME
    ops
  HDRS
    "ops.h"
  PUBLIC
)

iree_cc_library(
  NAME
    shims
  HDRS
    "shims.h"
  PUBLIC
)
