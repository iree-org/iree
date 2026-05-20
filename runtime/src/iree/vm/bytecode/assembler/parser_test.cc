// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/assembler/parser.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class VMBytecodeAssemblerParserTest : public testing::Test {
 protected:
  void SetUp() override {
    iree_vm_bytecode_assembler_module_initialize(iree_allocator_system(),
                                                 &module_);
  }

  void TearDown() override {
    iree_vm_bytecode_assembler_module_deinitialize(&module_);
  }

  iree_vm_bytecode_assembler_module_t module_;
};

TEST_F(VMBytecodeAssemblerParserTest, PadsFunctionBodiesBetweenDescriptors) {
  static const char kSource[] = R"(
vm.module @parser_test version 0

vm.func @one() -> () {
^bb0:
  vm.return
}

vm.func @two() -> () {
^bb0:
  vm.return
}
)";

  IREE_ASSERT_OK(iree_vm_bytecode_assembler_parse_source(
      &module_, iree_make_string_view(kSource, sizeof(kSource) - 1)));

  ASSERT_EQ(module_.function_count, 2u);
  EXPECT_EQ(module_.functions[0].bytecode_offset, 0u);
  EXPECT_EQ(module_.functions[0].bytecode_length, 4u);
  EXPECT_EQ(module_.functions[1].bytecode_offset, 8u);
  EXPECT_EQ(module_.functions[1].bytecode_length, 4u);
}

}  // namespace
