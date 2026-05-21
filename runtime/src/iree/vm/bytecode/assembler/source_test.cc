// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/assembler/source.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;

TEST(VMBytecodeAssemblerSourceTest, SplitsFirstToken) {
  iree_string_view_t token = iree_string_view_empty();
  iree_string_view_t remainder = iree_string_view_empty();
  iree_vm_bytecode_assembler_split_first_token(IREE_SV("  vm.func @foo()  "),
                                               &token, &remainder);
  EXPECT_TRUE(iree_string_view_equal(token, IREE_SV("vm.func")));
  EXPECT_TRUE(iree_string_view_equal(remainder, IREE_SV("@foo()")));
}

TEST(VMBytecodeAssemblerSourceTest, ParsesSymbolsWithoutSigils) {
  iree_string_view_t symbol = iree_string_view_empty();
  IREE_ASSERT_OK(iree_vm_bytecode_assembler_parse_symbol(
      IREE_SV(" @module.name "), &symbol));
  EXPECT_TRUE(iree_string_view_equal(symbol, IREE_SV("module.name")));
}

TEST(VMBytecodeAssemblerSourceTest, SplitsOperandsWithQuotedCommas) {
  iree_string_view_t operand = iree_string_view_empty();
  iree_string_view_t remainder = iree_string_view_empty();
  iree_vm_bytecode_assembler_split_operand(IREE_SV("%i0, \"smin(-3,2)=-3\""),
                                           &operand, &remainder);
  EXPECT_TRUE(iree_string_view_equal(operand, IREE_SV("%i0")));
  EXPECT_TRUE(iree_string_view_equal(remainder, IREE_SV("\"smin(-3,2)=-3\"")));

  iree_vm_bytecode_assembler_split_operand(remainder, &operand, &remainder);
  EXPECT_TRUE(iree_string_view_equal(operand, IREE_SV("\"smin(-3,2)=-3\"")));
  EXPECT_TRUE(iree_vm_bytecode_assembler_string_view_is_empty(remainder));
}

TEST(VMBytecodeAssemblerSourceTest, RejectsInvalidSymbols) {
  iree_string_view_t symbol = iree_string_view_empty();
  EXPECT_THAT(Status(iree_vm_bytecode_assembler_parse_symbol(
                  IREE_SV("module.name"), &symbol)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(Status(iree_vm_bytecode_assembler_parse_symbol(
                  IREE_SV("@module-name"), &symbol)),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST(VMBytecodeAssemblerSourceTest, AllowsVmSymbolCharacters) {
  EXPECT_TRUE(iree_vm_bytecode_assembler_is_symbol_char('_'));
  EXPECT_TRUE(iree_vm_bytecode_assembler_is_symbol_char('$'));
  EXPECT_TRUE(iree_vm_bytecode_assembler_is_symbol_char('.'));
  EXPECT_FALSE(iree_vm_bytecode_assembler_is_symbol_char('-'));
}

}  // namespace
