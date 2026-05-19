// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/assembler/module.h"

#include <cstring>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;

class VMBytecodeAssemblerModuleTest : public testing::Test {
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

TEST_F(VMBytecodeAssemblerModuleTest, RejectsDuplicateLabels) {
  IREE_ASSERT_OK(
      iree_vm_bytecode_assembler_append_label(&module_, IREE_SV("bb0"), 0));
  EXPECT_THAT(Status(iree_vm_bytecode_assembler_append_label(
                  &module_, IREE_SV("bb0"), 8)),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(VMBytecodeAssemblerModuleTest, RejectsDuplicateGlobals) {
  IREE_ASSERT_OK(
      iree_vm_bytecode_assembler_append_global(&module_, IREE_SV("value"), 4));
  EXPECT_THAT(Status(iree_vm_bytecode_assembler_append_global(
                  &module_, IREE_SV("value"), 8)),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(VMBytecodeAssemblerModuleTest, RejectsDuplicateExports) {
  IREE_ASSERT_OK(iree_vm_bytecode_assembler_append_export(
      &module_, IREE_SV("public"), IREE_SV("internal")));
  EXPECT_THAT(Status(iree_vm_bytecode_assembler_append_export(
                  &module_, IREE_SV("public"), IREE_SV("other_internal"))),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(VMBytecodeAssemblerModuleTest, AssignsGlobalStorageOrdinals) {
  IREE_ASSERT_OK(
      iree_vm_bytecode_assembler_append_global(&module_, IREE_SV("i64"), 8));
  IREE_ASSERT_OK(
      iree_vm_bytecode_assembler_append_global(&module_, IREE_SV("i32"), 4));
  IREE_ASSERT_OK(
      iree_vm_bytecode_assembler_append_global(&module_, IREE_SV("ref"), 0));

  IREE_ASSERT_OK(iree_vm_bytecode_assembler_assign_global_ordinals(&module_));

  EXPECT_EQ(module_.globals[0].ordinal, 8u);
  EXPECT_EQ(module_.globals[1].ordinal, 0u);
  EXPECT_EQ(module_.globals[2].ordinal, 0u);
  EXPECT_EQ(module_.global_byte_capacity, 16u);
  EXPECT_EQ(module_.global_ref_count, 1u);
}

TEST_F(VMBytecodeAssemblerModuleTest, ResolvesGlobalFixups) {
  IREE_ASSERT_OK(
      iree_vm_bytecode_assembler_append_global(&module_, IREE_SV("i32"), 4));
  IREE_ASSERT_OK(iree_vm_bytecode_assembler_assign_global_ordinals(&module_));

  char* bytes = NULL;
  IREE_ASSERT_OK(
      iree_string_builder_append_inline(&module_.bytecode_builder, 4, &bytes));
  memset(bytes, 0xCD, 4);
  IREE_ASSERT_OK(iree_vm_bytecode_assembler_append_global_fixup(
      &module_, IREE_SV("i32"), /*bytecode_offset=*/0, /*storage_size=*/4));

  IREE_ASSERT_OK(iree_vm_bytecode_assembler_resolve_global_fixups(&module_));

  const uint8_t* patched_bytes =
      (const uint8_t*)iree_string_builder_view(&module_.bytecode_builder).data;
  EXPECT_EQ(patched_bytes[0], 0u);
  EXPECT_EQ(patched_bytes[1], 0u);
  EXPECT_EQ(patched_bytes[2], 0u);
  EXPECT_EQ(patched_bytes[3], 0u);
}

TEST_F(VMBytecodeAssemblerModuleTest, RejectsUnknownGlobalFixups) {
  char* bytes = NULL;
  IREE_ASSERT_OK(
      iree_string_builder_append_inline(&module_.bytecode_builder, 4, &bytes));
  memset(bytes, 0xCD, 4);
  IREE_ASSERT_OK(iree_vm_bytecode_assembler_append_global_fixup(
      &module_, IREE_SV("missing"), /*bytecode_offset=*/0, /*storage_size=*/4));

  EXPECT_THAT(
      Status(iree_vm_bytecode_assembler_resolve_global_fixups(&module_)),
      StatusIs(StatusCode::kInvalidArgument));
}

}  // namespace
