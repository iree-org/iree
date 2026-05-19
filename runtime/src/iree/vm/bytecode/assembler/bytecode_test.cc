// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/assembler/bytecode.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;

class VMBytecodeAssemblerBytecodeTest : public testing::Test {
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

TEST_F(VMBytecodeAssemblerBytecodeTest, ParsesI32Registers) {
  uint16_t register_ordinal = 0;
  IREE_ASSERT_OK(iree_vm_bytecode_assembler_parse_register(
      &module_, IREE_SV("%i7"), IREE_VM_ISA_REGISTER_BANK_I32,
      &register_ordinal));
  EXPECT_EQ(register_ordinal, 7u);
  EXPECT_EQ(module_.i32_register_count, 8u);
}

TEST_F(VMBytecodeAssemblerBytecodeTest, ParsesI64RegisterPairs) {
  uint16_t register_ordinal = 0;
  IREE_ASSERT_OK(iree_vm_bytecode_assembler_parse_register(
      &module_, IREE_SV("%i2:3"), IREE_VM_ISA_REGISTER_BANK_I64,
      &register_ordinal));
  EXPECT_EQ(register_ordinal, 2u);
  EXPECT_EQ(module_.i32_register_count, 4u);
}

TEST_F(VMBytecodeAssemblerBytecodeTest, RejectsMalformedRegisterPairs) {
  uint16_t register_ordinal = 0;
  EXPECT_THAT(Status(iree_vm_bytecode_assembler_parse_register(
                  &module_, IREE_SV("%i2"), IREE_VM_ISA_REGISTER_BANK_I64,
                  &register_ordinal)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(Status(iree_vm_bytecode_assembler_parse_register(
                  &module_, IREE_SV("%i2:4"), IREE_VM_ISA_REGISTER_BANK_I64,
                  &register_ordinal)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(Status(iree_vm_bytecode_assembler_parse_register(
                  &module_, IREE_SV("%i2:3"), IREE_VM_ISA_REGISTER_BANK_I32,
                  &register_ordinal)),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(VMBytecodeAssemblerBytecodeTest, EncodesCoreInstruction) {
  IREE_ASSERT_OK(iree_vm_bytecode_assembler_parse_instruction(
      &module_, IREE_SV("%i2 = vm.add.i32 %i0, %i1")));

  iree_string_view_t bytecode =
      iree_string_builder_view(&module_.bytecode_builder);
  EXPECT_GT(bytecode.size, 0u);
  EXPECT_EQ(module_.i32_register_count, 3u);
}

TEST_F(VMBytecodeAssemblerBytecodeTest, EncodesSelectInstruction) {
  IREE_ASSERT_OK(iree_vm_bytecode_assembler_parse_instruction(
      &module_, IREE_SV("%i8:9 = vm.select.i64 %i6 ? %i4:5 : %i2:3")));

  iree_string_view_t bytecode =
      iree_string_builder_view(&module_.bytecode_builder);
  EXPECT_EQ(bytecode.size, 9u);
  EXPECT_EQ(module_.i32_register_count, 10u);
}

TEST_F(VMBytecodeAssemblerBytecodeTest, EncodesSwitchInstruction) {
  IREE_ASSERT_OK(iree_vm_bytecode_assembler_parse_instruction(
      &module_, IREE_SV("%i0:1 = vm.switch.i64 %i2[%i10, %i12] else %i4:5")));

  iree_string_view_t bytecode =
      iree_string_builder_view(&module_.bytecode_builder);
  EXPECT_EQ(bytecode.size, 14u);
  EXPECT_EQ(module_.i32_register_count, 14u);
}

TEST_F(VMBytecodeAssemblerBytecodeTest, EncodesF32ExtensionInstruction) {
  IREE_ASSERT_OK(iree_vm_bytecode_assembler_parse_instruction(
      &module_, IREE_SV("%i2 = vm.const.f32 1.500000")));

  iree_string_view_t bytecode =
      iree_string_builder_view(&module_.bytecode_builder);
  EXPECT_EQ(bytecode.size, 8u);
  EXPECT_EQ(module_.i32_register_count, 3u);
  EXPECT_EQ(module_.function_requirements, iree_vm_FeatureBits_EXT_F32);
  EXPECT_EQ(module_.module_requirements, iree_vm_FeatureBits_EXT_F32);
}

TEST_F(VMBytecodeAssemblerBytecodeTest, EncodesF64ExtensionInstruction) {
  IREE_ASSERT_OK(iree_vm_bytecode_assembler_parse_instruction(
      &module_, IREE_SV("%i2:3 = vm.const.f64 1.500000")));

  iree_string_view_t bytecode =
      iree_string_builder_view(&module_.bytecode_builder);
  EXPECT_EQ(bytecode.size, 12u);
  EXPECT_EQ(module_.i32_register_count, 4u);
  EXPECT_EQ(module_.function_requirements, iree_vm_FeatureBits_EXT_F64);
  EXPECT_EQ(module_.module_requirements, iree_vm_FeatureBits_EXT_F64);
}

}  // namespace
