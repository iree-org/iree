// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/gtest.h"
#include "iree/vm/bytecode/isa/encoding_table.h"

namespace {

TEST(VMISAEncodingTableTest, LooksUpCoreMnemonic) {
  const iree_vm_isa_instruction_t* instruction =
      iree_vm_isa_lookup_mnemonic(iree_make_cstring_view("add.i32"));
  ASSERT_NE(instruction, nullptr);
  EXPECT_TRUE(iree_string_view_equal(instruction->symbol,
                                     iree_make_cstring_view("AddI32")));
  EXPECT_TRUE(iree_string_view_equal(instruction->encoding,
                                     iree_make_cstring_view("i32_binary")));
  EXPECT_EQ(instruction->opcode_set, IREE_VM_ISA_OPCODE_SET_CORE);
  EXPECT_EQ(instruction->opcode, IREE_VM_OP_CORE_AddI32);
  EXPECT_EQ(instruction->required_features, 0);

  ASSERT_EQ(instruction->field_count, 3);
  EXPECT_TRUE(iree_string_view_equal(instruction->fields[0].name,
                                     iree_make_cstring_view("left")));
  EXPECT_EQ(instruction->fields[0].kind, IREE_VM_ISA_FIELD_KIND_REGISTER);
  EXPECT_EQ(instruction->fields[0].register_bank,
            IREE_VM_ISA_REGISTER_BANK_I32);
  EXPECT_EQ(instruction->fields[0].access, IREE_VM_ISA_FIELD_ACCESS_READ);

  EXPECT_TRUE(iree_string_view_equal(instruction->fields[1].name,
                                     iree_make_cstring_view("right")));
  EXPECT_EQ(instruction->fields[1].kind, IREE_VM_ISA_FIELD_KIND_REGISTER);
  EXPECT_EQ(instruction->fields[1].register_bank,
            IREE_VM_ISA_REGISTER_BANK_I32);
  EXPECT_EQ(instruction->fields[1].access, IREE_VM_ISA_FIELD_ACCESS_READ);

  EXPECT_TRUE(iree_string_view_equal(instruction->fields[2].name,
                                     iree_make_cstring_view("result")));
  EXPECT_EQ(instruction->fields[2].kind, IREE_VM_ISA_FIELD_KIND_REGISTER);
  EXPECT_EQ(instruction->fields[2].register_bank,
            IREE_VM_ISA_REGISTER_BANK_I32);
  EXPECT_EQ(instruction->fields[2].access, IREE_VM_ISA_FIELD_ACCESS_WRITE);
}

TEST(VMISAEncodingTableTest, LooksUpExtensionOpcode) {
  const iree_vm_isa_opcode_set_descriptor_t* opcode_set =
      iree_vm_isa_opcode_set_descriptor(IREE_VM_ISA_OPCODE_SET_EXT_F32);
  ASSERT_NE(opcode_set, nullptr);
  EXPECT_TRUE(iree_string_view_equal(opcode_set->id,
                                     iree_make_cstring_view("ext_f32")));
  EXPECT_TRUE(iree_string_view_equal(opcode_set->prefix_symbol,
                                     iree_make_cstring_view("PrefixExtF32")));
  EXPECT_EQ(opcode_set->prefix_opcode, IREE_VM_OP_CORE_PrefixExtF32);
  EXPECT_EQ(opcode_set->required_features, iree_vm_FeatureBits_EXT_F32);

  const iree_vm_isa_instruction_t* instruction = iree_vm_isa_lookup_opcode(
      IREE_VM_ISA_OPCODE_SET_EXT_F32, IREE_VM_OP_EXT_F32_AddF32);
  ASSERT_NE(instruction, nullptr);
  EXPECT_TRUE(iree_string_view_equal(instruction->mnemonic,
                                     iree_make_cstring_view("add.f32")));
  EXPECT_TRUE(iree_string_view_equal(instruction->encoding,
                                     iree_make_cstring_view("f32_binary")));
  EXPECT_EQ(instruction->required_features, iree_vm_FeatureBits_EXT_F32);
}

TEST(VMISAEncodingTableTest, MissingLookupsReturnNull) {
  EXPECT_EQ(iree_vm_isa_lookup_mnemonic(iree_make_cstring_view("missing")),
            nullptr);
  EXPECT_EQ(iree_vm_isa_lookup_opcode(IREE_VM_ISA_OPCODE_SET_COUNT, 0),
            nullptr);
  EXPECT_EQ(iree_vm_isa_opcode_set_descriptor(IREE_VM_ISA_OPCODE_SET_COUNT),
            nullptr);
}

}  // namespace
