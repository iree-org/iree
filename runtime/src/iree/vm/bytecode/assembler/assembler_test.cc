// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/assembler/assembler.h"

#include <string>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

namespace {

TEST(VMBytecodeAssemblerTest, ProducesLoadableModule) {
  static const char kSource[] = R"(
vm.module @assembler_test version 0
vm.export @add_i32
vm.func @add_i32(%i0: i32, %i1: i32) -> (i32) {
^bb0:
  %i2 = vm.add.i32 %i0, %i1
  vm.return %i2
}
)";

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_byte_span_t archive = iree_byte_span_empty();
  IREE_ASSERT_OK(iree_vm_bytecode_assembler_assemble(
      iree_make_string_view(kSource, sizeof(kSource) - 1), host_allocator,
      &archive));

  iree_vm_instance_t* instance = nullptr;
  IREE_ASSERT_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                         host_allocator, &instance));

  iree_vm_module_t* module = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      instance, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
      iree_const_cast_byte_span(archive), iree_allocator_null(), host_allocator,
      &module));

  iree_string_builder_t disassembly_builder;
  iree_string_builder_initialize(host_allocator, &disassembly_builder);
  IREE_ASSERT_OK(iree_vm_bytecode_module_disassemble_function(
      module, 0, IREE_VM_BYTECODE_DISASSEMBLY_FORMAT_DEFAULT,
      &disassembly_builder));
  iree_string_view_t disassembly =
      iree_string_builder_view(&disassembly_builder);
  std::string disassembly_text(disassembly.data, disassembly.size);
  EXPECT_THAT(disassembly_text, testing::HasSubstr("^bb0:"));
  EXPECT_THAT(disassembly_text,
              testing::HasSubstr("%i2 = vm.add.i32 %i0, %i1"));
  EXPECT_THAT(disassembly_text, testing::HasSubstr("vm.return %i2"));
  iree_string_builder_deinitialize(&disassembly_builder);

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      module, IREE_VM_FUNCTION_LINKAGE_EXPORT, IREE_SV("add_i32"), &function));

  iree_vm_module_release(module);
  iree_vm_instance_release(instance);
  iree_allocator_free(host_allocator, archive.data);
}

TEST(VMBytecodeAssemblerTest, AcceptsExplicitExportAlias) {
  static const char kSource[] = R"(
vm.module @assembler_test version 0
vm.export @public_add = @add_i32
vm.func @add_i32() -> () {
^bb0:
  vm.return
}
)";

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_byte_span_t archive = iree_byte_span_empty();
  IREE_ASSERT_OK(iree_vm_bytecode_assembler_assemble(
      iree_make_string_view(kSource, sizeof(kSource) - 1), host_allocator,
      &archive));

  iree_vm_instance_t* instance = nullptr;
  IREE_ASSERT_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                         host_allocator, &instance));

  iree_vm_module_t* module = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      instance, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
      iree_const_cast_byte_span(archive), iree_allocator_null(), host_allocator,
      &module));

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      module, IREE_VM_FUNCTION_LINKAGE_EXPORT, IREE_SV("public_add"),
      &function));

  iree_vm_module_release(module);
  iree_vm_instance_release(instance);
  iree_allocator_free(host_allocator, archive.data);
}

}  // namespace
