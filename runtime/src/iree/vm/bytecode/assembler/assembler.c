// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/assembler/assembler.h"

#include "iree/vm/bytecode/assembler/archive.h"
#include "iree/vm/bytecode/assembler/module.h"
#include "iree/vm/bytecode/assembler/parser.h"

IREE_API_EXPORT iree_status_t iree_vm_bytecode_assembler_assemble(
    iree_string_view_t source, iree_allocator_t host_allocator,
    iree_byte_span_t* out_archive) {
  IREE_ASSERT_ARGUMENT(out_archive);
  *out_archive = iree_byte_span_empty();

  iree_vm_bytecode_assembler_module_t module;
  iree_vm_bytecode_assembler_module_initialize(host_allocator, &module);

  iree_status_t status =
      iree_vm_bytecode_assembler_parse_source(&module, source);
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_assembler_assign_global_ordinals(&module);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_assembler_resolve_global_fixups(&module);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_assembler_resolve_function_fixups(&module);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_assembler_build_archive(&module, out_archive);
  }

  iree_vm_bytecode_assembler_module_deinitialize(&module);
  return status;
}
