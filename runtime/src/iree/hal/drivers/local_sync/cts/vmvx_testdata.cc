// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// VMVX executable format registration for the local-sync CTS2 backend.
//
// This is a separate library from backends.cc so that non-executable tests
// (allocator, semaphore, etc.) do not transitively depend on iree-compile.
// Only test binaries that link executable or dispatch test suites need this.

#include "iree/hal/cts2/util/registry.h"
#include "runtime/src/iree/hal/drivers/local_sync/cts/testdata_vmvx.h"

namespace iree::hal::cts {

static iree_const_byte_span_t GetVmvxExecutableData(
    iree_string_view_t file_name) {
  const iree_file_toc_t* toc = iree_cts_testdata_vmvx_create();
  for (size_t i = 0; toc[i].name != nullptr; ++i) {
    if (iree_string_view_equal(file_name,
                               iree_make_cstring_view(toc[i].name))) {
      return iree_make_const_byte_span(
          reinterpret_cast<const uint8_t*>(toc[i].data), toc[i].size);
    }
  }
  return iree_const_byte_span_empty();
}

static bool vmvx_format_registered_ =
    (CtsRegistry::RegisterExecutableFormat(
         "local_sync", {"vmvx", "vmvx-bytecode-fb", GetVmvxExecutableData}),
     true);

}  // namespace iree::hal::cts
