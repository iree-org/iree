// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IOCP backend registration for CTS tests.
//
// The IOCP backend uses I/O Completion Ports for Windows-native async I/O.
// It supports multishot via emulated resubmission and buffer registration
// via userspace pool management. Tests tagged with zerocopy, dmabuf, or
// futex capabilities will be skipped.
//
// Backends registered:
//   - iocp: Full IOCP backend with all available capabilities.

#include "iree/async/cts/util/registry.h"
#include "iree/async/platform/iocp/api.h"
#include "iree/async/proactor.h"

namespace iree::async::cts {

//===----------------------------------------------------------------------===//
// Factory functions
//===----------------------------------------------------------------------===//

// Creates an IOCP proactor with full capabilities.
static iree::StatusOr<iree_async_proactor_t*> CreateIOCPProactor() {
  iree_async_proactor_t* proactor = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_async_proactor_create_iocp(iree_async_proactor_options_default(),
                                      iree_allocator_system(), &proactor));
  return proactor;
}

//===----------------------------------------------------------------------===//
// Backend registration
//===----------------------------------------------------------------------===//

// IOCP backend: Windows-native, completion-based async I/O.
static bool iocp_registered_ = (CtsRegistry::RegisterBackend({
                                    "iocp",
                                    {"iocp", CreateIOCPProactor},
                                    {"portable", "multishot"},
                                }),
                                true);

}  // namespace iree::async::cts
