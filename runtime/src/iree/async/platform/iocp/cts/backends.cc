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
//   - iocp_legacy_wait: IOCP with NtAssociateWaitCompletionPacket disabled,
//     forcing the RegisterWaitForSingleObject fallback path.

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
static bool iocp_registered_ =
    (CtsRegistry::RegisterBackend({
         "iocp",
         {"iocp", CreateIOCPProactor},
         {"portable", "multishot", "shared_notification"},
     }),
     true);

// Creates an IOCP proactor with NtAssociateWaitCompletionPacket disabled.
// Forces the RegisterWaitForSingleObject fallback path for Event-to-IOCP
// bridging, exercising the threadpool-based wait registration code path even
// on systems where the NT wait completion packet API is available.
static iree::StatusOr<iree_async_proactor_t*> CreateIOCPLegacyWaitProactor() {
  iree_async_proactor_options_t options = iree_async_proactor_options_default();
  options.allowed_capabilities &=
      ~IREE_ASYNC_PROACTOR_CAPABILITY_WAIT_COMPLETION_PACKET;
  iree_async_proactor_t* proactor = nullptr;
  IREE_RETURN_IF_ERROR(iree_async_proactor_create_iocp(
      options, iree_allocator_system(), &proactor));
  return proactor;
}

// IOCP with RegisterWaitForSingleObject fallback: tests the threadpool-based
// Event-to-IOCP bridging path for both event wait operations and shared
// notification wake events.
static bool iocp_legacy_wait_registered_ =
    (CtsRegistry::RegisterBackend({
         "iocp_legacy_wait",
         {"iocp_legacy_wait", CreateIOCPLegacyWaitProactor},
         {"portable", "multishot", "shared_notification"},
     }),
     true);

}  // namespace iree::async::cts
