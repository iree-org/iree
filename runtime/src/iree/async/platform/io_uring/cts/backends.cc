// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// io_uring backend registration for CTS tests.
//
// This file registers io_uring proactor configurations with different
// capability combinations. Each configuration validates a different code path:
//
// Configurations:
//   - io_uring: Full capabilities (zerocopy, multishot). Tests optimal path.
//   - io_uring_no_zerocopy: Multishot only. Tests non-zerocopy send path.
//   - io_uring_no_multishot: Zerocopy only. Tests single-shot accept/recv.
//   - io_uring_no_messaging: No MSG_RING. Tests fallback message passing.
//   - io_uring_minimal: No optional features. Tests base io_uring path.
//
// Tags:
//   - zerocopy: SO_ZEROCOPY / MSG_ZEROCOPY support (kernel 4.14+)
//   - multishot: Multishot accept/recv support (kernel 5.19+)

#include "iree/async/cts/util/registry.h"
#include "iree/async/platform/io_uring/api.h"
#include "iree/async/proactor.h"

namespace iree::async::cts {

//===----------------------------------------------------------------------===//
// Factory functions
//===----------------------------------------------------------------------===//

// Creates an io_uring proactor with default options (full capabilities).
static iree::StatusOr<iree_async_proactor_t*> CreateIoUringProactor() {
  iree_async_proactor_t* proactor = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_async_proactor_create_io_uring(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor));
  return proactor;
}

// Creates an io_uring proactor with zerocopy disabled.
// Tests the non-zerocopy send path even on kernels that support it.
static iree::StatusOr<iree_async_proactor_t*> CreateIoUringNoZerocopy() {
  iree_async_proactor_options_t options = iree_async_proactor_options_default();
  options.allowed_capabilities &=
      ~IREE_ASYNC_PROACTOR_CAPABILITY_ZERO_COPY_SEND;
  iree_async_proactor_t* proactor = nullptr;
  IREE_RETURN_IF_ERROR(iree_async_proactor_create_io_uring(
      options, iree_allocator_system(), &proactor));
  return proactor;
}

// Creates an io_uring proactor with multishot disabled.
// Tests the single-shot accept/recv path even on kernels that support
// multishot.
static iree::StatusOr<iree_async_proactor_t*> CreateIoUringNoMultishot() {
  iree_async_proactor_options_t options = iree_async_proactor_options_default();
  options.allowed_capabilities &= ~IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT;
  iree_async_proactor_t* proactor = nullptr;
  IREE_RETURN_IF_ERROR(iree_async_proactor_create_io_uring(
      options, iree_allocator_system(), &proactor));
  return proactor;
}

// Creates an io_uring proactor with minimal capabilities.
// Tests the base io_uring path without optional features.
static iree::StatusOr<iree_async_proactor_t*> CreateIoUringMinimal() {
  iree_async_proactor_options_t options = iree_async_proactor_options_default();
  options.allowed_capabilities &=
      ~(IREE_ASYNC_PROACTOR_CAPABILITY_ZERO_COPY_SEND |
        IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT);
  iree_async_proactor_t* proactor = nullptr;
  IREE_RETURN_IF_ERROR(iree_async_proactor_create_io_uring(
      options, iree_allocator_system(), &proactor));
  return proactor;
}

// Creates an io_uring proactor with PROACTOR_MESSAGING disabled.
// Tests the fallback message passing path (MPSC queue instead of MSG_RING).
static iree::StatusOr<iree_async_proactor_t*> CreateIoUringNoMessaging() {
  iree_async_proactor_options_t options = iree_async_proactor_options_default();
  options.allowed_capabilities &=
      ~IREE_ASYNC_PROACTOR_CAPABILITY_PROACTOR_MESSAGING;
  iree_async_proactor_t* proactor = nullptr;
  IREE_RETURN_IF_ERROR(iree_async_proactor_create_io_uring(
      options, iree_allocator_system(), &proactor));
  return proactor;
}

//===----------------------------------------------------------------------===//
// Backend registration
//===----------------------------------------------------------------------===//

// Full capabilities: zerocopy + multishot enabled.
// ZeroCopyTest and MultishotTest will run against this configuration.
static bool io_uring_full_registered_ =
    (CtsRegistry::RegisterBackend({
         "io_uring",
         {"io_uring", CreateIoUringProactor},
         {"zerocopy", "multishot"},
     }),
     true);

// No zerocopy: tests non-zerocopy send path.
// ZeroCopyTest will NOT run (lacks "zerocopy" tag).
// MultishotTest will run (has "multishot" tag).
static bool io_uring_no_zc_registered_ =
    (CtsRegistry::RegisterBackend({
         "io_uring_no_zerocopy",
         {"io_uring_no_zerocopy", CreateIoUringNoZerocopy},
         {"multishot"},
     }),
     true);

// No multishot: tests single-shot accept/recv path.
// MultishotTest will NOT run (lacks "multishot" tag).
// ZeroCopyTest will run (has "zerocopy" tag).
static bool io_uring_no_ms_registered_ =
    (CtsRegistry::RegisterBackend({
         "io_uring_no_multishot",
         {"io_uring_no_multishot", CreateIoUringNoMultishot},
         {"zerocopy"},
     }),
     true);

// No messaging: tests fallback message passing path.
// Uses MPSC queue instead of MSG_RING for cross-proactor messaging.
// Still has zerocopy and multishot capabilities.
static bool io_uring_no_msg_registered_ =
    (CtsRegistry::RegisterBackend({
         "io_uring_no_messaging",
         {"io_uring_no_messaging", CreateIoUringNoMessaging},
         {"zerocopy", "multishot"},
     }),
     true);

// Minimal: no optional features.
// Neither ZeroCopyTest nor MultishotTest will run.
// Tests base io_uring functionality with fallback paths.
static bool io_uring_minimal_registered_ =
    (CtsRegistry::RegisterBackend({
         "io_uring_minimal",
         {"io_uring_minimal", CreateIoUringMinimal},
         {},
     }),
     true);

}  // namespace iree::async::cts
