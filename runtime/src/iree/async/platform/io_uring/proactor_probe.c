// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Capability detection for io_uring proactor.
//
// This module probes the kernel for supported io_uring features and opcodes
// to determine what capabilities are available at runtime. The probing happens
// once during proactor creation and the results are cached.

#include <errno.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "iree/async/platform/io_uring/defs.h"
#include "iree/async/platform/io_uring/uring.h"
#include "iree/async/proactor.h"

//===----------------------------------------------------------------------===//
// Capability detection
//===----------------------------------------------------------------------===//

// Maximum opcode we need to probe. SEND_ZC is 47, SENDMSG_ZC is 48.
// Round up to 64 for a clean uint64_t bitmask.
#define IREE_IO_URING_PROBE_OPCODE_COUNT 64

// Probes the kernel for supported io_uring opcodes.
// Returns a bitmask where bit N is set if opcode N is supported.
// This is used to detect kernel features that don't have feature flags.
//
// The probe is performed via IORING_REGISTER_PROBE which requires an
// active ring (so this must be called after io_uring_setup succeeds).
static iree_status_t iree_async_proactor_io_uring_probe_opcodes(
    int ring_fd, uint64_t* out_supported_opcodes) {
  *out_supported_opcodes = 0;

  // Stack-allocate the probe structure with space for opcodes.
  // We use a union to ensure proper alignment for the probe struct while
  // providing enough space for the flexible array member.
  union {
    iree_io_uring_probe_t probe;
    uint8_t storage[sizeof(iree_io_uring_probe_t) +
                    IREE_IO_URING_PROBE_OPCODE_COUNT *
                        sizeof(iree_io_uring_probe_op_t)];
  } probe_buffer;
  memset(&probe_buffer, 0, sizeof(probe_buffer));

  iree_io_uring_probe_t* probe = &probe_buffer.probe;

  // Perform the probe syscall. Retry on EINTR since io_uring_register can be
  // interrupted by signals (common when attached to debuggers/profilers).
  long ret;
  do {
    ret = syscall(IREE_IO_URING_SYSCALL_REGISTER, ring_fd,
                  IREE_IORING_REGISTER_PROBE, probe,
                  IREE_IO_URING_PROBE_OPCODE_COUNT);
  } while (ret < 0 && errno == EINTR);
  if (ret < 0) {
    // Probe not supported on older kernels (pre-5.6). This is fine - we just
    // won't enable advanced capabilities.
    return iree_make_status(iree_status_code_from_errno(errno),
                            "io_uring_register PROBE failed");
  }

  // Build the bitmask from probe results.
  uint64_t mask = 0;
  uint8_t ops_count = probe->ops_len;
  if (ops_count > IREE_IO_URING_PROBE_OPCODE_COUNT) {
    ops_count = IREE_IO_URING_PROBE_OPCODE_COUNT;
  }
  for (uint8_t i = 0; i < ops_count; ++i) {
    if (probe->ops[i].flags & IREE_IO_URING_OP_SUPPORTED) {
      uint8_t opcode = probe->ops[i].op;
      if (opcode < 64) {
        mask |= (1ULL << opcode);
      }
    }
  }

  *out_supported_opcodes = mask;
  return iree_ok_status();
}

// Minimum kernel feature requirements and capability detection.
//
// Baseline: kernel 5.7+ (IORING_FEAT_FAST_POLL)
// This baseline provides:
//   - Linked SQEs (IOSQE_IO_LINK, 5.3+)
//   - Absolute timeouts (IORING_TIMEOUT_ABS, 5.4+)
//   - Fast poll for efficient POLL_ADD (5.7+)
//   - Fixed files via IORING_REGISTER_FILES (5.1+)
//   - Registered buffers via IORING_REGISTER_BUFFERS (5.1+)
//
// Conditional capabilities (probed via io_uring_register PROBE):
//   - MULTISHOT: SOCKET opcode (45) present → kernel 5.19+
//   - ZERO_COPY_SEND: SEND_ZC opcode (47) present → kernel 6.0+
//
// When bumping the baseline to 6.0+, move MULTISHOT and ZERO_COPY_SEND
// to the unconditional set and optionally remove the probe logic.
//
// Returns IREE_STATUS_UNAVAILABLE if the kernel is too old.
iree_status_t iree_async_proactor_io_uring_detect_capabilities(
    int ring_fd, uint32_t ring_features,
    iree_async_proactor_capabilities_t* out_capabilities) {
  *out_capabilities = IREE_ASYNC_PROACTOR_CAPABILITY_NONE;

  // Require kernel 5.7+ for baseline io_uring stability.
  // IORING_FEAT_FAST_POLL was added in 5.7 and is our minimum requirement.
  if (!iree_any_bit_set(ring_features, IREE_IORING_FEAT_FAST_POLL)) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "io_uring requires Linux kernel 5.7 or later; this kernel lacks "
        "IORING_FEAT_FAST_POLL (features=0x%08x). On older kernels, IREE "
        "falls back to the portable threaded backend. See "
        "https://iree.dev/guides/deployment/linux-kernel-requirements",
        ring_features);
  }

  // Baseline capabilities (5.7+): always present when we get here.
  *out_capabilities = IREE_ASYNC_PROACTOR_CAPABILITY_ABSOLUTE_TIMEOUT |
                      IREE_ASYNC_PROACTOR_CAPABILITY_LINKED_OPERATIONS |
                      IREE_ASYNC_PROACTOR_CAPABILITY_FIXED_FILES |
                      IREE_ASYNC_PROACTOR_CAPABILITY_REGISTERED_BUFFERS |
                      IREE_ASYNC_PROACTOR_CAPABILITY_DMABUF |
                      IREE_ASYNC_PROACTOR_CAPABILITY_DEVICE_FENCE;

  // Probe for newer capabilities that require specific opcodes.
  uint64_t supported_opcodes = 0;
  iree_status_t probe_status =
      iree_async_proactor_io_uring_probe_opcodes(ring_fd, &supported_opcodes);
  if (iree_status_is_ok(probe_status)) {
    // MULTISHOT: available since 5.19 with ACCEPT.
    // We detect 5.19+ by checking for SOCKET opcode (45), added in 5.19.
    if (supported_opcodes & (1ULL << IREE_IORING_OP_SOCKET)) {
      *out_capabilities |= IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT;
    }

    // ZERO_COPY_SEND: available since 6.0.
    // Check for SEND_ZC opcode (47) directly.
    if (supported_opcodes & (1ULL << IREE_IORING_OP_SEND_ZC)) {
      *out_capabilities |= IREE_ASYNC_PROACTOR_CAPABILITY_ZERO_COPY_SEND;
    }

    // FUTEX_OPERATIONS: available since 6.7.
    // Check for FUTEX_WAIT opcode (51) directly.
    if (supported_opcodes & (1ULL << IREE_IORING_OP_FUTEX_WAIT)) {
      *out_capabilities |= IREE_ASYNC_PROACTOR_CAPABILITY_FUTEX_OPERATIONS;
    }

    // PROACTOR_MESSAGING: available since 5.18.
    // Check for MSG_RING opcode (40) directly.
    if (supported_opcodes & (1ULL << IREE_IORING_OP_MSG_RING)) {
      *out_capabilities |= IREE_ASYNC_PROACTOR_CAPABILITY_PROACTOR_MESSAGING;
    }
  } else {
    // Probe failed - not fatal, just means we don't enable advanced features.
    // This can happen on kernels 5.6 and earlier that don't support PROBE.
    iree_status_ignore(probe_status);
  }

  return iree_ok_status();
}
