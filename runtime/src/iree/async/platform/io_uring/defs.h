// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// io_uring kernel interface definitions.
//
// These structures and constants match the Linux kernel's io_uring interface.
// We define them here rather than including <linux/io_uring.h> to:
//   - Avoid dependency on kernel headers at specific versions
//   - Support cross-compilation where host headers differ from target
//   - Control exactly which features we use
//
// Structures must match the kernel ABI exactly - do not modify field order,
// sizes, or alignment.

#ifndef IREE_ASYNC_PLATFORM_IO_URING_DEFS_H_
#define IREE_ASYNC_PLATFORM_IO_URING_DEFS_H_

#include <stdint.h>

#include "iree/base/target_platform.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Syscall numbers and setup flags
//===----------------------------------------------------------------------===//

// io_uring syscall numbers for verified architectures. x86, ARM, and RISC-V
// (both 32-bit and 64-bit) all assign 425/426/427. ARM and RISC-V share the
// generic syscall table; x86 happens to use the same values in both its 32-bit
// and 64-bit tables. Other architectures (MIPS, s390x) have different numbering
// schemes — add them explicitly after verifying.
#if defined(IREE_ARCH_X86_32) || defined(IREE_ARCH_X86_64) || \
    defined(IREE_ARCH_ARM_32) || defined(IREE_ARCH_ARM_64) || \
    defined(IREE_ARCH_RISCV_32) || defined(IREE_ARCH_RISCV_64)
#define IREE_IO_URING_SYSCALL_SETUP 425
#define IREE_IO_URING_SYSCALL_ENTER 426
#define IREE_IO_URING_SYSCALL_REGISTER 427
#else
#error "io_uring syscall numbers not defined for this architecture"
#endif

// Setup flags (io_uring_params.flags for io_uring_setup).

#define IREE_IORING_SETUP_IOPOLL (1u << 0)      // Busy-poll for completions.
#define IREE_IORING_SETUP_SQPOLL (1u << 1)      // Kernel SQ polling thread.
#define IREE_IORING_SETUP_SQ_AFF (1u << 2)      // Affinity for SQPOLL thread.
#define IREE_IORING_SETUP_CQSIZE (1u << 3)      // Custom CQ size.
#define IREE_IORING_SETUP_CLAMP (1u << 4)       // Clamp ring sizes to max.
#define IREE_IORING_SETUP_ATTACH_WQ (1u << 5)   // Share async backend.
#define IREE_IORING_SETUP_R_DISABLED (1u << 6)  // Start with ring disabled.
#define IREE_IORING_SETUP_SUBMIT_ALL (1u << 7)  // Submit all on enter.
#define IREE_IORING_SETUP_COOP_TASKRUN \
  (1u << 8)  // Cooperative task running (5.19+)
#define IREE_IORING_SETUP_TASKRUN_FLAG (1u << 9)  // Set IORING_SQ_TASKRUN.
#define IREE_IORING_SETUP_SQE128 (1u << 10)       // 128-byte SQEs.
#define IREE_IORING_SETUP_CQE32 (1u << 11)        // 32-byte CQEs.
#define IREE_IORING_SETUP_SINGLE_ISSUER \
  (1u << 12)  // Single task submits (6.0+)
#define IREE_IORING_SETUP_DEFER_TASKRUN (1u << 13)  // Defer task running (6.1+)
#define IREE_IORING_SETUP_NO_MMAP (1u << 14)  // User allocates rings (6.5+)
#define IREE_IORING_SETUP_REGISTERED_FD_ONLY \
  (1u << 15)                                     // Only fixed fds (6.6+)
#define IREE_IORING_SETUP_NO_SQARRAY (1u << 16)  // No SQ array (6.6+)

// Feature flags (io_uring_params.features, read-only from setup).

#define IREE_IORING_FEAT_SINGLE_MMAP (1u << 0)
#define IREE_IORING_FEAT_NODROP (1u << 1)
#define IREE_IORING_FEAT_SUBMIT_STABLE (1u << 2)
#define IREE_IORING_FEAT_RW_CUR_POS (1u << 3)
#define IREE_IORING_FEAT_CUR_PERSONALITY (1u << 4)
#define IREE_IORING_FEAT_FAST_POLL (1u << 5)
#define IREE_IORING_FEAT_POLL_32BITS (1u << 6)
#define IREE_IORING_FEAT_SQPOLL_NONFIXED (1u << 7)
#define IREE_IORING_FEAT_EXT_ARG (1u << 8)
#define IREE_IORING_FEAT_NATIVE_WORKERS (1u << 9)
#define IREE_IORING_FEAT_RSRC_TAGS (1u << 10)
#define IREE_IORING_FEAT_CQE_SKIP (1u << 11)
#define IREE_IORING_FEAT_LINKED_FILE (1u << 12)

//===----------------------------------------------------------------------===//
// Opcodes and operation flags
//===----------------------------------------------------------------------===//

#define IREE_IORING_OP_NOP 0
#define IREE_IORING_OP_READV 1
#define IREE_IORING_OP_WRITEV 2
#define IREE_IORING_OP_FSYNC 3
#define IREE_IORING_OP_READ_FIXED 4
#define IREE_IORING_OP_WRITE_FIXED 5
#define IREE_IORING_OP_POLL_ADD 6
#define IREE_IORING_OP_POLL_REMOVE 7
#define IREE_IORING_OP_SYNC_FILE_RANGE 8
#define IREE_IORING_OP_SENDMSG 9
#define IREE_IORING_OP_RECVMSG 10
#define IREE_IORING_OP_TIMEOUT 11
#define IREE_IORING_OP_TIMEOUT_REMOVE 12
#define IREE_IORING_OP_ACCEPT 13
#define IREE_IORING_OP_ASYNC_CANCEL 14
#define IREE_IORING_OP_LINK_TIMEOUT 15
#define IREE_IORING_OP_CONNECT 16
#define IREE_IORING_OP_FALLOCATE 17
#define IREE_IORING_OP_OPENAT 18
#define IREE_IORING_OP_CLOSE 19
#define IREE_IORING_OP_FILES_UPDATE 20
#define IREE_IORING_OP_STATX 21
#define IREE_IORING_OP_READ 22
#define IREE_IORING_OP_WRITE 23
#define IREE_IORING_OP_FADVISE 24
#define IREE_IORING_OP_MADVISE 25
#define IREE_IORING_OP_SEND 26
#define IREE_IORING_OP_RECV 27
#define IREE_IORING_OP_OPENAT2 28
#define IREE_IORING_OP_EPOLL_CTL 29
#define IREE_IORING_OP_SPLICE 30
#define IREE_IORING_OP_PROVIDE_BUFFERS 31
#define IREE_IORING_OP_REMOVE_BUFFERS 32
#define IREE_IORING_OP_TEE 33
#define IREE_IORING_OP_SHUTDOWN 34
#define IREE_IORING_OP_RENAMEAT 35
#define IREE_IORING_OP_UNLINKAT 36
#define IREE_IORING_OP_MKDIRAT 37
#define IREE_IORING_OP_SYMLINKAT 38
#define IREE_IORING_OP_LINKAT 39
#define IREE_IORING_OP_MSG_RING 40
#define IREE_IORING_OP_FSETXATTR 41
#define IREE_IORING_OP_SETXATTR 42
#define IREE_IORING_OP_FGETXATTR 43
#define IREE_IORING_OP_GETXATTR 44
#define IREE_IORING_OP_SOCKET 45
#define IREE_IORING_OP_URING_CMD 46
#define IREE_IORING_OP_SEND_ZC 47           // Zero-copy send (6.0+)
#define IREE_IORING_OP_SENDMSG_ZC 48        // Zero-copy sendmsg (6.0+)
#define IREE_IORING_OP_READ_MULTISHOT 49    // Multishot read (6.5+)
#define IREE_IORING_OP_WAITID 50            // waitid (6.5+)
#define IREE_IORING_OP_FUTEX_WAIT 51        // Futex wait (6.7+)
#define IREE_IORING_OP_FUTEX_WAKE 52        // Futex wake (6.7+)
#define IREE_IORING_OP_FUTEX_WAITV 53       // Multi-futex wait (6.7+)
#define IREE_IORING_OP_FIXED_FD_INSTALL 54  // Install fixed fd (6.8+)

// Futex operation flags (matches kernel FUTEX2_* flags).
#define IREE_FUTEX2_SIZE_U8 0x00
#define IREE_FUTEX2_SIZE_U16 0x01
#define IREE_FUTEX2_SIZE_U32 0x02
#define IREE_FUTEX2_SIZE_U64 0x03
#define IREE_FUTEX2_NUMA 0x04
#define IREE_FUTEX2_PRIVATE IREE_FUTEX_PRIVATE_FLAG

// Private flag - futex is process-private (faster).
#define IREE_FUTEX_PRIVATE_FLAG 128

// MSG_RING command types (stored in sqe->addr, mutually exclusive):
// Post arbitrary data to target ring (kernel 5.18+).
// sqe->len becomes target cqe->res, sqe->off becomes target cqe->user_data.
#define IREE_IORING_MSG_DATA 0

// Send a registered file descriptor to target ring.
#define IREE_IORING_MSG_SEND_FD 1

// MSG_RING flags (stored in sqe->msg_ring_flags, separate from command type):
// Skip generating a CQE on the source ring (kernel 5.19+).
// Only the target ring receives a CQE for the message.
#define IREE_IORING_MSG_RING_CQE_SKIP (1u << 0)

// Pass flags from sqe->file_index to the target CQE's flags field
// (kernel 6.3+). Allows setting CQE flags like IORING_CQE_F_MORE on the target
// CQE.
#define IREE_IORING_MSG_RING_FLAGS_PASS (1u << 1)

// SQE flags (io_uring_sqe.flags).

#define IREE_IOSQE_FIXED_FILE (1u << 0)   // Use fixed file index.
#define IREE_IOSQE_IO_DRAIN (1u << 1)     // Wait for prior SQEs.
#define IREE_IOSQE_IO_LINK (1u << 2)      // Link to next SQE.
#define IREE_IOSQE_IO_HARDLINK (1u << 3)  // Hard link (no short-circuit errors)
#define IREE_IOSQE_ASYNC (1u << 4)        // Force async execution.
#define IREE_IOSQE_BUFFER_SELECT (1u << 5)     // Select buffer from group.
#define IREE_IOSQE_CQE_SKIP_SUCCESS (1u << 6)  // Skip CQE on success.

// io_uring_enter flags.

#define IREE_IORING_ENTER_GETEVENTS (1u << 0)
#define IREE_IORING_ENTER_SQ_WAKEUP (1u << 1)
#define IREE_IORING_ENTER_SQ_WAIT (1u << 2)
#define IREE_IORING_ENTER_EXT_ARG (1u << 3)
#define IREE_IORING_ENTER_REGISTERED_RING (1u << 4)

// SQ ring flags (read from mapped sq_ring->flags).

// Set by kernel when SQPOLL thread needs wakeup via IORING_ENTER_SQ_WAKEUP.
#define IREE_IORING_SQ_NEED_WAKEUP (1u << 0)

// Set by kernel when CQ ring has overflowed.
#define IREE_IORING_SQ_CQ_OVERFLOW (1u << 1)

// Set by kernel when task work is pending (IORING_SETUP_TASKRUN_FLAG).
#define IREE_IORING_SQ_TASKRUN (1u << 2)

// CQ ring flags (read from mapped cq_ring->flags).

// Set by user to disable eventfd notifications.
#define IREE_IORING_CQ_EVENTFD_DISABLED (1u << 0)

// CQE flags (io_uring_cqe.flags).

#define IREE_IORING_CQE_F_BUFFER (1u << 0)  // Buffer ID in upper 16 bits.
#define IREE_IORING_CQE_F_MORE (1u << 1)    // More CQEs coming for this SQE.
#define IREE_IORING_CQE_F_SOCK_NONEMPTY (1u << 2)  // Socket has more data.
#define IREE_IORING_CQE_F_NOTIF (1u << 3)          // ZC notification CQE.

#define IREE_IORING_CQE_BUFFER_SHIFT 16

// ZC notification CQE res value when REPORT_USAGE is set and kernel fell back
// to copying instead of achieving true zero-copy. When res == 0, ZC succeeded.
#define IREE_IORING_NOTIF_USAGE_ZC_COPIED (1u << 31)

// Multishot flags (in sqe.ioprio for specific opcodes).

// For IORING_OP_ACCEPT: enable multishot mode (kernel 5.19+).
// Single SQE submission produces multiple CQEs (one per accepted connection).
// Each CQE has CQE_F_MORE set until the operation terminates.
#define IREE_IORING_ACCEPT_MULTISHOT (1u << 0)

// For IORING_OP_RECV/RECVMSG: enable multishot mode (kernel 5.19+).
// Single SQE submission produces multiple CQEs (one per received message).
// Each CQE has CQE_F_MORE set until the connection closes or an error occurs.
#define IREE_IORING_RECV_MULTISHOT (1u << 1)

// Additional recv/send ioprio flags.
#define IREE_IORING_RECVSEND_POLL_FIRST (1u << 0)   // Poll before attempting.
#define IREE_IORING_RECVSEND_FIXED_BUF (1u << 2)    // Use fixed buffer.
#define IREE_IORING_SEND_ZC_REPORT_USAGE (1u << 3)  // Report zero-copy usage.

// Poll flags (for IORING_OP_POLL_ADD sqe.len field).

#define IREE_IORING_POLL_ADD_MULTI (1u << 0)         // Multishot poll.
#define IREE_IORING_POLL_UPDATE_EVENTS (1u << 1)     // Update poll events.
#define IREE_IORING_POLL_UPDATE_USER_DATA (1u << 2)  // Update user_data.
#define IREE_IORING_POLL_ADD_LEVEL (1u << 3)         // Level-triggered poll.

// Async cancel flags (for IORING_OP_ASYNC_CANCEL).

#define IREE_IORING_ASYNC_CANCEL_ALL (1u << 0)       // Cancel all matching.
#define IREE_IORING_ASYNC_CANCEL_FD (1u << 1)        // Match by fd.
#define IREE_IORING_ASYNC_CANCEL_ANY (1u << 2)       // Cancel any request.
#define IREE_IORING_ASYNC_CANCEL_FD_FIXED (1u << 3)  // fd is fixed file index.
#define IREE_IORING_ASYNC_CANCEL_USERDATA (1u << 4)  // Match by user_data.
#define IREE_IORING_ASYNC_CANCEL_OP (1u << 5)        // Match by opcode.

// Standard poll bits from <poll.h>.
#define IREE_POLLIN 0x0001
#define IREE_POLLOUT 0x0004
#define IREE_POLLERR 0x0008
#define IREE_POLLHUP 0x0010

// MSG_MORE: hint that more data is coming. Kernel corks the TCP stream,
// coalescing small writes into larger packets (Nagle-like behavior on demand).
#define IREE_MSG_MORE 0x8000

// Register opcodes (for io_uring_register).

#define IREE_IORING_REGISTER_BUFFERS 0
#define IREE_IORING_UNREGISTER_BUFFERS 1
#define IREE_IORING_REGISTER_FILES 2
#define IREE_IORING_UNREGISTER_FILES 3
#define IREE_IORING_REGISTER_EVENTFD 4
#define IREE_IORING_UNREGISTER_EVENTFD 5
#define IREE_IORING_REGISTER_FILES_UPDATE 6
#define IREE_IORING_REGISTER_EVENTFD_ASYNC 7
#define IREE_IORING_REGISTER_PROBE 8
#define IREE_IORING_REGISTER_PERSONALITY 9
#define IREE_IORING_UNREGISTER_PERSONALITY 10
#define IREE_IORING_REGISTER_RESTRICTIONS 11
#define IREE_IORING_REGISTER_ENABLE_RINGS 12
#define IREE_IORING_REGISTER_FILES2 13
#define IREE_IORING_REGISTER_FILES_UPDATE2 14
#define IREE_IORING_REGISTER_BUFFERS2 15
#define IREE_IORING_REGISTER_BUFFERS_UPDATE 16
#define IREE_IORING_REGISTER_IOWQ_AFF 17
#define IREE_IORING_UNREGISTER_IOWQ_AFF 18
#define IREE_IORING_REGISTER_IOWQ_MAX_WORKERS 19
#define IREE_IORING_REGISTER_RING_FDS 20
#define IREE_IORING_UNREGISTER_RING_FDS 21
#define IREE_IORING_REGISTER_PBUF_RING 22
#define IREE_IORING_UNREGISTER_PBUF_RING 23
#define IREE_IORING_REGISTER_SYNC_CANCEL 24
#define IREE_IORING_REGISTER_FILE_ALLOC_RANGE 25

// Sparse resource table flag for IORING_REGISTER_BUFFERS2/FILES2.
// When set, creates an empty table of the specified size. Slots are populated
// later via IORING_REGISTER_BUFFERS_UPDATE/FILES_UPDATE2. Available since
// kernel 5.19.
#define IREE_IORING_RSRC_REGISTER_SPARSE (1U << 0)

// Registration argument for IORING_REGISTER_BUFFERS2 and
// IORING_REGISTER_FILES2. Passed via
// io_uring_register(ring_fd, opcode, &arg, sizeof(arg)).
typedef struct iree_io_uring_rsrc_register {
  // Number of entries (table size when used with SPARSE flag).
  uint32_t nr;
  // IREE_IORING_RSRC_REGISTER_SPARSE, etc.
  uint32_t flags;
  uint64_t resv2;  // Must be 0.
  // Pointer to iovec/fd array (0 for sparse empty table).
  uint64_t data;
  // Pointer to tags array (0 for sparse empty table).
  uint64_t tags;
} iree_io_uring_rsrc_register_t;

// Update argument for IORING_REGISTER_BUFFERS_UPDATE and
// IORING_REGISTER_FILES_UPDATE2. Populates or clears individual slots in a
// sparse resource table.
typedef struct iree_io_uring_rsrc_update2 {
  // Starting slot index in the table.
  uint32_t offset;
  uint32_t resv;  // Must be 0.
  // Pointer to iovec/fd array with new entries.
  uint64_t data;
  // Pointer to tags array (0 for no tags).
  uint64_t tags;
  // Number of entries to update.
  uint32_t nr;
  uint32_t resv2;  // Must be 0.
} iree_io_uring_rsrc_update2_t;

//===----------------------------------------------------------------------===//
// Ring structures
//===----------------------------------------------------------------------===//

// Maximum SQ entries the kernel will accept (IORING_MAX_ENTRIES).
// See linux/io_uring/io_uring.c::IORING_MAX_ENTRIES.
#define IREE_IORING_MAX_ENTRIES 32768

// mmap offsets.

#define IREE_IORING_OFF_SQ_RING 0ULL
#define IREE_IORING_OFF_CQ_RING 0x8000000ULL
#define IREE_IORING_OFF_SQES 0x10000000ULL

// Offsets into the SQ ring mmap.
typedef struct iree_io_sqring_offsets {
  uint32_t head;
  uint32_t tail;
  uint32_t ring_mask;
  uint32_t ring_entries;
  uint32_t flags;
  uint32_t dropped;
  uint32_t array;
  uint32_t resv1;
  uint64_t user_addr;  // For IORING_SETUP_NO_MMAP rings.
} iree_io_sqring_offsets_t;

// Offsets into the CQ ring mmap.
typedef struct iree_io_cqring_offsets {
  uint32_t head;
  uint32_t tail;
  uint32_t ring_mask;
  uint32_t ring_entries;
  uint32_t overflow;
  uint32_t cqes;
  uint32_t flags;
  uint32_t resv1;
  uint64_t user_addr;  // For IORING_SETUP_NO_MMAP rings.
} iree_io_cqring_offsets_t;

typedef struct iree_io_uring_params {
  uint32_t sq_entries;
  uint32_t cq_entries;
  uint32_t flags;
  uint32_t sq_thread_cpu;
  uint32_t sq_thread_idle;
  uint32_t features;
  uint32_t wq_fd;
  uint32_t resv[3];
  iree_io_sqring_offsets_t sq_off;
  iree_io_cqring_offsets_t cq_off;
} iree_io_uring_params_t;

// Submission Queue Entry (SQE) - 64 bytes.
typedef struct iree_io_uring_sqe {
  uint8_t opcode;  // Operation type (IORING_OP_*).
  uint8_t flags;   // SQE flags (IOSQE_*).
  // I/O priority or request-specific flags.
  uint16_t ioprio;
  // File descriptor (or fixed file index).
  int32_t fd;

  union {
    uint64_t off;    // Offset for read/write ops.
    uint64_t addr2;  // Secondary address.
    struct {
      uint32_t cmd_op;
      uint32_t __pad1;
    };
  };

  union {
    uint64_t addr;           // Buffer address.
    uint64_t splice_off_in;  // Splice input offset.
    struct {
      // Socket option level (for SOCKET cmd).
      uint32_t level;
      // Socket option name (for SOCKET cmd).
      uint32_t optname;
    };
  };

  uint32_t len;  // Buffer length or count.

  union {
    uint32_t rw_flags;     // Read/write flags.
    uint32_t fsync_flags;  // Fsync flags.
    // Poll events (16 bits used).
    uint16_t poll_events;
    uint32_t poll32_events;     // Poll events (32-bit).
    uint32_t sync_range_flags;  // Sync range flags.
    uint32_t msg_flags;         // Send/recv message flags.
    uint32_t timeout_flags;     // Timeout flags.
    uint32_t accept_flags;      // Accept flags.
    uint32_t cancel_flags;      // Cancel flags.
    uint32_t open_flags;        // Open flags.
    uint32_t statx_flags;       // Statx flags.
    uint32_t fadvise_advice;    // Fadvise advice.
    uint32_t splice_flags;      // Splice flags.
    uint32_t rename_flags;      // Rename flags.
    uint32_t unlink_flags;      // Unlink flags.
    uint32_t hardlink_flags;    // Hardlink flags.
    uint32_t xattr_flags;       // Xattr flags.
    uint32_t msg_ring_flags;    // Message ring flags.
    uint32_t uring_cmd_flags;   // Uring cmd flags.
    uint32_t waitid_flags;      // Waitid flags.
    uint32_t futex_flags;       // Futex flags (FUTEX2_*).
    uint32_t install_fd_flags;  // Fixed FD install flags.
  };

  // Returned in CQE for correlation.
  uint64_t user_data;

  union {
    uint16_t buf_index;  // Fixed buffer index.
    uint16_t buf_group;  // Buffer group ID.
  };

  uint16_t personality;  // Credentials index.

  union {
    int32_t splice_fd_in;  // Splice source fd.
    // Fixed file index for direct ops.
    uint32_t file_index;
    // Socket option length (for SOCKET cmd).
    uint32_t optlen;
    struct {
      uint16_t addr_len;
      uint16_t __pad3[1];
    };
  };

  union {
    struct {
      uint64_t addr3;
      uint64_t __pad2[1];
    };
    // Socket option value pointer (for SOCKET cmd).
    uint64_t optval;
    // Uring command data (variable length).
    uint8_t cmd[0];
  };
} iree_io_uring_sqe_t;

// Completion Queue Entry (CQE) - 16 bytes (32 bytes with CQE32).
typedef struct iree_io_uring_cqe {
  // Copied from SQE for correlation.
  uint64_t user_data;
  // Result: bytes transferred or -errno.
  int32_t res;
  uint32_t flags;  // CQE flags (IORING_CQE_F_*).
  // For CQE32 mode, 16 bytes of extra data follow.
} iree_io_uring_cqe_t;

// Probe structures (for IORING_REGISTER_PROBE).

#define IREE_IO_URING_OP_SUPPORTED (1u << 0)

typedef struct iree_io_uring_probe_op {
  uint8_t op;
  uint8_t resv;
  uint16_t flags;  // IO_URING_OP_SUPPORTED if available.
  uint32_t resv2;
} iree_io_uring_probe_op_t;

typedef struct iree_io_uring_probe {
  uint8_t last_op;  // Highest opcode supported.
  uint8_t ops_len;  // Length of ops array.
  uint16_t resv;
  uint32_t resv2[3];
  struct iree_io_uring_probe_op ops[];  // Flexible array.
} iree_io_uring_probe_t;

// Timeout flags (for IORING_OP_TIMEOUT).

// Interpret the timespec as an absolute deadline (vs. relative duration).
// Available since kernel 5.4. Uses CLOCK_MONOTONIC by default.
#define IREE_IORING_TIMEOUT_ABS (1u << 0)

// Update existing timeout instead of adding new one. Since kernel 5.11.
#define IREE_IORING_TIMEOUT_UPDATE (1u << 1)

// Use CLOCK_BOOTTIME instead of CLOCK_MONOTONIC. Available since kernel 5.11.
#define IREE_IORING_TIMEOUT_BOOTTIME (1u << 2)

// Use CLOCK_REALTIME instead of the default CLOCK_MONOTONIC for timeout
// interpretation. Available since kernel 5.11. The default CLOCK_MONOTONIC
// matches iree_time_now() — only use REALTIME when the deadline source is
// wall-clock time (e.g., from gettimeofday).
#define IREE_IORING_TIMEOUT_REALTIME (1u << 3)

// Update existing linked timeout. Since kernel 5.15.
#define IREE_IORING_LINK_TIMEOUT_UPDATE (1u << 4)

// Treat -ETIME as success. Since kernel 5.19.
#define IREE_IORING_TIMEOUT_ETIME_SUCCESS (1u << 5)

// Multishot timeout. Since kernel 6.0.
#define IREE_IORING_TIMEOUT_MULTISHOT (1u << 6)

// Clock source masks.
#define IREE_IORING_TIMEOUT_CLOCK_MASK \
  (IREE_IORING_TIMEOUT_BOOTTIME | IREE_IORING_TIMEOUT_REALTIME)
#define IREE_IORING_TIMEOUT_UPDATE_MASK \
  (IREE_IORING_TIMEOUT_UPDATE | IREE_IORING_LINK_TIMEOUT_UPDATE)

// Kernel timespec (for timeout operations).
typedef struct iree_kernel_timespec {
  int64_t tv_sec;
  int64_t tv_nsec;
} iree_kernel_timespec_t;

// Extended argument for io_uring_enter (with timeout).
typedef struct iree_io_uring_getevents_arg {
  uint64_t sigmask;
  uint32_t sigmask_sz;
  uint32_t pad;
  uint64_t ts;  // Pointer to iree_kernel_timespec.
} iree_io_uring_getevents_arg_t;

//===----------------------------------------------------------------------===//
// Provided buffers and buffer ring
//===----------------------------------------------------------------------===//

// Registration argument for IORING_REGISTER_PBUF_RING.
// The ring_addr must be page-aligned memory of size (ring_entries * 16).
typedef struct iree_io_uring_buf_reg {
  uint64_t ring_addr;  // User-allocated ring address (page-aligned).
  // Number of entries (power of 2, max 32768).
  uint32_t ring_entries;
  uint16_t bgid;  // Buffer group ID.
  uint16_t pad;
  uint64_t resv[3];
} iree_io_uring_buf_reg_t;

// A single buffer entry in a provided buffer ring.
// When adding buffers to the ring, userspace fills these structures.
// When a receive completes, the CQE flags indicate which buffer was used.
typedef struct iree_io_uring_buf {
  uint64_t addr;  // Buffer address.
  uint32_t len;   // Buffer length.
  // Buffer ID (returned in CQE).
  uint16_t bid;
  uint16_t resv;
} iree_io_uring_buf_t;

// Provided buffer ring header. The ring is laid out as:
//   [0]: iree_io_uring_buf_ring_t (this struct)
//   [1..ring_entries]: iree_io_uring_buf_t entries
//
// Userspace maintains 'tail' and advances it when adding buffers.
// Kernel consumes buffers and implicitly advances the head (not visible).
// The ring_entries must be a power of 2, max 32768.
typedef struct iree_io_uring_buf_ring {
  union {
    struct {
      uint64_t resv1;
      uint32_t resv2;
      uint16_t resv3;
      uint16_t tail;  // Userspace-maintained tail index.
    };
    // The first entry slot overlaps the header (union layout).
    iree_io_uring_buf_t bufs[0];
  };
} iree_io_uring_buf_ring_t;

// Flags for IORING_REGISTER_PBUF_RING (in buf_reg.flags if extended).
// IOU_PBUF_RING_MMAP: kernel allocates the ring memory (kernel 6.4+).
#define IREE_IOU_PBUF_RING_MMAP (1u << 0)

// Maximum entries for a provided buffer ring.
#define IREE_IO_URING_MAX_PBUF_RING_ENTRIES (1u << 15)

//===----------------------------------------------------------------------===//
// Helper macros and utilities
//===----------------------------------------------------------------------===//

// Atomic store/load with release/acquire semantics for kernel-shared memory.
// These match liburing's io_uring_smp_store_release /
// io_uring_smp_load_acquire. The compiler infers the correct size from the
// pointer type.
//
// Release semantics on store ensures prior writes (e.g., buffer entry data)
// are visible to the kernel before it sees the updated index.

#define iree_io_uring_store_release(ptr, value) \
  __atomic_store_n((ptr), (value), __ATOMIC_RELEASE)

#define iree_io_uring_load_acquire(ptr) __atomic_load_n((ptr), __ATOMIC_ACQUIRE)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_IO_URING_DEFS_H_
