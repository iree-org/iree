// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Windows handshake handle exchange: passes HANDLEs across processes using
// DuplicateHandle over overlapped ReadFile/WriteFile on a named pipe.
//
// Each handshake message consists of:
//   1. The fixed-size header (same as POSIX) written as pipe data.
//   2. A Windows-specific payload: sender PID + raw HANDLE values.
//
// The receiver uses the sender's PID to open their process via OpenProcess
// with PROCESS_DUP_HANDLE access, then calls DuplicateHandle for each handle
// to copy it into the receiver's address space. This works for any handle type
// (file mappings, Events, etc.) without requiring named objects.
//
// Named pipes opened with FILE_FLAG_OVERLAPPED require overlapped I/O for
// ReadFile/WriteFile. Each function creates a manual-reset event for the
// OVERLAPPED structure and uses WaitForSingleObject for completion/timeout.

#include "iree/net/carrier/shm/handshake.h"

#if defined(IREE_PLATFORM_WINDOWS)

#include <string.h>
#include <windows.h>

// Maximum number of handles sent in a single handshake message.
// OFFER sends 3 (shm_region, wake_epoch_shm, signal_primitive).
// ACCEPT sends 2 (wake_epoch_shm, signal_primitive).
#define MAX_HANDSHAKE_HANDLES 3

// Windows-specific payload appended after the standard header on the wire.
// Contains the sender's PID so the receiver can open the sender's process for
// DuplicateHandle, plus the raw HANDLE values in the sender's address space.
typedef struct iree_net_shm_handshake_win32_payload_t {
  uint32_t sender_pid;
  uint32_t handle_count;
  uint64_t handles[MAX_HANDSHAKE_HANDLES];
} iree_net_shm_handshake_win32_payload_t;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Extracts the raw uint64 value from an iree_shm_handle_t.
// Returns 0 for invalid handles.
static uint64_t iree_shm_handle_to_uint64(iree_shm_handle_t handle) {
  if (!iree_shm_handle_is_valid(handle)) return 0;
  return handle.value;
}

static iree_shm_handle_t iree_shm_handle_from_uint64(uint64_t value) {
  iree_shm_handle_t handle;
  handle.value = value;
  return handle;
}

// Extracts the raw uint64 value from an iree_async_primitive_t.
// Returns 0 for NONE primitives or non-WIN32_HANDLE types.
static uint64_t iree_async_primitive_to_uint64(
    iree_async_primitive_t primitive) {
  if (primitive.type != IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE) return 0;
  return (uint64_t)primitive.value.win32_handle;
}

//===----------------------------------------------------------------------===//
// Overlapped pipe I/O helpers
//===----------------------------------------------------------------------===//

// Writes exactly |length| bytes to the pipe using overlapped WriteFile.
// |event| is a manual-reset event for the OVERLAPPED structure (caller-owned,
// reused across calls).
static iree_status_t iree_net_shm_handshake_win32_send_all(HANDLE channel,
                                                           HANDLE event,
                                                           const void* data,
                                                           DWORD length) {
  const char* cursor = (const char*)data;
  DWORD remaining = length;
  while (remaining > 0) {
    OVERLAPPED overlapped;
    memset(&overlapped, 0, sizeof(overlapped));
    overlapped.hEvent = event;
    ResetEvent(event);

    DWORD written = 0;
    if (!WriteFile(channel, cursor, remaining, &written, &overlapped)) {
      DWORD error = GetLastError();
      if (error == ERROR_IO_PENDING) {
        DWORD wait_result = WaitForSingleObject(event, INFINITE);
        if (wait_result != WAIT_OBJECT_0) {
          return iree_make_status(
              iree_status_code_from_win32_error(GetLastError()),
              "handshake send WaitForSingleObject failed");
        }
        if (!GetOverlappedResult(channel, &overlapped, &written, FALSE)) {
          return iree_make_status(
              iree_status_code_from_win32_error(GetLastError()),
              "handshake send GetOverlappedResult failed");
        }
      } else {
        return iree_make_status(iree_status_code_from_win32_error(error),
                                "handshake WriteFile failed");
      }
    }
    cursor += written;
    remaining -= written;
  }
  return iree_ok_status();
}

// Reads exactly |length| bytes from the pipe using overlapped ReadFile.
// |event| is a manual-reset event for the OVERLAPPED structure (caller-owned,
// reused across calls). Uses |timeout_ms| for WaitForSingleObject; on timeout,
// cancels the I/O and returns DEADLINE_EXCEEDED.
static iree_status_t iree_net_shm_handshake_win32_recv_all(
    HANDLE channel, HANDLE event, void* data, DWORD length, DWORD timeout_ms) {
  char* cursor = (char*)data;
  DWORD remaining = length;
  while (remaining > 0) {
    OVERLAPPED overlapped;
    memset(&overlapped, 0, sizeof(overlapped));
    overlapped.hEvent = event;
    ResetEvent(event);

    DWORD bytes_read = 0;
    if (!ReadFile(channel, cursor, remaining, &bytes_read, &overlapped)) {
      DWORD error = GetLastError();
      if (error == ERROR_IO_PENDING) {
        DWORD wait_result = WaitForSingleObject(event, timeout_ms);
        if (wait_result == WAIT_TIMEOUT) {
          CancelIoEx(channel, &overlapped);
          // Wait for the cancellation to complete so the OVERLAPPED is safe to
          // go out of scope.
          WaitForSingleObject(event, INFINITE);
          return iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED,
                                  "handshake timed out during recv");
        }
        if (wait_result != WAIT_OBJECT_0) {
          return iree_make_status(
              iree_status_code_from_win32_error(GetLastError()),
              "handshake recv WaitForSingleObject failed");
        }
        if (!GetOverlappedResult(channel, &overlapped, &bytes_read, FALSE)) {
          return iree_make_status(
              iree_status_code_from_win32_error(GetLastError()),
              "handshake recv GetOverlappedResult failed");
        }
      } else if (error == ERROR_BROKEN_PIPE) {
        return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "handshake peer disconnected during recv "
                                "(%lu of %lu bytes received)",
                                (unsigned long)(length - remaining),
                                (unsigned long)length);
      } else {
        return iree_make_status(iree_status_code_from_win32_error(error),
                                "handshake ReadFile failed");
      }
    }
    if (bytes_read == 0) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "handshake peer disconnected during recv "
                              "(%lu of %lu bytes received)",
                              (unsigned long)(length - remaining),
                              (unsigned long)length);
    }
    cursor += bytes_read;
    remaining -= bytes_read;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Send/recv with DuplicateHandle
//===----------------------------------------------------------------------===//

iree_status_t iree_net_shm_handshake_send(
    iree_async_primitive_t channel,
    const iree_net_shm_handshake_header_t* header,
    const iree_net_shm_handshake_handles_t* handles) {
  if (channel.type != IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "handshake channel is not a valid Windows handle");
  }
  HANDLE channel_handle = (HANDLE)channel.value.win32_handle;

  // Build the Windows-specific payload with our PID and raw handle values.
  // The receiver will use our PID to call DuplicateHandle across processes.
  iree_net_shm_handshake_win32_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  payload.sender_pid = GetCurrentProcessId();
  payload.handle_count = 0;

  // Collect handles in the same order as POSIX: shm_region, epoch, signal.
  uint64_t shm_raw = iree_shm_handle_to_uint64(handles->shm_region);
  if (shm_raw != 0) {
    payload.handles[payload.handle_count++] = shm_raw;
  }
  uint64_t epoch_raw = iree_shm_handle_to_uint64(handles->wake_epoch_shm);
  if (epoch_raw != 0) {
    payload.handles[payload.handle_count++] = epoch_raw;
  }
  uint64_t signal_raw =
      iree_async_primitive_to_uint64(handles->signal_primitive);
  if (signal_raw != 0) {
    payload.handles[payload.handle_count++] = signal_raw;
  }

  // Create event for overlapped I/O.
  HANDLE event = CreateEventW(NULL, /*bManualReset=*/TRUE,
                              /*bInitialState=*/FALSE, NULL);
  if (!event) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "handshake send CreateEvent failed");
  }

  // Send header, then payload.
  iree_status_t status = iree_net_shm_handshake_win32_send_all(
      channel_handle, event, header, (DWORD)sizeof(*header));
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_handshake_win32_send_all(
        channel_handle, event, &payload, (DWORD)sizeof(payload));
  }

  CloseHandle(event);
  return status;
}

iree_status_t iree_net_shm_handshake_recv(
    iree_async_primitive_t channel, iree_net_shm_handshake_header_t* out_header,
    iree_net_shm_handshake_handles_t* out_handles) {
  if (channel.type != IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "handshake channel is not a valid Windows handle");
  }
  HANDLE channel_handle = (HANDLE)channel.value.win32_handle;

  memset(out_header, 0, sizeof(*out_header));
  memset(out_handles, 0, sizeof(*out_handles));
  out_handles->shm_region = IREE_SHM_HANDLE_INVALID;
  out_handles->wake_epoch_shm = IREE_SHM_HANDLE_INVALID;

  // Create event for overlapped I/O.
  HANDLE event = CreateEventW(NULL, /*bManualReset=*/TRUE,
                              /*bInitialState=*/FALSE, NULL);
  if (!event) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "handshake recv CreateEvent failed");
  }

  // Receive the header (with timeout for the initial wait).
  iree_status_t status = iree_net_shm_handshake_win32_recv_all(
      channel_handle, event, out_header, (DWORD)sizeof(*out_header),
      IREE_NET_SHM_HANDSHAKE_TIMEOUT_MS);

  // Receive the Windows-specific payload (data should be available
  // immediately after the header, but use the same timeout for safety).
  iree_net_shm_handshake_win32_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_handshake_win32_recv_all(
        channel_handle, event, &payload, (DWORD)sizeof(payload),
        IREE_NET_SHM_HANDSHAKE_TIMEOUT_MS);
  }

  CloseHandle(event);

  if (!iree_status_is_ok(status)) return status;

  // Validate the handle count.
  if (payload.handle_count > MAX_HANDSHAKE_HANDLES) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "handshake payload has invalid handle count: %u",
                            payload.handle_count);
  }

  // Open the sender's process to duplicate handles into ours.
  // PROCESS_DUP_HANDLE is sufficient — no need for broader access.
  HANDLE sender_process =
      OpenProcess(PROCESS_DUP_HANDLE, FALSE, payload.sender_pid);
  if (!sender_process) {
    return iree_make_status(
        iree_status_code_from_win32_error(GetLastError()),
        "OpenProcess(%lu) failed — cannot duplicate peer handles "
        "(are both processes running as the same user?)",
        (unsigned long)payload.sender_pid);
  }

  // Duplicate each handle from the sender's process into ours.
  HANDLE local_handles[MAX_HANDSHAKE_HANDLES];
  memset(local_handles, 0, sizeof(local_handles));
  for (uint32_t i = 0; i < payload.handle_count; ++i) {
    HANDLE source = (HANDLE)(uintptr_t)payload.handles[i];
    if (!DuplicateHandle(sender_process, source, GetCurrentProcess(),
                         &local_handles[i], 0, FALSE, DUPLICATE_SAME_ACCESS)) {
      // Close any handles we already duplicated.
      for (uint32_t j = 0; j < i; ++j) {
        CloseHandle(local_handles[j]);
      }
      DWORD error = GetLastError();
      CloseHandle(sender_process);
      return iree_make_status(
          iree_status_code_from_win32_error(error),
          "DuplicateHandle failed for handle %u (source=0x%llx from pid %lu)",
          i, (unsigned long long)payload.handles[i],
          (unsigned long)payload.sender_pid);
    }
  }
  CloseHandle(sender_process);

  // Unpack handles based on message type.
  // OFFER: 3 handles (shm_region, wake_epoch_shm, signal_primitive).
  // ACCEPT: 2 handles (wake_epoch_shm, signal_primitive).
  if (out_header->type == IREE_NET_SHM_HANDSHAKE_MESSAGE_OFFER) {
    if (payload.handle_count != 3) {
      for (uint32_t i = 0; i < payload.handle_count; ++i) {
        CloseHandle(local_handles[i]);
      }
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "OFFER expected 3 handles, got %u",
                              payload.handle_count);
    }
    out_handles->shm_region =
        iree_shm_handle_from_uint64((uint64_t)(uintptr_t)local_handles[0]);
    out_handles->wake_epoch_shm =
        iree_shm_handle_from_uint64((uint64_t)(uintptr_t)local_handles[1]);
    out_handles->signal_primitive =
        iree_async_primitive_from_win32_handle((uintptr_t)local_handles[2]);
  } else if (out_header->type == IREE_NET_SHM_HANDSHAKE_MESSAGE_ACCEPT) {
    if (payload.handle_count != 2) {
      for (uint32_t i = 0; i < payload.handle_count; ++i) {
        CloseHandle(local_handles[i]);
      }
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "ACCEPT expected 2 handles, got %u",
                              payload.handle_count);
    }
    out_handles->wake_epoch_shm =
        iree_shm_handle_from_uint64((uint64_t)(uintptr_t)local_handles[0]);
    out_handles->signal_primitive =
        iree_async_primitive_from_win32_handle((uintptr_t)local_handles[1]);
  } else {
    // Unknown message type — close any received handles.
    for (uint32_t i = 0; i < payload.handle_count; ++i) {
      CloseHandle(local_handles[i]);
    }
  }

  return iree_ok_status();
}

void iree_net_shm_handshake_handles_close(
    iree_net_shm_handshake_handles_t* handles) {
  if (!handles) return;
  iree_shm_handle_close(&handles->shm_region);
  iree_shm_handle_close(&handles->wake_epoch_shm);
  iree_async_primitive_close(&handles->signal_primitive);
}

#endif  // IREE_PLATFORM_WINDOWS
