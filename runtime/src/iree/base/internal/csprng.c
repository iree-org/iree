// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/csprng.h"

//===----------------------------------------------------------------------===//
// Platform CSPRNG implementations
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_WINDOWS)

#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")

IREE_API_EXPORT iree_status_t iree_csprng_fill(iree_byte_span_t buffer) {
  if (buffer.data_length == 0) return iree_ok_status();
  NTSTATUS status =
      BCryptGenRandom(NULL, buffer.data, (ULONG)buffer.data_length,
                      BCRYPT_USE_SYSTEM_PREFERRED_RNG);
  if (!BCRYPT_SUCCESS(status)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "BCryptGenRandom failed: 0x%08lX", status);
  }
  return iree_ok_status();
}

#elif defined(IREE_PLATFORM_APPLE) || defined(IREE_PLATFORM_BSD)

#include <stdlib.h>

IREE_API_EXPORT iree_status_t iree_csprng_fill(iree_byte_span_t buffer) {
  if (buffer.data_length == 0) return iree_ok_status();
  arc4random_buf(buffer.data, buffer.data_length);
  return iree_ok_status();
}

#elif defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)

#include <errno.h>
#include <sys/random.h>

IREE_API_EXPORT iree_status_t iree_csprng_fill(iree_byte_span_t buffer) {
  if (buffer.data_length == 0) return iree_ok_status();
  iree_host_size_t total_read = 0;
  while (total_read < buffer.data_length) {
    ssize_t bytes_read =
        getrandom(buffer.data + total_read, buffer.data_length - total_read, 0);
    if (bytes_read < 0) {
      if (errno == EINTR) continue;
      return iree_make_status(iree_status_code_from_errno(errno),
                              "getrandom failed");
    }
    total_read += (iree_host_size_t)bytes_read;
  }
  return iree_ok_status();
}

#elif defined(IREE_PLATFORM_WASI)

// WASI provides random_get as a host syscall (wasi_snapshot_preview1).
extern int32_t __wasi_random_get(uint8_t* buffer, uint32_t length)
    __attribute__((__import_module__("wasi_snapshot_preview1"),
                   __import_name__("random_get")));

IREE_API_EXPORT iree_status_t iree_csprng_fill(iree_byte_span_t buffer) {
  if (buffer.data_length == 0) return iree_ok_status();
  int32_t error = __wasi_random_get(buffer.data, (uint32_t)buffer.data_length);
  if (error != 0) {
    return iree_make_status(IREE_STATUS_INTERNAL, "wasi random_get failed: %d",
                            error);
  }
  return iree_ok_status();
}

#elif defined(IREE_PLATFORM_WEB)

// Provided by JS host via Wasm import (see csprng.js).
// Fills buffer with crypto.getRandomValues(), handling the 65536-byte
// per-call browser limit internally. Returns 0 on success.
extern int iree_wasm_csprng_fill(uint8_t* buffer, uint32_t length);

IREE_API_EXPORT iree_status_t iree_csprng_fill(iree_byte_span_t buffer) {
  if (buffer.data_length == 0) return iree_ok_status();
  int result = iree_wasm_csprng_fill(buffer.data, (uint32_t)buffer.data_length);
  if (result != 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "crypto.getRandomValues failed");
  }
  return iree_ok_status();
}

#else

#error "CSPRNG not implemented for this platform"

#endif  // IREE_PLATFORM_*
