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

#elif defined(IREE_PLATFORM_EMSCRIPTEN)

#include <emscripten.h>

IREE_API_EXPORT iree_status_t iree_csprng_fill(iree_byte_span_t buffer) {
  if (buffer.data_length == 0) return iree_ok_status();
  // crypto.getRandomValues has a max size of 65536 bytes.
  // For larger requests, chunk the calls.
  iree_host_size_t offset = 0;
  while (offset < buffer.data_length) {
    iree_host_size_t chunk_size = buffer.data_length - offset;
    if (chunk_size > 65536) chunk_size = 65536;

    // Use JavaScript's crypto.getRandomValues via Emscripten.
    // Returns 0 on success, 1 on failure.
    int result = EM_ASM_INT(
        {
          try {
            var buf = new Uint8Array(Module.HEAPU8.buffer, $0, $1);
            crypto.getRandomValues(buf);
            return 0;
  }
  catch(e) { return 1; }
},
        buffer.data + offset, chunk_size);

if (result != 0) {
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "crypto.getRandomValues failed");
}
offset += chunk_size;
}
return iree_ok_status();
}

#else

#error "CSPRNG not implemented for this platform"

#endif  // IREE_PLATFORM_*
