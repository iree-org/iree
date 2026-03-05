// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>

#include "iree/base/internal/base64.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  iree_string_view_t encoded = {(const char*)data, size};

  iree_host_size_t max_size = iree_base64_decoded_size(encoded);
  if (max_size > 1024 * 1024) return 0;  // Skip unreasonably large inputs.

  uint8_t* buffer = nullptr;
  if (max_size > 0) {
    buffer = (uint8_t*)malloc(max_size);
    if (!buffer) return 0;
  }

  iree_host_size_t actual_length = 0;
  iree_status_t status =
      iree_base64_decode(encoded, max_size, buffer, &actual_length);
  iree_status_ignore(status);

  free(buffer);
  return 0;
}
