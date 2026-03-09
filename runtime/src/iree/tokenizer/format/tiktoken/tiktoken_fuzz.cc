// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>

#include "iree/tokenizer/format/tiktoken/tiktoken.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // Use the cl100k_base config as a representative config.
  const iree_tokenizer_tiktoken_config_t* config =
      iree_tokenizer_tiktoken_config_cl100k_base();

  iree_string_view_t input = {(const char*)data, size};

  iree_tokenizer_t* tokenizer = NULL;
  iree_status_t status = iree_tokenizer_from_tiktoken(
      input, config, iree_allocator_system(), &tokenizer);
  if (iree_status_is_ok(status)) {
    iree_tokenizer_free(tokenizer);
  } else {
    iree_status_ignore(status);
  }

  return 0;
}
