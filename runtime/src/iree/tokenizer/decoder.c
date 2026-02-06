// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder.h"

//===----------------------------------------------------------------------===//
// Decoder Base Implementation
//===----------------------------------------------------------------------===//

void iree_tokenizer_decoder_initialize(
    iree_tokenizer_decoder_t* decoder,
    const iree_tokenizer_decoder_vtable_t* vtable, iree_host_size_t state_size,
    iree_tokenizer_decoder_capability_t capabilities) {
  decoder->vtable = vtable;
  decoder->state_size = state_size;
  decoder->capabilities = capabilities;
}

void iree_tokenizer_decoder_free(iree_tokenizer_decoder_t* decoder) {
  if (!decoder) return;
  decoder->vtable->destroy(decoder);
}
