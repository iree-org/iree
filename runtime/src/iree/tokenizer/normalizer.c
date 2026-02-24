// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer.h"

//===----------------------------------------------------------------------===//
// Normalizer Base Implementation
//===----------------------------------------------------------------------===//

void iree_tokenizer_normalizer_initialize(
    iree_tokenizer_normalizer_t* normalizer,
    const iree_tokenizer_normalizer_vtable_t* vtable,
    iree_host_size_t state_size) {
  normalizer->vtable = vtable;
  normalizer->state_size = state_size;
}

void iree_tokenizer_normalizer_free(iree_tokenizer_normalizer_t* normalizer) {
  if (!normalizer) return;
  normalizer->vtable->destroy(normalizer);
}
