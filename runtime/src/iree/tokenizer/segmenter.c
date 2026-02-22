// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/segmenter.h"

//===----------------------------------------------------------------------===//
// Segmenter Base Implementation
//===----------------------------------------------------------------------===//

void iree_tokenizer_segmenter_initialize(
    iree_tokenizer_segmenter_t* segmenter,
    const iree_tokenizer_segmenter_vtable_t* vtable,
    iree_host_size_t state_size) {
  segmenter->vtable = vtable;
  segmenter->state_size = state_size;
}

void iree_tokenizer_segmenter_free(iree_tokenizer_segmenter_t* segmenter) {
  if (!segmenter) return;
  segmenter->vtable->destroy(segmenter);
}
