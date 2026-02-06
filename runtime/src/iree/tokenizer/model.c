// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/model.h"

//===----------------------------------------------------------------------===//
// Model Base Implementation
//===----------------------------------------------------------------------===//

void iree_tokenizer_model_initialize(
    iree_tokenizer_model_t* model, const iree_tokenizer_model_vtable_t* vtable,
    iree_host_size_t state_size, iree_string_view_t type_name) {
  model->vtable = vtable;
  model->state_size = state_size;
  model->type_name = type_name;
}

void iree_tokenizer_model_free(iree_tokenizer_model_t* model) {
  if (!model) return;
  model->vtable->destroy(model);
}
