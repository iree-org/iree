// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Indexing/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::Indexing {

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Dialect/Indexing/Transforms/Passes.h.inc" // IWYU pragma: export
} // namespace

void registerIndexingPasses() {
  // Generated.
  registerPasses();
}

} // namespace mlir::iree_compiler::IREE::Indexing
