// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Preprocessing/Common/Passes.h"

namespace mlir::iree_compiler::Preprocessing {

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Preprocessing/Common/Passes.h.inc" // IWYU pragma: export
} // namespace

void registerCommonPreprocessingPasses() { registerPasses(); }

} // namespace mlir::iree_compiler::Preprocessing
