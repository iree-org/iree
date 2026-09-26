// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OOTEX_TRANSFORMS_PASSES_H_
#define OOTEX_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace ootex {

#define GEN_PASS_DECL
#include "ootex/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ootex/Transforms/Passes.h.inc"

}  // namespace ootex

#endif  // OOTEX_TRANSFORMS_PASSES_H_
