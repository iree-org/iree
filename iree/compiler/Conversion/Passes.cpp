// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Conversion/Passes.h"

namespace mlir {
namespace iree_compiler {

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Conversion/Passes.h.inc"
}  // namespace

void registerConversionPasses() {
  // Generated.
  registerPasses();
}

}  // namespace iree_compiler
}  // namespace mlir
