// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"  // IWYU pragma: export
}  // namespace

void registerTransformPasses() {
  // Generated.
  registerPasses();
}

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
