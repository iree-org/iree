// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_ANNOTATECONSTANTTRANSIENTSIZEPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-stream-annotate-constant-transient-size
//===----------------------------------------------------------------------===//

struct AnnotateConstantTransientSizePass
    : public IREE::Stream::impl::AnnotateConstantTransientSizePassBase<
          AnnotateConstantTransientSizePass> {
  void runOnOperation() override {
    // TODO(benvanik): Implement AnnotateConstantTransientSizePass for Phase 2.
    // This pass should:
    // 1. Iterate over transient size query functions (identified by reflection
    //    metadata on the original functions).
    // 2. Check if function body has folded to arith.constant return.
    // 3. Extract constant size value.
    // 4. Add iree.reflection metadata to original function with constant size.
    //
    // Pulled forward from Phase 3 to validate constant size detection early.
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
