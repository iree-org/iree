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
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_SYNCINITIALIZERSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-stream-sync-initializers
//===----------------------------------------------------------------------===//

// Finds any global stores of timepoints in initializers and adds them to a
// single blocking wait prior to returning.
struct SyncInitializersPass
    : public IREE::Stream::impl::SyncInitializersPassBase<
          SyncInitializersPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty()) {
      return;
    }

    // NOTE: today we set all timepoint awaits within the initializer to sync.
    // This does not cover awaits in calls reachable from the initializer.
    for (auto initializerOp : moduleOp.getOps<IREE::Util::InitializerOp>()) {
      initializerOp.walk([&](IREE::Stream::TimepointAwaitOp awaitOp) {
        // NOTE: we only need to set sync on awaits that have their results end
        // up in global stores. That requires an analysis that we don't
        // currently perform (walk all uses of each result and see if any can
        // potentially reach a store). The Explorer could be used to help with
        // that.
        awaitOp.setSync(true);
      });
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
