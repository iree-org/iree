// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/Partitioning.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-stream-execution-placement"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_EXECUTIONPLACEMENTPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

struct ExecutionPlacementPass
    : public IREE::Stream::impl::ExecutionPlacementPassBase<
          ExecutionPlacementPass> {
  void runOnOperation() override {

    getOperation()->walk([](IREE::Stream::AsyncTransferOp transfer) {
      if (transfer.getAffinityAttr())
        return;

      auto operand = transfer.getSource();
      auto producer = operand.getDefiningOp();
      auto streamable =
          dyn_cast_or_null<IREE::Stream::StreamableOpInterface>(producer);
      auto srcAffinity =
          dyn_cast_or_null<IREE::Stream::AffinityOpInterface>(producer);

      bool hasOneUse = operand.hasOneUse();
      if (hasOneUse && streamable && srcAffinity) {
        transfer.setAffinityAttr(srcAffinity.getAffinityAttr());
        return;
      }
      transfer.setAffinityAttr(transfer.getResultAffinityAttr());
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::Stream
