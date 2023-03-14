// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-pack-allocations"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {
namespace {

//===----------------------------------------------------------------------===//
// -iree-stream-pack-allocations
//===----------------------------------------------------------------------===//

class PackAllocationsPass : public PackAllocationsBase<PackAllocationsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto parentOp = getOperation();
    if (!parentOp.getCallableRegion() ||
        parentOp.getCallableRegion()->empty()) {
      return;
    }

    // This is pretty lazy: we just turn stream.resource.alloc ops into a
    // stream.resource.pack + stream.resource.alloc of a single resource.
    // This way we reuse all the resource constraints stuff that the pack op
    // provides even though all of the resources we allocate have perfectly
    // overlapping lifetime spans.
    //
    // In the future, we should be doing deeper lifetime analysis here and
    // subdividing the allocs based on which resources travel together. We can
    // also do things like overlap the lifetime of inputs and outputs to
    // execution regions as usually inputs end their lifetime before the outputs
    // are produced. In this way we'd use the slice intervals to denote which
    // are mutually exclusive.
    parentOp.walk([&](IREE::Stream::ResourceAllocOp allocOp) {
      // If just one result then ignore (nothing to pack).
      if (allocOp.getResults().size() == 1) return;
      auto resourceType = allocOp.getResults().front().getType();

      // NOTE: this is risky: we are assuming right now that all of the
      // allocations will fit within the constraints of the system. This is not
      // guaranteed: a very low maximum buffer range may lead to packed slabs
      // that are not fully addressable. For now we are processing models with
      // small enough workloads and our target devices are relatively lax on
      // things so long as we stay under UINT32_MAX boundaries.

      // All slices are 0-0 (overlapping).
      size_t sliceCount = allocOp.getResults().size();
      SmallVector<int64_t> lifetimeIntervals(sliceCount * 2, 0);

      OpBuilder builder(allocOp);
      auto indexType = builder.getIndexType();
      SmallVector<Type> packedOffsetTypes(sliceCount, indexType);
      auto packOp = builder.create<IREE::Stream::ResourcePackOp>(
          allocOp.getLoc(), indexType, packedOffsetTypes, /*offset=*/nullptr,
          builder.getIndexArrayAttr(lifetimeIntervals),
          allocOp.getStorageSizes(), allocOp.getAffinityAttr());

      // Change the alloc to build just a single resource.
      auto newOp = builder.create<IREE::Stream::ResourceAllocOp>(
          allocOp.getLoc(), resourceType, packOp.getTotalLength(),
          allocOp.getUninitializedAttr(), allocOp.getAffinityAttr());
      auto slab = newOp.getResults().front();
      auto slabSize = packOp.getTotalLength();

      // Replace all resources with subviews into the new slab.
      for (auto [originalValue, subviewOffset, subviewLength] :
           llvm::zip_equal(allocOp.getResults(), packOp.getPackedOffsets(),
                           allocOp.getStorageSizes())) {
        auto subviewOp = builder.create<IREE::Stream::ResourceSubviewOp>(
            allocOp.getLoc(), slab, slabSize, subviewOffset, subviewLength);
        originalValue.replaceAllUsesWith(subviewOp.getResult());
      }

      allocOp.erase();
    });
  }
};

}  // namespace

std::unique_ptr<InterfacePass<CallableOpInterface>>
createPackAllocationsPass() {
  return std::make_unique<PackAllocationsPass>();
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
