// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <list>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Utils/IndexSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-layout-slices"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {
namespace {

using Slice = IREE::Stream::ResourcePackOp::Slice;

// Packs slices back-to-back with no aliasing. Useful when debugging to remove
// the aliasing that makes data breakpoints useless.
//
// Slice packed offset SSA values will be updated and start at the given
// |baseOffset|. Returns |baseOffset| + the total size of the allocation
// aligned to the requirements of |resourceConfig|.
static Value
packSlicesWithNoAliasing(IREE::Stream::ResourcePackOp packOp, Value baseOffset,
                         ArrayRef<Slice> slices,
                         IREE::Stream::ResourceConfigAttr resourceConfig,
                         IndexSet &indexSet, OpBuilder &builder) {
  auto loc = packOp.getLoc();
  int64_t offsetAlignment = resourceConfig.getMinBufferOffsetAlignment();
  int64_t rangeAlignment = resourceConfig.getMinBufferRangeAlignment();

  Value offset = baseOffset;
  for (auto &slice : slices) {
    auto sliceSize = builder.createOrFold<IREE::Util::AlignOp>(
        loc, slice.dynamicSize, rangeAlignment);
    slice.packedOffset.replaceAllUsesWith(offset);
    auto valueToAlign =
        builder.createOrFold<arith::AddIOp>(loc, offset, sliceSize);
    offset = builder.createOrFold<IREE::Util::AlignOp>(loc, valueToAlign,
                                                       offsetAlignment);
  }

  return builder.createOrFold<IREE::Util::AlignOp>(loc, offset, rangeAlignment);
}

// Packs a set of statically-sized slices by greedy strip packing.
//
// This is the same algorithm used in tflite here:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/simple_memory_arena.cc
// It's not fantastic and can end up with a significant amount of wastage.
// They do the packing at runtime and as such care more about performance than
// we do while doing the packing here offline. For this initial version I
// wanted to ensure we matched apples/apples their implementation.
//
// We should use a set of heuristics and pick the smallest one. There are also
// some really great papers that have approximations (as all of these are -
// 2D strip packing is NP-hard) such as
// https://www.sciencedirect.com/science/article/pii/S0925772113001016 that
// someone with a brain able to parse mathy papers can try implementing.
//
// Slice packed offset SSA values will be updated and start at the given
// |baseOffset|. Returns |baseOffset| + the total size of the allocation
// aligned to the requirements of |resourceConfig|.
static Value
packStaticSlicesGreedily(IREE::Stream::ResourcePackOp packOp, Value baseOffset,
                         ArrayRef<Slice> slices,
                         IREE::Stream::ResourceConfigAttr resourceConfig,
                         IndexSet &indexSet, OpBuilder &builder) {
  int64_t offsetAlignment = resourceConfig.getMinBufferOffsetAlignment();
  int64_t rangeAlignment = resourceConfig.getMinBufferRangeAlignment();

  struct Reservation {
    const Slice *slice = nullptr;
    int64_t staticOffset = 0;
    int64_t staticSize = 0;
  };
  static constexpr int64_t UNASSIGNED = INT64_MAX;

  std::list<Reservation> reservations;
  int64_t highwaterMark = 0;
  for (auto &slice : slices) {
    int64_t bestOffset = UNASSIGNED;
    int64_t bestOffsetFit = UNASSIGNED;
    int64_t staticSize =
        cast<arith::ConstantIndexOp>(slice.dynamicSize.getDefiningOp()).value();
    int64_t alignedSize = IREE::Util::align(staticSize, rangeAlignment);

    // Iterate through reservations (sorted by ascending offset) and identify
    // gaps in which the slice will fit. To reduce wastage we want to find the
    // smallest gap.
    int64_t currentOffset = 0;
    for (auto &reservation : reservations) {
      if (!reservation.slice->intersects(slice)) {
        // Non-overlapping - we can reuse the currentOffset (assuming we find
        // no better place).
        continue;
      }

      // If we found a gap >= the required size and smaller than
      // previous best fit take it.
      int64_t alignedOffset = IREE::Util::align(currentOffset, offsetAlignment);
      if (alignedOffset + alignedSize <= reservation.staticOffset &&
          reservation.staticOffset - alignedOffset < bestOffsetFit) {
        bestOffset = alignedOffset;
        bestOffsetFit = reservation.staticOffset - currentOffset;
      }
      currentOffset = std::max(currentOffset, reservation.staticOffset +
                                                  reservation.staticSize);
    }
    if (bestOffset == UNASSIGNED) {
      bestOffset = IREE::Util::align(currentOffset, offsetAlignment);
    }

    // Reserve the memory.
    Reservation reservation;
    reservation.slice = &slice;
    reservation.staticOffset = bestOffset;
    reservation.staticSize = alignedSize;
    auto insertionIt = reservations.begin();
    while (insertionIt != reservations.end() &&
           insertionIt->staticOffset < reservation.staticOffset) {
      ++insertionIt;
    }
    reservations.insert(insertionIt, reservation);
    slice.packedOffset.replaceAllUsesWith(builder.createOrFold<arith::AddIOp>(
        packOp.getLoc(), baseOffset, indexSet.get(bestOffset)));

    // Update highwater mark indicating how much memory needs to be allocated
    // for the entire slab.
    highwaterMark = std::max(highwaterMark, bestOffset + alignedSize);
  }

  highwaterMark = IREE::Util::align(highwaterMark, rangeAlignment);
  return builder.createOrFold<arith::AddIOp>(packOp.getLoc(), baseOffset,
                                             indexSet.get(highwaterMark));
}

// Packs a set of dynamically-sized slices based on the structural information
// in the IR. Only slices that have the exact same size will be allowed to
// alias.
//
// We can improve this if we know which sizes are larger/smaller relative to
// others as then we can do things like reuse an allocation bucket with
// non-overlapping lifetimes if the thing we are trying to pack in it is
// definitely <=. If we end up knowing that certain sizes are less than some
// absolute value then we could also alias with static allocations like if %sz
// is known less than 1000b it could reuse any static allocation >= 1000b.
//
// We could also emit code for efficient runtime bucketing by providing the
// sorted, compacted, delta-coded lifetime intervals and runtime-computed
// sizes if we wanted to produce the smallest buffers.
//
// Slice packed offset SSA values will be updated and start at the given
// |baseOffset|. Returns |baseOffset| + the total size of the allocation
// aligned to the requirements of |resourceConfig|.
static Value
packDynamicSlicesConservatively(IREE::Stream::ResourcePackOp packOp,
                                Value baseOffset, ArrayRef<Slice> slices,
                                IREE::Stream::ResourceConfigAttr resourceConfig,
                                IndexSet &indexSet, OpBuilder &builder) {
  auto loc = packOp.getLoc();
  int64_t offsetAlignment = resourceConfig.getMinBufferOffsetAlignment();
  int64_t rangeAlignment = resourceConfig.getMinBufferRangeAlignment();

  // Bucket all slices by their size SSA value. We rely on shapes being
  // lowered to computed byte sizes and CSE to then dedupe the values for us.
  llvm::MapVector<Value, SmallVector<const Slice *>> slicesBySize;
  for (auto &slice : slices) {
    slicesBySize[slice.dynamicSize].push_back(&slice);
  }

  // Allocate each bucket while observing the lifetime of the slices within.
  // Two or more slices may overlap in lifetime and need their own unique
  // reservation.
  Value offset = baseOffset;
  for (auto &sizeBucket : slicesBySize) {
    auto sliceSize = builder.createOrFold<IREE::Util::AlignOp>(
        loc, sizeBucket.first, rangeAlignment);
    auto &slices = sizeBucket.second;
    std::stable_sort(slices.begin(), slices.end());

    // Bin the slices by those that do not overlap. All of the allocations in
    // each bin can alias.
    // NOTE: O(n^2) in the worst case but there's usually only a small number
    // of bins (<10) as we have already bucketed by size class. We could do
    // some sorting and make this O(nlogn) or O(logn) with some interval tree
    // magic.
    struct Bin {
      Value offset;
      SmallVector<const Slice *> slices;
      bool intersects(const Slice &slice) const {
        for (auto *binSlice : slices) {
          if (binSlice->intersects(slice))
            return true;
        }
        return false;
      }
    };
    SmallVector<Bin> bins;
    for (auto *slice : slices) {
      // Try to find a bin we can reuse (non-intersecting lifetime).
      Bin *targetBin = nullptr;
      for (auto &bin : bins) {
        if (!bin.intersects(*slice)) {
          targetBin = &bin;
          break;
        }
      }
      if (!targetBin) {
        // Allocate a new bin for this slice.
        bins.push_back({offset, {}});
        targetBin = &bins.back();
        auto binSize =
            builder.createOrFold<arith::AddIOp>(loc, offset, sliceSize);
        offset = builder.createOrFold<IREE::Util::AlignOp>(loc, binSize,
                                                           offsetAlignment);
      }
      targetBin->slices.push_back(slice);
      slice->packedOffset.replaceAllUsesWith(targetBin->offset);
    }
  }

  return builder.createOrFold<IREE::Util::AlignOp>(loc, offset, rangeAlignment);
}

//===----------------------------------------------------------------------===//
// -iree-stream-layout-slices
//===----------------------------------------------------------------------===//

class LayoutSlicesPass : public LayoutSlicesBase<LayoutSlicesPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto parentOp = getOperation();
    if (!parentOp.getCallableRegion() ||
        parentOp.getCallableRegion()->empty()) {
      return;
    }

    // NOTE: we could try several algorithms and compute which packs best. For
    // now we just pack greedily as it's fast and what most existing ML
    // frameworks do.
    parentOp.walk([&](IREE::Stream::ResourcePackOp packOp) {
      // Derive resource constraints based on pack affinity.
      auto resourceConfig = IREE::Stream::ResourceConfigAttr::lookup(packOp);

      // Bucket into static and dynamic sizes. Static packing is a much more
      // constrained problem.
      auto allSlices = packOp.getSlices();
      SmallVector<Slice> staticSlices;
      SmallVector<Slice> dynamicSlices;
      staticSlices.reserve(allSlices.size());
      dynamicSlices.reserve(allSlices.size());
      for (auto &slice : allSlices) {
        if (isa_and_nonnull<arith::ConstantOp>(
                slice.dynamicSize.getDefiningOp())) {
          staticSlices.push_back(slice);
        } else {
          dynamicSlices.push_back(slice);
        }
      }

      OpBuilder builder(packOp);
      IndexSet indexSet(packOp.getLoc(), builder);

      // First pack all static slices as these are entirely knowable here at
      // compile time.
      auto offset = packOp.getOffset() ? packOp.getOffset() : indexSet.get(0);
      if (!staticSlices.empty()) {
        offset = packStaticSlicesGreedily(packOp, offset, staticSlices,
                                          resourceConfig, indexSet, builder);

        // TODO(benvanik): make this an option; it can be useful for debugging
        // this code.
        // offset = packSlicesWithNoAliasing(packOp, offset, staticSlices,
        //                                   resourceConfig, indexSet, builder);
      }

      // Next pack all dynamic slices. Depending on the analysis information
      // available we could reuse static slices with non-overlapping lifetimes
      // in some cases.
      if (!dynamicSlices.empty()) {
        offset = packDynamicSlicesConservatively(
            packOp, offset, dynamicSlices, resourceConfig, indexSet, builder);
      }

      // Total packed length is the current offset after all slices are
      // allocated. This should be aligned to the range constraints.
      packOp.getTotalLength().replaceAllUsesWith(offset);

      packOp.erase();
    });
  }
};

} // namespace

std::unique_ptr<InterfacePass<CallableOpInterface>> createLayoutSlicesPass() {
  return std::make_unique<LayoutSlicesPass>();
}

} // namespace Stream
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
