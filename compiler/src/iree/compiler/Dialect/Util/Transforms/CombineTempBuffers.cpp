// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/IndexSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"

#define DEBUG_TYPE "iree-util-combine-initializers"

using llvm::dbgs;

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {
namespace {

struct StaticBuffer {
  // The 0-based index of this allocation in the analyzed scope.
  int index;
  int64_t size;
  int64_t alignment;
  Value buffer;
  Operation *allocOp;
  // Index of the last buffer that overlaps access with this one.
  int liveEnd;
  int64_t packedOffset = 0;
  StaticBuffer(int index, int64_t size, int64_t alignment, Value buffer,
               Operation *allocOp)
      : index(index),
        size(size),
        alignment(alignment),
        buffer(buffer),
        allocOp(allocOp),
        liveEnd(index) {}

  bool intersects(StaticBuffer &other) {
    return (index >= other.index && index <= other.liveEnd) ||
           (other.index >= index && other.index <= liveEnd);
  }
};

class CombineTempBuffersPass
    : public CombineTempBuffersBase<CombineTempBuffersPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    if (getOperation().getBlocks().size() != 1) {
      LLVM_DEBUG(dbgs() << "Skipping function (not single block)\n");
      return;
    }
    Block &entryBlock = getOperation().getBlocks().front();
    Liveness liveness(getOperation());

    // TODO: Multi-block functions need more elaborate escape analysis.
    auto functionEscapes = liveness.getLiveOut(&entryBlock);
    auto isTemporaryBuffer = [&](BufferAllocOp allocOp) {
      if (functionEscapes.contains(allocOp.getResult())) return false;
      return true;
    };

    SmallVector<StaticBuffer> staticBuffers;

    // Only consider allocations at the top-level of the block.
    for (Operation &op : entryBlock) {
      if (auto allocOp = dyn_cast<BufferAllocOp>(op)) {
        if (!isTemporaryBuffer(allocOp)) continue;
        // Switch on static vs dynamic.
        APInt staticSize;
        if (matchPattern(allocOp.getStorageSize(),
                         m_ConstantInt(&staticSize))) {
          int offset = static_cast<int>(staticBuffers.size());
          int64_t staticSizeInt = staticSize.getZExtValue();
          int64_t alignment = 8;
          if (allocOp.getAlignment()) {
            alignment = allocOp.getAlignment()->getZExtValue();
          }
          LLVM_DEBUG(dbgs()
                     << "TEMP BUFFER[" << offset << "]: size " << staticSizeInt
                     << ", alignment " << alignment << "\n");
          staticBuffers.push_back(StaticBuffer(/*index=*/offset,
                                               /*size=*/staticSizeInt,
                                               /*alignment=*/alignment,
                                               /*buffer=*/allocOp.getResult(),
                                               /*allocOp=*/allocOp));
        } else {
          // TODO: Dynamic.
        }
      }
    }

    // Calculate live ranges.
    for (size_t i = 0; i < staticBuffers.size(); ++i) {
      StaticBuffer &checkBuffer = staticBuffers[i];
      for (size_t j = i + 1; j < staticBuffers.size(); ++j) {
        StaticBuffer &laterBuffer = staticBuffers[j];
        if (liveness.isDeadAfter(checkBuffer.buffer, laterBuffer.allocOp)) {
          break;
        }
        checkBuffer.liveEnd = j;
      }
    }

    // Pack static buffers.
    int64_t staticSlabSize = packStaticBuffersGreedily(staticBuffers);
    LLVM_DEBUG(dbgs() << "STATIC SLAB SIZE: " << staticSlabSize << "\n");
    LLVM_DEBUG(dbgs() << "STATIC BUFFER LIVE RANGES:\n");
    LLVM_DEBUG(dbgs() << "--------------------------\n");
    for (StaticBuffer &sb : staticBuffers) {
      LLVM_DEBUG(dbgs() << "  BUFFER[" << sb.index << "] -> " << sb.liveEnd
                        << " : offset " << sb.packedOffset << "\n");
    }

    // Apply static buffer updates.
    if (!staticBuffers.empty()) {
      applyStaticBufferPacking(getOperation().getLoc(), entryBlock,
                               staticSlabSize, staticBuffers);
    }
  }

  void applyStaticBufferPacking(Location loc, Block &block,
                                int64_t staticSlabSize,
                                ArrayRef<StaticBuffer> staticBuffers) {
    OpBuilder builder = OpBuilder::atBlockBegin(&block);
    IndexSet indexSet(loc, builder);

    // Create slab.
    int64_t alignment = 8;
    for (auto &sb : staticBuffers) {
      alignment = std::max(alignment, sb.alignment);
    }
    Value slabSize = indexSet.get(staticSlabSize);
    Value slab = builder.create<BufferAllocOp>(loc, slabSize,
                                               builder.getIndexAttr(alignment));

    // Create alias buffers.
    for (auto &sb : staticBuffers) {
      // Remove any existing deallocs.
      for (Operation *user : sb.buffer.getUsers()) {
        if (llvm::isa<BufferDeallocOp>(user)) {
          user->erase();
        }
      }

      // Create a new view and replace.
      Value newView = builder.create<BufferSubspanOp>(
          sb.allocOp->getLoc(), slab, slabSize, indexSet.get(sb.packedOffset),
          indexSet.get(sb.size));
      sb.buffer.replaceAllUsesWith(newView);
      sb.allocOp->erase();
    }

    // Deallocate.
    builder = OpBuilder::atBlockTerminator(&block);
    builder.create<BufferDeallocOp>(loc, slab, slabSize);
  }

  // Packs an array of static buffers into a single slab of memory.
  // Returns the total slab size.
  int64_t packStaticBuffersGreedily(
      MutableArrayRef<StaticBuffer> staticBuffers) {
    struct Reservation {
      StaticBuffer *sb = nullptr;
      int64_t staticOffset = 0;
      int64_t staticSize = 0;
    };
    static constexpr int64_t UNASSIGNED = INT64_MAX;
    std::vector<Reservation> reservations;
    reservations.reserve(staticBuffers.size());
    int64_t highwaterMark = 0;

    for (auto &sb : staticBuffers) {
      int64_t bestOffset = UNASSIGNED;
      int64_t bestOffsetFit = UNASSIGNED;
      int64_t alignedSize = IREE::Util::align(sb.size, sb.alignment);

      // Iterate through reservations (sorted by ascending offset) and identify
      // gaps in which the slice will fit. To reduce wastage we want to find the
      // smallest gap.
      int64_t currentOffset = 0;
      for (auto &reservation : reservations) {
        if (!reservation.sb->intersects(sb)) {
          // Non-overlapping - we can reuse the currentOffset (assuming we find
          // no better place).
          continue;
        }

        // If we found a gap >= the required size and smaller than
        // previous best fit take it.
        int64_t alignedOffset = IREE::Util::align(currentOffset, sb.alignment);
        if (alignedOffset + alignedSize <= reservation.staticOffset &&
            reservation.staticOffset - alignedOffset < bestOffsetFit) {
          bestOffset = alignedOffset;
          bestOffsetFit = reservation.staticOffset - currentOffset;
        }
        currentOffset = std::max(
            currentOffset, reservation.staticOffset + reservation.staticSize);
      }
      if (bestOffset == UNASSIGNED) {
        bestOffset = IREE::Util::align(currentOffset, sb.alignment);
      }

      // Reserve the memory.
      Reservation reservation;
      reservation.sb = &sb;
      reservation.staticOffset = bestOffset;
      reservation.staticSize = alignedSize;
      auto insertionIt = reservations.begin();
      while (insertionIt != reservations.end() &&
             insertionIt->staticOffset < reservation.staticOffset) {
        ++insertionIt;
      }
      reservations.insert(insertionIt, reservation);
      sb.packedOffset = bestOffset;

      // Update highwater mark indicating how much memory needs to be allocated
      // for the entire slab.
      highwaterMark = std::max(highwaterMark, bestOffset + alignedSize);
    }
    return highwaterMark;
  }
};

}  // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createCombineTempBuffersPass() {
  return std::make_unique<CombineTempBuffersPass>();
}

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
