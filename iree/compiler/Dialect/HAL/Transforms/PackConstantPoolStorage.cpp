// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// TODO(#7277): remove when switched to streams (happens there now).
class PackConstantPoolStoragePass
    : public PassWrapper<PackConstantPoolStoragePass,
                         OperationPass<ConstantPoolOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
  }

  StringRef getArgument() const override {
    return "iree-hal-pack-constant-pool-storage";
  }

  StringRef getDescription() const override {
    return "Packs all constants in a hal.constant_pool into their possibly "
           "target-dependent storage formats.";
  }

  void runOnOperation() override {
    auto poolOp = getOperation();
    auto bufferConstraints = poolOp.buffer_constraints();
    if (failed(packConstantPool(poolOp, bufferConstraints))) {
      signalPassFailure();
      return;
    }
  }

 private:
  // Packs all constant values within |poolOp| into storage buffers.
  // Zero or more top-level module byte buffers will be inserted.
  // Safe to call on constant pools that have already been packed; only newly
  // inserted constant values will get packed and they will be placed into
  // new buffers.
  //
  // New module-level operations will be inserted before |moduleInsertionPoint|.
  LogicalResult packConstantPool(ConstantPoolOp poolOp,
                                 BufferConstraintsAttr bufferConstraints) {
    // We only pack values either into storage (dense, real data) or represent
    // them as values that will be filled at runtime (splatted values).
    SmallVector<ConstantPoolValueOp, 8> denseValueOps;
    SmallVector<ConstantPoolValueOp, 8> splatValueOps;
    poolOp.walk([&](ConstantPoolValueOp valueOp) {
      if (auto splatAttr = valueOp.value().dyn_cast<SplatElementsAttr>()) {
        splatValueOps.push_back(valueOp);
      } else if (auto denseAttr =
                     valueOp.value().dyn_cast<DenseElementsAttr>()) {
        denseValueOps.push_back(valueOp);
      }
    });

    // Create splat values that future passes handling the runtime work will
    // use to splat the element value directly into memory.
    for (auto splatValueOp : splatValueOps) {
      OpBuilder builder(poolOp.getContext());
      builder.setInsertionPointAfter(splatValueOp);
      auto splatOp = builder.create<ConstantPoolSplatOp>(
          splatValueOp.getLoc(), splatValueOp.getName(),
          splatValueOp.value().cast<SplatElementsAttr>());
      splatOp.setNested();
      splatValueOp.erase();
    }

    // Perform the packing of dense values to compute the storage buffers we
    // will need and where each value will be placed.
    auto storageBuffers = computePackingMap(denseValueOps, bufferConstraints,
                                            poolOp.getContext());
    if (storageBuffers.empty()) return success();

    // Create the storage buffer variables.
    SymbolTable poolSymbolTable(poolOp);
    for (auto storageBuffer : storageBuffers) {
      auto storageBufferLoc = storageBuffer.loc.hasValue()
                                  ? storageBuffer.loc.getValue()
                                  : UnknownLoc::get(poolOp.getContext());
      auto storageBufferOp =
          OpBuilder(poolOp.getContext())
              .create<ConstantStorageOp>(storageBufferLoc, "_storage",
                                         storageBuffer.data);
      poolSymbolTable.insert(storageBufferOp);
      storageBufferOp.setNested();

      // TODO(benvanik): specify alignment attribute for file serialization
      // (minStorageBufferOffsetAlignment) and get vm.rodata handling it.

      // Replace each constant value with a span referencing the storage
      // buffers.
      for (auto constantSpan : storageBuffer.spans) {
        auto valueOp = constantSpan.valueOp;
        OpBuilder poolBuilder(poolOp.getContext());
        poolBuilder.setInsertionPointAfter(valueOp);
        auto spanOp = poolBuilder.create<ConstantPoolSpanOp>(
            valueOp.getLoc(), valueOp.getName(), valueOp.value().getType(),
            SymbolRefAttr::get(storageBufferOp),
            IREE::Util::ByteRangeAttr::get(
                poolOp.getContext(), constantSpan.offset, constantSpan.length),
            SymbolRefAttr{}, IREE::Util::ByteRangeAttr{});
        spanOp.setNested();
        valueOp.erase();
      }
    }

    return success();
  }

  struct ConstantSpan {
    // Original value op this span represents.
    ConstantPoolValueOp valueOp;
    // Byte offset within the storage buffer.
    uint64_t offset = 0;
    // Length of the valid data when padded out.
    // This is only accounting for the padding of the valid data itself and not
    // any additional padding for other spans within the buffer (like start
    // offset alignment).
    uint64_t length = 0;
  };

  struct StorageBuffer {
    // Total size in bytes (including padding).
    uint64_t totalSize = 0;
    // Fused location of all spans that make up this storage buffer.
    Optional<Location> loc;
    // Constant spans packed into this buffer.
    SmallVector<ConstantSpan, 8> spans;
    // Packed byte data that must be embedded in the final module.
    // It must be written with an alignment as required by the constraints.
    IREE::Util::CompositeAttr data;
  };

  // Returns zero or more storage buffers and the spans values map into.
  // Assume that |valueOps| have been ordered by prior passes and that order may
  // have some performance-sensitivity (constants are grouped by
  // locality/lifetime/etc).
  SmallVector<StorageBuffer, 8> computePackingMap(
      ArrayRef<ConstantPoolValueOp> valueOps,
      BufferConstraintsAttr bufferConstraints, MLIRContext *context) {
    // This is literally all my brain has brain for right now. The ideal here is
    // that we have a basic static (and ideally profile-guided) sorting pass
    // that keeps constant values that are accessed sorted together.
    //
    // We want good spatial locality as being in the same storage buffer means
    // that constants are likelier to be pulled into memory together (by disk
    // prefetcher pulling in mapped pages, TLB cache being hot, etc). We want
    // good temporal locality because then we have a higher chance of the
    // constants being placed into the same runtime buffer and that reduces the
    // amount of buffer swapping/bindings we need to manage when recording
    // commands.
    //
    // <story time> Funnily enough, the same reasons we care here (and this same
    // algorithm) are the same that console developers encountered in the early
    // days of CD-ROMs; inserting padding to force alignment on block
    // boundaries, ensuring that temporally related content was together even if
    // it meant repeating things, etc - like always physically duplicate the
    // music near each level that uses it and potentially even interleave those
    // together in block order on the disc, as being able to stream the music
    // and still seek to blocks in level content was worth the % of lost space.
    // You could listen for how well-optimized a game was by the noise level of
    // the read head! (incidentally, same case too with tapes and floppies,
    // however the space limitations were almost always the top concern there -
    // it wasn't until CD-ROM and beyond that there was enough space to shuffle
    // things around and waste on silly things like loading times).
    //
    // Here it's all descriptor sets and mapped pages but same thing pretty
    // much, and passes earlier on may duplicate constants in the pool if it
    // means they can improve locality at runtime. This pass doesn't dedupe and
    // just sticks to packing for that reason.

    // Build a list of buffers and spans (append to current or spill to new).
    auto storageBuffers =
        bucketValuesIntoStorageBuffers(valueOps, bufferConstraints);

    // Pack each storage buffer bucket into a single data blob.
    for (auto &storageBuffer : storageBuffers) {
      packStorageBufferData(storageBuffer, context);
    }

    return storageBuffers;
  }

  // Buckets |valueOps| into one or more storage buffers based on
  // |bufferConstraints|.
  SmallVector<StorageBuffer, 8> bucketValuesIntoStorageBuffers(
      ArrayRef<ConstantPoolValueOp> valueOps,
      BufferConstraintsAttr bufferConstraints) {
    // TODO(benvanik): replace with a better strategy (best-fit, etc).
    SmallVector<StorageBuffer, 8> storageBuffers;
    storageBuffers.push_back({});
    StorageBuffer *currentBuffer = &storageBuffers.back();
    for (auto valueOp : valueOps) {
      uint64_t offset =
          IREE::Util::align(currentBuffer->totalSize,
                            bufferConstraints.min_buffer_offset_alignment());
      uint64_t unpaddedLength =
          valueOp.value().cast<DenseElementsAttr>().getRawData().size();
      uint64_t paddedLength = IREE::Util::align(
          unpaddedLength, bufferConstraints.min_buffer_range_alignment());
      if (offset + unpaddedLength >
          bufferConstraints.max_allocation_size().getZExtValue()) {
        // Spilling buffer; make a new one.
        storageBuffers.push_back({});
        currentBuffer = &storageBuffers.back();
        offset = 0;
      }
      currentBuffer->spans.push_back({valueOp, offset, unpaddedLength});
      currentBuffer->totalSize =
          std::max(currentBuffer->totalSize, offset + paddedLength);
    }
    if (storageBuffers.back().spans.empty()) {
      storageBuffers.pop_back();
    }
    return storageBuffers;
  }

  // Packs all span data into a single data attribute we can tag on the buffer.
  // The data produced will contain all spans at the specified offsets with no
  // additional padding.
  //
  // NOTE: data can overlap so do not assume that the order between spans
  // is contiguous or always increasing! Always seek!
  void packStorageBufferData(StorageBuffer &storageBuffer,
                             MLIRContext *context) {
    // The constants get rolled into the buffer. This neat bit of info would
    // be useful if we wanted to map back a module size through data blobs.
    // With buffer <-> constant it's possible to build a tree map of
    // contributions in the source. TBD ;)
    storageBuffer.loc =
        FusedLoc::get(context, llvm::to_vector<8>(llvm::map_range(
                                   storageBuffer.spans, [](ConstantSpan &span) {
                                     return span.valueOp.getLoc();
                                   })));

    // Construct a composite attribute that contains references to the original
    // uniqued storage values. This avoids needing to reallocate/unique/copy
    // the entire constant storage. Since we may have inserted padding between
    // values to meet the target buffer constraints we may need to insert some
    // padding bytes between the elements to keep the composite dense.
    auto i8Type = IntegerType::get(context, 8);
    auto zeroAttr = IntegerAttr::get(i8Type, 0);
    SmallVector<Attribute> values;
    int64_t offset = 0;
    for (auto &constantSpan : storageBuffer.spans) {
      if (constantSpan.length == 0) continue;

      int64_t start = constantSpan.offset;
      int64_t end = start + constantSpan.length;

      // TODO(benvanik): when we start overlapping data we'll want to have a way
      // to build subranges here by slicing out parts of the attributes we have
      // coming in. This could be done with a #util.slice<> attr or something
      // that when serialized performs the offsetting.
      assert(start >= offset && "expect ordered spans");

      int64_t spanPadding = start - offset;
      if (spanPadding > 0) {
        // 0-pad until the start of this span.
        values.push_back(DenseElementsAttr::get(
            VectorType::get({spanPadding}, i8Type), zeroAttr));
        offset += spanPadding;
      }

      values.push_back(constantSpan.valueOp.value());
      offset = end;
    }

    // Add tail padding. Unfortunate but some implementations will require the
    // bytes and it's possible (and in some cases intentional) that the storage
    // buffers fall at the end of mapped memory and need the valid bytes for
    // page-granularity access.
    int64_t tailPadding = storageBuffer.totalSize - offset;
    if (tailPadding > 0) {
      // 0-pad until the start of this span.
      values.push_back(DenseElementsAttr::get(
          VectorType::get({tailPadding}, i8Type), zeroAttr));
      offset += tailPadding;
    }

    storageBuffer.data = IREE::Util::CompositeAttr::get(context, values);
  }
};

std::unique_ptr<OperationPass<ConstantPoolOp>>
createPackConstantPoolStoragePass() {
  return std::make_unique<PackConstantPoolStoragePass>();
}

static PassRegistration<PackConstantPoolStoragePass> pass;

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
