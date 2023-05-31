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
#include "iree/compiler/Utils/IndexSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-pack-constants"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {
namespace {

//===----------------------------------------------------------------------===//
// Pool packing and storage assignment
//===----------------------------------------------------------------------===//

struct ConstantSlice {
  // Result SSA value from the pooling op.
  Value result;
  // Size, in bytes, of the encoded constant value as an SSA value.
  Value resultSize;
  // Constant value being encoded.
  Attribute value;

  // Returns the length, in bytes, of the constant value prior to alignment or
  // padding.
  uint64_t getStorageSize() const {
    if (auto serializableAttr =
            llvm::dyn_cast<IREE::Util::SerializableAttrInterface>(value)) {
      return serializableAttr.getStorageSize();
    } else if (auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(value)) {
      return denseAttr.getRawData().size();
    } else {
      assert(false && "invalid constant attr type");
      return 0;
    }
  }
};

struct PackedSpan {
  // Original slice this span represents.
  ConstantSlice slice;
  // Byte offset within the storage buffer.
  uint64_t offset = 0;
  // Length of the valid data when padded out.
  // This is only accounting for the padding of the valid data itself and not
  // any additional padding for other spans within the buffer (like start
  // offset alignment).
  uint64_t length = 0;
};

struct StorageResource {
  // Fused location of all spans that make up this storage buffer.
  Location loc;
  // Total size in bytes (including padding).
  uint64_t totalSize = 0;
  // Constant spans packed into this resource.
  SmallVector<PackedSpan, 8> spans;
  // Packed byte data that must be embedded in the final module.
  // It must be written with an alignment as required by the constraints.
  IREE::Util::CompositeAttr data;
};

// Buckets |slices| into 1+ storage resources based on |resourceConfig|.
static SmallVector<StorageResource, 8> bucketValuesIntoStorageResources(
    ArrayRef<ConstantSlice> slices,
    IREE::Stream::ResourceConfigAttr resourceConfig) {
  // TODO(benvanik): replace with a better strategy (best-fit, etc).
  SmallVector<StorageResource, 8> storageBuffers;
  storageBuffers.push_back({UnknownLoc::get(resourceConfig.getContext())});
  StorageResource *currentBuffer = &storageBuffers.back();
  for (auto slice : slices) {
    uint64_t offset = IREE::Util::align(
        currentBuffer->totalSize, resourceConfig.getMinBufferOffsetAlignment());
    uint64_t unpaddedLength = slice.getStorageSize();
    uint64_t paddedLength = IREE::Util::align(
        unpaddedLength, resourceConfig.getMinBufferRangeAlignment());
    if (offset + unpaddedLength > resourceConfig.getMaxAllocationSize()) {
      // Spilling buffer; make a new one.
      storageBuffers.push_back({UnknownLoc::get(resourceConfig.getContext())});
      currentBuffer = &storageBuffers.back();
      offset = 0;
    }
    currentBuffer->spans.push_back({slice, offset, unpaddedLength});
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
static void packStorageResourceData(StorageResource &storageBuffer,
                                    MLIRContext *context) {
  // The constants get rolled into the buffer. This neat bit of info would
  // be useful if we wanted to map back a module size through data blobs.
  // With buffer <-> constant it's possible to build a tree map of
  // contributions in the source. TBD ;)
  storageBuffer.loc =
      FusedLoc::get(context, llvm::to_vector<8>(llvm::map_range(
                                 storageBuffer.spans, [](PackedSpan &span) {
                                   return span.slice.result.getLoc();
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

    values.push_back(constantSpan.slice.value);
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
  assert(storageBuffer.data && "unable to build composite attr");
}

// Returns zero or more storage resources and the spans values map into.
// Assume that |slices| have been ordered by prior passes and that order may
// have some performance-sensitivity (constants are grouped by
// locality/lifetime/etc).
static SmallVector<StorageResource, 8> computePackingMap(
    ArrayRef<ConstantSlice> slices,
    IREE::Stream::ResourceConfigAttr resourceConfig, MLIRContext *context) {
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

  // Build a list of resources and spans (append to current or spill to new).
  auto storageBuffers =
      bucketValuesIntoStorageResources(slices, resourceConfig);

  // Pack each storage resource bucket into a single data blob.
  for (auto &storageBuffer : storageBuffers) {
    packStorageResourceData(storageBuffer, context);
  }

  return storageBuffers;
}

//===----------------------------------------------------------------------===//
// Upload materialization
//===----------------------------------------------------------------------===//

struct AllocatedStorage {
  // Resource storing all packed constants.
  Value resource;
  // Total size, in bytes, of the storage resource.
  Value resourceSize;
};

struct UploadResult {
  // Timepoint when the storage is initialized with the constant values.
  Value timepoint;
  // Each (resource, resourceSize) allocated.
  SmallVector<AllocatedStorage> allocations;
};

// Maps constants as a staging buffer and then issues copy commands.
// Per-storage resource we map the source rodata, allocate the result, and then
// issue an async copy from source to result. To avoid a bunch of overhead when
// there are multiple storage buffers we invert the logic so that we put all the
// async copies into a single region.
static UploadResult buildStagingUpload(
    Location loc, IREE::Stream::AffinityAttr affinityAttr,
    IREE::Stream::ResourceType resourceType,
    ArrayRef<StorageResource> storageResources, ArrayRef<Value> storageBuffers,
    IndexSet &indexSet, OpBuilder &builder) {
  UploadResult uploadResult;
  auto stagingType = builder.getType<IREE::Stream::ResourceType>(
      IREE::Stream::Lifetime::Staging);

  // Map all of the storage data and allocate the result buffers.
  // This will produce a list of copies we should perform from staging->final.
  struct Copy {
    Location loc;
    Value source;
    Value sourceSize;
    Value sourceOffset;
    Value target;
    Value targetSize;
    Value targetOffset;
    Value length;
  };
  SmallVector<Copy> copies;
  SmallVector<Value> capturedResources;
  SmallVector<Value> capturedResourceSizes;
  for (auto [storageResource, storageBuffer] :
       llvm::zip_equal(storageResources, storageBuffers)) {
    // Today we assume 1:1 lengths of storage data and uploaded data, but this
    // need not be the case if we want to pad the buffer for runtime.
    auto totalLength = indexSet.get(storageResource.totalSize);

    // Map the source staging resource rodata.
    auto mapOp = builder.create<IREE::Stream::ResourceMapOp>(
        storageResource.loc, stagingType, storageBuffer, indexSet.get(0),
        totalLength, affinityAttr);

    // Allocate the resulting storage resource of the final resource type.
    auto allocOp = builder.create<IREE::Stream::ResourceAllocOp>(
        storageResource.loc, resourceType, mapOp.getResultSize(),
        /*uninitialized=*/builder.getUnitAttr(), affinityAttr);

    uploadResult.allocations.push_back({
        allocOp.getResults().front(),
        allocOp.getStorageSizes().front(),
    });

    // Queue copy for processing below.
    Copy copy{
        storageResource.loc,
        mapOp.getResult(),
        mapOp.getResultSize(),
        indexSet.get(0),
        allocOp.getResults().front(),
        allocOp.getStorageSizes().front(),
        indexSet.get(0),
        totalLength,
    };
    capturedResources.push_back(copy.source);
    capturedResourceSizes.push_back(copy.sourceSize);
    capturedResources.push_back(copy.target);
    capturedResourceSizes.push_back(copy.targetSize);
    copies.push_back(std::move(copy));
  }

  // Create the execution op capturing the resources.
  auto executeOp = builder.create<IREE::Stream::CmdExecuteOp>(
      loc, /*awaitTimepoint=*/Value{}, capturedResources,
      capturedResourceSizes);
  if (affinityAttr) executeOp.setAffinityAttr(affinityAttr);
  uploadResult.timepoint = executeOp.getResultTimepoint();

  // Map captured resources into the execution region.
  IRMapping mapping;
  auto *entryBlock = new Block();
  executeOp.getBody().push_back(entryBlock);
  for (auto outerValue : capturedResources) {
    auto arg =
        entryBlock->addArgument(outerValue.getType(), outerValue.getLoc());
    mapping.map(outerValue, arg);
  }

  // Issue copies. Note that we use the captured resources.
  auto executionBuilder = OpBuilder::atBlockBegin(entryBlock);
  for (auto &copy : copies) {
    executionBuilder.create<IREE::Stream::CmdCopyOp>(
        copy.loc, mapping.lookup(copy.source), copy.sourceSize,
        copy.sourceOffset, mapping.lookup(copy.target), copy.targetSize,
        copy.targetOffset, copy.length);
  }
  executionBuilder.create<IREE::Stream::YieldOp>(executeOp.getLoc());

  return uploadResult;
}

// Emits IR to first try mapping the storage resources directly into usable
// constant resources. If the mapping fails (the target can't use the memory)
// then fall back to staging uploads.
static UploadResult buildTryMapConstantResources(
    Location loc, IREE::Stream::AffinityAttr affinityAttr,
    IREE::Stream::ResourceType resourceType,
    ArrayRef<StorageResource> storageResources, ArrayRef<Value> storageBuffers,
    IndexSet &indexSet, OpBuilder &builder) {
  // Try mapping each resource. We do this as an all-or-nothing across the
  // storage: if any fails we fallback to the allocation path. This is mostly
  // just to get more predictable behavior in the face of weird platform
  // requirements: we want something like misaligned mappings to be easily
  // visible in tracing.
  SmallVector<Value> mappedResources;
  SmallVector<Type> resultTypes;
  Value ok;
  auto zero = indexSet.get(0);
  for (auto [storageResource, storageBuffer] :
       llvm::zip_equal(storageResources, storageBuffers)) {
    auto tryMapOp = builder.create<IREE::Stream::ResourceTryMapOp>(
        storageResource.loc, builder.getI1Type(), resourceType, storageBuffer,
        zero, indexSet.get(storageResource.totalSize), affinityAttr);
    if (!ok) {
      ok = tryMapOp.getDidMap();
    } else {
      ok = builder.createOrFold<arith::AndIOp>(tryMapOp.getLoc(), ok,
                                               tryMapOp.getDidMap());
    }
    mappedResources.push_back(tryMapOp.getResult());
    resultTypes.push_back(tryMapOp.getResult().getType());
  }

  // If we are able to directly map the resources then we don't need to wait.
  auto timepointType = builder.getType<IREE::Stream::TimepointType>();
  resultTypes.push_back(timepointType);

  // if ok: return mapped resources
  // else: allocate and upload
  auto ifOp = builder.create<scf::IfOp>(
      loc, ok,
      [&](OpBuilder &thenBuilder, Location loc) {
        // Just return the resources + an immediate timepoint.
        SmallVector<Value> ifResults = mappedResources;
        ifResults.push_back(
            thenBuilder.create<IREE::Stream::TimepointImmediateOp>(loc));
        thenBuilder.create<scf::YieldOp>(loc, ifResults);
      },
      [&](OpBuilder &elseBuilder, Location loc) {
        // Fallback to upload and then
        auto stagingResult = buildStagingUpload(
            loc, affinityAttr, resourceType, storageResources, storageBuffers,
            indexSet, builder);
        SmallVector<Value> ifResults;
        for (auto &allocation : stagingResult.allocations) {
          ifResults.push_back(allocation.resource);
        }
        ifResults.push_back(stagingResult.timepoint);
        elseBuilder.create<scf::YieldOp>(loc, ifResults);
      });
  auto ifTimepoint = ifOp.getResults().back();
  auto ifResources = ifOp.getResults().slice(0, ifOp.getResults().size() - 1);

  // Use the result of either the direct mapping or the staging upload.
  UploadResult uploadResult;
  uploadResult.timepoint = ifTimepoint;
  for (auto [storageResource, ifResource] :
       llvm::zip_equal(storageResources, ifResources)) {
    uploadResult.allocations.push_back({
        ifResource,
        indexSet.get(storageResource.totalSize),
    });
  }
  return uploadResult;
}

static Value generateUpload(IREE::Stream::ResourceConstantsOp constantsOp,
                            IREE::Stream::Lifetime lifetime,
                            IREE::Stream::ResourceConfigAttr resourceConfig,
                            IndexSet &indexSet, OpBuilder &builder) {
  // Gather the slices produced by this constant pooling op.
  SmallVector<ConstantSlice> slices;
  slices.reserve(constantsOp.getResults().size());
  for (auto [result, resultSize, value] :
       llvm::zip_equal(constantsOp.getResults(), constantsOp.getResultSizes(),
                       constantsOp.getValues())) {
    auto resourceType =
        llvm::cast<IREE::Stream::ResourceType>(result.getType());
    if (resourceType.getLifetime() != lifetime) continue;
    slices.push_back(ConstantSlice{
        result,
        resultSize,
        value,
    });
  }

  // Perform the packing of dense values to compute the storage resources we
  // will need and where each value will be placed.
  auto storageResources =
      computePackingMap(slices, resourceConfig, constantsOp.getContext());
  if (storageResources.empty()) return nullptr;

  // Emit rodata storage for the constant values.
  // As our upload paths may vary this ensures that we are only emitting
  // them once regardless of how many strategies we emit IR for.
  SmallVector<Value> storageBuffers;
  for (auto &storageResource : storageResources) {
    auto rodataOp = builder.create<IREE::Util::BufferConstantOp>(
        storageResource.loc, /*name=*/nullptr, storageResource.data,
        builder.getIndexAttr(resourceConfig.getMinBufferOffsetAlignment()),
        /*mimeType=*/nullptr);
    storageBuffers.push_back(rodataOp);
  }

  // If this is producing constants (vs variables) we can try to go on a
  // fast-path where we directly map the constant memory. If producing
  // variables then we always need to stage and clone.
  auto anyResult = slices.front().result;
  auto resourceType =
      llvm::cast<IREE::Stream::ResourceType>(anyResult.getType());
  UploadResult uploadResult;
  if (resourceType.getLifetime() == IREE::Stream::Lifetime::Constant) {
    uploadResult = buildTryMapConstantResources(
        constantsOp.getLoc(), constantsOp.getAffinityAttr(), resourceType,
        storageResources, storageBuffers, indexSet, builder);
  } else {
    uploadResult = buildStagingUpload(
        constantsOp.getLoc(), constantsOp.getAffinityAttr(), resourceType,
        storageResources, storageBuffers, indexSet, builder);
  }

  // Build subviews for all packed spans back into storage buffers.
  for (auto [storageResource, allocatedStorage] :
       llvm::zip_equal(storageResources, uploadResult.allocations)) {
    for (auto &span : storageResource.spans) {
      auto loc = span.slice.result.getLoc();
      auto subviewOp = builder.create<IREE::Stream::ResourceSubviewOp>(
          loc, allocatedStorage.resource, allocatedStorage.resourceSize,
          indexSet.get(span.offset), span.slice.resultSize);
      span.slice.result.replaceAllUsesWith(subviewOp.getResult());
    }
  }

  // Join on storage timepoints for our transitive dependencies to await.
  return uploadResult.timepoint;
}

//===----------------------------------------------------------------------===//
// -iree-stream-pack-constants
//===----------------------------------------------------------------------===//

// NOTE: this pass currently produces suboptimal packing when multiple constant
// pools exist within the module. Each pool is packed independently even if
// there is overlap in subelement spans - as often is the case in multi-device
// partitioning where a constant may be used on multiple devices. By being
// purely local here we would pack those constants for each use into the final
// binary. A global pass for gathering constants from the whole program and
// splitting into dedicated target-agnostic staging buffers that we bake once
// would be much nicer. For now, though, we don't do multi-device so there's
// never a case where this matters by construction; which is a feature :P

class PackConstantsPass : public PackConstantsBase<PackConstantsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto parentOp = getOperation();
    if (!parentOp || !parentOp.getCallableRegion() ||
        parentOp.getCallableRegion()->empty()) {
      return;
    }

    parentOp.walk([&](IREE::Stream::ResourceConstantsOp constantsOp) {
      // Derive resource constraints based on pack affinity.
      auto resourceConfig =
          IREE::Stream::ResourceConfigAttr::lookup(constantsOp);

      OpBuilder builder(constantsOp);
      IndexSet indexSet(constantsOp.getLoc(), builder);
      indexSet.populate(constantsOp.getResultSizes());

      // Perform upload/processing for immutable and mutable constants.
      SmallVector<Value> timepoints;
      if (auto timepoint =
              generateUpload(constantsOp, IREE::Stream::Lifetime::Constant,
                             resourceConfig, indexSet, builder)) {
        timepoints.push_back(timepoint);
      }
      if (auto timepoint =
              generateUpload(constantsOp, IREE::Stream::Lifetime::Variable,
                             resourceConfig, indexSet, builder)) {
        timepoints.push_back(timepoint);
      }
      if (timepoints.empty()) return;

      // Join on storage timepoints for our transitive dependencies to await.
      // We could do this at a finer granularity if we were to split the
      // constants op into multiple units earlier on.
      Value joinTimepoint;
      if (timepoints.size() > 1) {
        joinTimepoint = builder.create<IREE::Stream::TimepointJoinOp>(
            constantsOp.getLoc(), timepoints.front().getType(), timepoints);
      } else {
        joinTimepoint = timepoints.front();
      }
      constantsOp.getResultTimepoint().replaceAllUsesWith(joinTimepoint);

      constantsOp.erase();
    });
  }
};

}  // namespace

std::unique_ptr<InterfacePass<CallableOpInterface>> createPackConstantsPass() {
  return std::make_unique<PackConstantsPass>();
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
