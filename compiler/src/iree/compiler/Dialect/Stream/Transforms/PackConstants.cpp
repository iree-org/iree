// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Utils/IndexSet.h"
#include "llvm/Support/Debug.h"
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

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_PACKCONSTANTSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

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
    if (auto storageAttr = dyn_cast<IREE::Util::SizedStorageAttr>(value)) {
      return storageAttr.getStorageSize();
    } else {
      assert(false && "invalid constant attr type");
      return 0;
    }
  }
};

struct PackedSpan {
  // Original slice this span represents.
  ConstantSlice slice;
  // Byte offset within the target storage buffer.
  uint64_t offset = 0;
  // Length of the valid data when padded out in the target storage buffer.
  // This is only accounting for the padding of the valid data itself and not
  // any additional padding for other spans within the buffer (like start
  // offset alignment).
  uint64_t length = 0;
};

// A storage resource backed by data packed in storage.
// A storage resource may be comprised of data packed by the compiler into its
// at-rest form (allowing for easy single operation loads/mappings) or gathered
// from disparate storage locations (more expensive). Degenerate resources
// may only contain a single packed logical resource which allows for easier
// parameter loading with zero-copies at the cost of more runtime overhead.
struct StorageResource {
  // Fused location of all spans that make up this storage buffer.
  Location loc;
  // Total size in bytes (including padding).
  uint64_t totalSize = 0;
  // Constant spans packed into this resource.
  SmallVector<PackedSpan> spans;
  // Packed byte data that must be embedded in the final module.
  // It must be written with an alignment as required by the constraints.
  // If not set then each span may have unique storage.
  IREE::Util::CompositeAttr packedData;
};

// Buckets |slices| into 1+ storage resources based on |resourceConfig|.
static SmallVector<StorageResource> bucketValuesIntoStorageResources(
    ArrayRef<ConstantSlice> slices,
    IREE::Stream::ResourceConfigAttr resourceConfig) {
  // TODO(benvanik): replace with a better strategy (best-fit, etc).
  SmallVector<StorageResource> storageBuffers;
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
  storageBuffer.loc = FusedLoc::get(
      context,
      llvm::map_to_vector<8>(storageBuffer.spans, [](PackedSpan &span) {
        return span.slice.result.getLoc();
      }));

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
    if (constantSpan.length == 0)
      continue;

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

  storageBuffer.packedData = IREE::Util::CompositeAttr::get(context, values);
  assert(storageBuffer.packedData && "unable to build composite attr");
}

// Returns zero or more storage resources and the spans values map into.
// Assume that |slices| have been ordered by prior passes and that order may
// have some performance-sensitivity (constants are grouped by
// locality/lifetime/etc).
static SmallVector<StorageResource>
computePackingMap(ArrayRef<ConstantSlice> slices,
                  IREE::Stream::ResourceConfigAttr resourceConfig,
                  MLIRContext *context) {
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

struct TimepointResource {
  Value timepoint;
  Value resource;
  Value resourceSize;
};

struct ParameterSlice {
  IREE::Stream::NamedParameterAttr parameterAttr;
  Value sourceOffset;
  Value sourceLength;
};

static ParameterSlice getParameterSlice(Location loc, Attribute value,
                                        IndexSet &indexSet,
                                        OpBuilder &builder) {
  auto parameterAttr = cast<IREE::Stream::NamedParameterAttr>(value);
  Value sourceOffset;
  Value sourceLength;
  if (auto configAttr = parameterAttr.getConfig()) {
    if (auto offsetAttr = configAttr.getAs<IntegerAttr>("offset")) {
      sourceOffset =
          builder.create<arith::ConstantIntOp>(loc, offsetAttr.getInt(), 64);
    }
    if (auto lengthAttr = configAttr.getAs<IntegerAttr>("length")) {
      sourceLength = indexSet.get(lengthAttr.getInt());
    }
  }
  if (!sourceOffset)
    sourceOffset = builder.create<arith::ConstantIntOp>(loc, 0, 64);
  if (!sourceLength)
    sourceLength = indexSet.get(parameterAttr.getStorageSize());
  return ParameterSlice{parameterAttr, sourceOffset, sourceLength};
}

static Value buildParameterLoad(Value awaitTimepoint,
                                IREE::Stream::AffinityAttr affinityAttr,
                                Type targetType, StringAttr scope,
                                ArrayRef<StorageResource *> storageResources,
                                IndexSet &indexSet, OpBuilder &builder) {
  SmallVector<Location> spanLocs;
  SmallVector<Attribute> sourceKeys;
  SmallVector<Value> sourceOffsets;
  SmallVector<Type> targetTypes;
  SmallVector<Value> targetLengths;
  for (auto *storageResource : storageResources) {
    assert(storageResource->spans.size() == 1 &&
           "expected single span per resource for load");
    for (auto &packedSpan : storageResource->spans) {
      auto spanLoc = packedSpan.slice.result.getLoc();
      auto parameterSlice =
          getParameterSlice(spanLoc, packedSpan.slice.value, indexSet, builder);
      spanLocs.push_back(spanLoc);
      sourceKeys.push_back(parameterSlice.parameterAttr.getKey());
      sourceOffsets.push_back(parameterSlice.sourceOffset);
      targetTypes.push_back(targetType);
      targetLengths.push_back(indexSet.get(packedSpan.length));
    }
  }

  // Load all in a batch. One resource is returned per parameter but they may
  // alias depending on the runtime implementation.
  auto loadOp = builder.create<IREE::Stream::ParameterLoadOp>(
      builder.getFusedLoc(spanLocs), targetTypes,
      builder.getType<IREE::Stream::TimepointType>(), scope,
      builder.getArrayAttr(sourceKeys), sourceOffsets, targetLengths,
      awaitTimepoint, affinityAttr);

  // Slice out each span from the allocation.
  // Note that access must be guarded by the final ready timepoint.
  unsigned resultIndex = 0;
  for (auto *storageResource : storageResources) {
    for (auto &packedSpan : storageResource->spans) {
      auto subviewOp = builder.create<IREE::Stream::ResourceSubviewOp>(
          packedSpan.slice.result.getLoc(), loadOp.getResult(resultIndex),
          loadOp.getResultSize(resultIndex), indexSet.get(packedSpan.offset),
          packedSpan.slice.resultSize);
      packedSpan.slice.result.replaceAllUsesWith(subviewOp.getResult());
      ++resultIndex;
    }
  }

  return loadOp.getResultTimepoint();
}

static TimepointResource
buildParameterGather(Location loc, Value awaitTimepoint,
                     IREE::Stream::AffinityAttr affinityAttr, Type targetType,
                     Value targetSize, MutableArrayRef<PackedSpan> packedSpans,
                     IndexSet &indexSet, OpBuilder &builder) {
  // Allocate the resulting storage resource of the final resource type.
  auto allocOp = builder.create<IREE::Stream::ResourceAllocOp>(
      loc, targetType, targetSize,
      /*uninitialized=*/builder.getUnitAttr(), affinityAttr);

  // Parameters may be from multiple scopes - bucket by scope and gather from
  // each in turn.
  llvm::MapVector<StringAttr, SmallVector<PackedSpan>> scopeSpans;
  for (auto &packedSpan : packedSpans) {
    auto parameterAttr =
        cast<IREE::Stream::NamedParameterAttr>(packedSpan.slice.value);
    scopeSpans[parameterAttr.getScope()].push_back(packedSpan);
  }

  // Gather from each unique scope.
  SmallVector<Value> gatherTimepoints;
  for (auto &[scope, packedSpans] : scopeSpans) {
    SmallVector<Attribute> sourceKeys;
    SmallVector<Value> sourceOffsets;
    SmallVector<Value> targetOffsets;
    SmallVector<Value> targetLengths;
    sourceKeys.reserve(packedSpans.size());
    for (auto &packedSpan : packedSpans) {
      auto parameterSlice =
          getParameterSlice(loc, packedSpan.slice.value, indexSet, builder);
      sourceKeys.push_back(parameterSlice.parameterAttr.getKey());
      sourceOffsets.push_back(parameterSlice.sourceOffset);
      targetOffsets.push_back(indexSet.get(packedSpan.offset));
      targetLengths.push_back(indexSet.get(packedSpan.length));
    }
    auto gatherOp = builder.create<IREE::Stream::ParameterGatherOp>(
        loc, builder.getType<IREE::Stream::TimepointType>(), scope,
        builder.getArrayAttr(sourceKeys), sourceOffsets, allocOp.getResult(),
        allocOp.getResultSize(0), targetOffsets, targetLengths, awaitTimepoint,
        affinityAttr);
    gatherTimepoints.push_back(gatherOp.getResultTimepoint());
  }

  // Slice out each span from the allocation.
  // Note that access must be guarded by the final ready timepoint.
  for (auto &packedSpan : packedSpans) {
    auto subviewOp = builder.create<IREE::Stream::ResourceSubviewOp>(
        packedSpan.slice.result.getLoc(), allocOp.getResult(),
        allocOp.getResultSize(0), indexSet.get(packedSpan.offset),
        packedSpan.slice.resultSize);
    packedSpan.slice.result.replaceAllUsesWith(subviewOp.getResult());
  }

  // Wait until all gathers have completed.
  Value readyTimepoint =
      IREE::Stream::TimepointJoinOp::join(gatherTimepoints, builder);
  return TimepointResource{readyTimepoint, allocOp.getResult(),
                           allocOp.getResultSize(0)};
}

static TimepointResource buildFileRead(Location loc, Value awaitTimepoint,
                                       IREE::Stream::AffinityAttr affinityAttr,
                                       IREE::Stream::ResourceType resourceType,
                                       Value storageResourceSize,
                                       Value storageBuffer,
                                       Value storageBufferSize,
                                       IndexSet &indexSet, OpBuilder &builder) {
  // Allocate the resulting storage resource of the final resource type.
  auto allocOp = builder.create<IREE::Stream::ResourceAllocOp>(
      loc, resourceType, storageResourceSize,
      /*uninitialized=*/builder.getUnitAttr(), affinityAttr);

  // Create the file backed by the constant resource buffer.
  auto fileOp = builder.create<IREE::Stream::FileConstantOp>(
      loc, storageBuffer, storageBufferSize, indexSet.get(0),
      storageResourceSize, affinityAttr);

  // Issue asynchronous file read into the buffer.
  auto zeroI64 = builder.create<arith::ConstantIntOp>(loc, 0, 64);
  auto readOp = builder.create<IREE::Stream::FileReadOp>(
      loc, fileOp.getResult(), zeroI64, allocOp.getResult(),
      allocOp.getResultSize(0), indexSet.get(0), storageResourceSize,
      awaitTimepoint, affinityAttr);

  return TimepointResource{readOp.getResultTimepoint(), readOp.getTarget(),
                           readOp.getTargetSize()};
}

// Emits IR to first try mapping the storage resource directly into a usable
// constant resource. If the mapping fails (the target can't use the memory)
// then fall back to staging uploads.
// Returns a timepoint indicating the operation has completed.
static TimepointResource buildTryMapConstantResource(
    Location loc, Value awaitTimepoint, IREE::Stream::AffinityAttr affinityAttr,
    IREE::Stream::ResourceType resourceType, Value storageResourceSize,
    Value storageBuffer, Value storageBufferSize, IndexSet &indexSet,
    OpBuilder &builder) {
  // Try mapping; this may fail if the device can't use the storage buffer as
  // the type of resource requested.
  auto tryMapOp = builder.create<IREE::Stream::ResourceTryMapOp>(
      loc, builder.getI1Type(), resourceType, storageBuffer, indexSet.get(0),
      storageResourceSize, affinityAttr);

  // If we are able to directly map the resources then we don't need to wait.
  // Otherwise we need to stage the storage buffer into memory via the file
  // streaming API.
  auto ifOp = builder.create<scf::IfOp>(
      loc, tryMapOp.getDidMap(),
      [&](OpBuilder &thenBuilder, Location loc) {
        // Just return the resources + an immediate timepoint.
        thenBuilder.create<scf::YieldOp>(loc, ValueRange{
                                                  awaitTimepoint,
                                                  tryMapOp.getResult(),
                                              });
      },
      [&](OpBuilder &elseBuilder, Location loc) {
        auto readResult =
            buildFileRead(loc, awaitTimepoint, affinityAttr, resourceType,
                          storageResourceSize, storageBuffer, storageBufferSize,
                          indexSet, elseBuilder);
        elseBuilder.create<scf::YieldOp>(loc, ValueRange{
                                                  readResult.timepoint,
                                                  readResult.resource,
                                              });
      });
  auto ifTimepoint = ifOp.getResults().front();
  auto ifResource = ifOp.getResults().back();
  return TimepointResource{ifTimepoint, ifResource, storageResourceSize};
}

static Value generateSerializedUpload(
    Value awaitTimepoint, IREE::Stream::AffinityAttr affinityAttr,
    IREE::Stream::ResourceConfigAttr resourceConfig,
    ArrayRef<ConstantSlice> slices, IndexSet &indexSet, OpBuilder &builder) {
  // Perform the packing of dense values to compute the storage resources we
  // will need and where each value will be placed.
  auto storageResources =
      computePackingMap(slices, resourceConfig, builder.getContext());
  if (storageResources.empty())
    return nullptr;

  // TODO(benvanik): should be able to have a single buffer constant and
  // subrange it so that we don't need so many files.

  auto anyResult = slices.front().result;
  auto resourceType =
      llvm::cast<IREE::Stream::ResourceType>(anyResult.getType());

  // Emit rodata storage for the constant values.
  // As our upload paths may vary this ensures that we are only emitting
  // them once regardless of how many strategies we emit IR for.
  Value currentTimepoint = awaitTimepoint;
  for (auto &storageResource : storageResources) {
    // Serialized resources are stored as packed host data.
    Value storageBuffer = builder.create<IREE::Util::BufferConstantOp>(
        storageResource.loc, /*name=*/nullptr, storageResource.packedData,
        builder.getIndexAttr(resourceConfig.getMinBufferOffsetAlignment()),
        /*mimeType=*/nullptr);

    // If this is producing constants (vs variables) we can try to go on a
    // fast-path where we directly map the constant memory. If producing
    // variables then we always need to stage and clone.
    TimepointResource uploadedResource;
    auto resourceSize = indexSet.get(storageResource.totalSize);
    if (resourceType.getLifetime() == IREE::Stream::Lifetime::Constant) {
      uploadedResource = buildTryMapConstantResource(
          storageResource.loc, currentTimepoint, affinityAttr, resourceType,
          resourceSize, storageBuffer, resourceSize, indexSet, builder);
    } else {
      uploadedResource = buildFileRead(
          storageResource.loc, currentTimepoint, affinityAttr, resourceType,
          resourceSize, storageBuffer, resourceSize, indexSet, builder);
    }

    for (auto &span : storageResource.spans) {
      auto loc = span.slice.result.getLoc();
      auto subviewOp = builder.create<IREE::Stream::ResourceSubviewOp>(
          loc, uploadedResource.resource, uploadedResource.resourceSize,
          indexSet.get(span.offset), span.slice.resultSize);
      span.slice.result.replaceAllUsesWith(subviewOp.getResult());
    }

    currentTimepoint = uploadedResource.timepoint;
  }

  // Join on storage timepoints for our transitive dependencies to await.
  return currentTimepoint;
}

static Value generateParameterUpload(
    Value awaitTimepoint, IREE::Stream::AffinityAttr affinityAttr,
    IREE::Stream::ResourceConfigAttr resourceConfig,
    ArrayRef<ConstantSlice> slices, IndexSet &indexSet, OpBuilder &builder) {
  auto anyResult = slices.front().result;
  auto resourceType =
      llvm::cast<IREE::Stream::ResourceType>(anyResult.getType());

  // Perform the packing of dense values to compute the storage resources we
  // will need and where each value will be placed unless we have a chance to
  // reuse parameter storage. This is a big switch today (either we try to
  // emit one resource per parameter for loading _or_ we gather everything) but
  // could be refined to only try loading large resources while we pack the
  // small resources. e.g. try to reuse a 1GB parameter but pack 1000 128B
  // parameters together.
  SmallVector<StorageResource> storageResources;
  if (resourceType.getLifetime() == IREE::Stream::Lifetime::Constant &&
      resourceConfig.getMemoryModel() == IREE::Stream::MemoryModel::Unified) {
    for (auto &slice : slices) {
      uint64_t sliceSize = slice.getStorageSize();
      storageResources.push_back(StorageResource{slice.result.getLoc(),
                                                 sliceSize,
                                                 {
                                                     PackedSpan{
                                                         slice,
                                                         /*offset=*/0,
                                                         /*length=*/sliceSize,
                                                     },
                                                 }});
    }
  } else {
    storageResources =
        computePackingMap(slices, resourceConfig, builder.getContext());
  }
  if (storageResources.empty())
    return nullptr;

  // Sort resources by type so we can batch them.
  // Loads are only possible if we are using the parameter as a constant and
  // it is a single span as we can't pack externally owned parameters.
  // A batch of loads happens from a single source scope so we bucket here.
  // Note that we do this separate from the walk above as we may pack parameters
  // such that they have a single parameter per resource and introduce more that
  // we can load than if just looking at the original pre-packed state.
  llvm::MapVector<StringAttr, SmallVector<StorageResource *>> resourceLoads;
  SmallVector<StorageResource *> resourceGathers;
  for (auto &storageResource : storageResources) {
    if (storageResource.spans.size() == 1) {
      auto parameterAttr = cast<IREE::Stream::NamedParameterAttr>(
          storageResource.spans.front().slice.value);
      resourceLoads[parameterAttr.getScope()].push_back(&storageResource);
    } else {
      resourceGathers.push_back(&storageResource);
    }
  }

  // Emit all loads as a single operation per scope.
  SmallVector<Value> uploadTimepoints;
  for (auto &[scope, scopeResources] : resourceLoads) {
    uploadTimepoints.push_back(
        buildParameterLoad(awaitTimepoint, affinityAttr, resourceType, scope,
                           scopeResources, indexSet, builder));
  }

  // Emit gathers, of which there may be multiple batches based on the target
  // resource as gathers are 1:1 per target.
  for (auto *storageResource : resourceGathers) {
    auto resourceSize = indexSet.get(storageResource->totalSize);
    auto uploadedResource = buildParameterGather(
        storageResource->loc, awaitTimepoint, affinityAttr, resourceType,
        resourceSize, storageResource->spans, indexSet, builder);
    uploadTimepoints.push_back(uploadedResource.timepoint);
  }

  // Join on storage timepoints for our transitive dependencies to await.
  return IREE::Stream::TimepointJoinOp::join(uploadTimepoints, builder);
}

static Value generateUploads(Value awaitTimepoint,
                             IREE::Stream::ResourceConstantsOp constantsOp,
                             IREE::Stream::ResourceConfigAttr resourceConfig,
                             IndexSet &indexSet, OpBuilder &builder) {
  // Split the slices based on whether they are sourced from serialized data or
  // externally-defined parameters.
  // TODO(benvanik): remove stream.resource.constants and this coupling;
  // parameters should be handled by a dedicated pass. This is a hack that
  // allows us to reuse the packing code for performing variable parameter packs
  // and have everything happen atomically but is pretty terrible.
  SmallVector<ConstantSlice> serializedSlices;
  SmallVector<ConstantSlice> parameterSlices;
  for (auto [result, resultSize, value] :
       llvm::zip_equal(constantsOp.getResults(), constantsOp.getResultSizes(),
                       constantsOp.getValues())) {
    auto slice = ConstantSlice{
        result,
        resultSize,
        value,
    };
    if (isa<IREE::Stream::NamedParameterAttr>(value)) {
      parameterSlices.push_back(slice);
    } else {
      serializedSlices.push_back(slice);
    }
  }

  SmallVector<Value> uploadTimepoints;
  if (!serializedSlices.empty()) {
    uploadTimepoints.push_back(generateSerializedUpload(
        awaitTimepoint, constantsOp.getAffinityAttr(), resourceConfig,
        serializedSlices, indexSet, builder));
  }
  if (!parameterSlices.empty()) {
    uploadTimepoints.push_back(generateParameterUpload(
        awaitTimepoint, constantsOp.getAffinityAttr(), resourceConfig,
        parameterSlices, indexSet, builder));
  }
  return IREE::Stream::TimepointJoinOp::join(uploadTimepoints, builder);
}

//===----------------------------------------------------------------------===//
// --iree-stream-pack-constants
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

struct PackConstantsPass
    : public IREE::Stream::impl::PackConstantsPassBase<PackConstantsPass> {
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

      // Packing creates a lot of index values given that all the sizes are
      // statically-known - CSE would collapse them but we use an IndexSet to
      // reduce the IR churn.
      OpBuilder builder(constantsOp);
      IndexSet indexSet(constantsOp.getLoc(), builder);
      indexSet.populate(constantsOp.getResultSizes());

      // Perform upload/processing for immutable and mutable constants.
      Value awaitTimepoint = builder.create<IREE::Stream::TimepointImmediateOp>(
          constantsOp.getLoc());
      auto uploadTimepoint = generateUploads(awaitTimepoint, constantsOp,
                                             resourceConfig, indexSet, builder);
      constantsOp.getResultTimepoint().replaceAllUsesWith(uploadTimepoint);

      constantsOp.erase();
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
