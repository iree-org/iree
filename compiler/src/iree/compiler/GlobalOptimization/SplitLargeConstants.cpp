// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "iree-global-opt-split-large-constants"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_SPLITLARGECONSTANTSPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

/// How an oversized global is cut into shards: consecutive ranges of
/// `splitDim`. All dimensions before `splitDim` have extent 1, so every shard
/// is a contiguous byte range of the original value.
struct SplitPlan {
  RankedTensorType type;
  int64_t splitDim = 0;
  // Bytes covered by one index of `splitDim`.
  int64_t sliceBytes = 0;
  // Offset and extent along `splitDim` of each shard.
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> sizes;

  int64_t getNumShards() const { return offsets.size(); }
  RankedTensorType getShardType(int64_t shard) const {
    SmallVector<int64_t> shape(type.getShape());
    shape[splitDim] = sizes[shard];
    return RankedTensorType::get(shape, type.getElementType(),
                                 type.getEncoding());
  }
};

/// A use of the oversized value as an input operand of a structured op. The
/// op is split along `loopDim`, the loop that indexes `splitDim` of the value.
struct OperandUse {
  TilingInterface op;
  unsigned operandNumber;
  unsigned loopDim;
  bool isReduction;
};

/// A use of the oversized value captured in the body of an elementwise
/// `linalg.generic` and read with one `tensor.extract`, e.g. an embedding
/// lookup.
struct GatherUse {
  linalg::GenericOp op;
  tensor::ExtractOp extractOp;
};

struct UsePlan {
  SmallVector<OperandUse> operandUses;
  SmallVector<GatherUse> gatherUses;
};

/// Returns the bytes of the initial value, or std::nullopt if it is not an
/// inline dense value whose bytes can be sliced along the outer dimension.
/// Named parameters are not inline; they are split by referring to byte ranges
/// of the parameter instead (see buildShardParameter).
static std::optional<ArrayRef<char>> getRawBytes(Attribute value) {
  if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(value)) {
    const AsmResourceBlob *blob = resourceAttr.getRawHandle().getBlob();
    if (!blob) {
      return std::nullopt;
    }
    return blob->getData();
  }
  if (auto denseAttr = dyn_cast<DenseElementsAttr>(value)) {
    if (denseAttr.isSplat()) {
      return ArrayRef<char>();
    }
    return denseAttr.getRawData();
  }
  return std::nullopt;
}

/// Returns how to split `globalOp`, or std::nullopt when it fits into one
/// allocation or cannot be split (a warning is emitted for the latter).
static std::optional<SplitPlan> planSplit(IREE::Util::GlobalOp globalOp) {
  auto type = dyn_cast<RankedTensorType>(globalOp.getType());
  if (!type || !type.hasStaticShape()) {
    return std::nullopt;
  }
  int64_t storageSize = IREE::Util::getRoundedPhysicalStorageSize(type);
  int64_t maxSize =
      IREE::Stream::ResourceConfigAttr::lookup(globalOp).getMaxAllocationSize();
  if (storageSize <= maxSize) {
    return std::nullopt;
  }

  auto warn = [&]() {
    InFlightDiagnostic diag = globalOp.emitWarning();
    diag << "constant of " << storageSize
         << " bytes exceeds the max allocation size of " << maxSize
         << " bytes and is not split: ";
    return diag;
  };
  if (globalOp.isGlobalMutable() || !globalOp.isPrivate()) {
    warn() << "only private immutable globals are split";
    return std::nullopt;
  }
  Attribute initialValue = globalOp.getInitialValueAttr();
  auto parameterAttr =
      dyn_cast_if_present<IREE::Flow::NamedParameterAttr>(initialValue);
  std::optional<ArrayRef<char>> rawBytes =
      initialValue && !parameterAttr ? getRawBytes(initialValue) : std::nullopt;
  if (!parameterAttr && !rawBytes) {
    warn() << "the initial value is neither an inline dense constant nor a "
              "named parameter";
    return std::nullopt;
  }
  if (parameterAttr && parameterAttr.getConfig() &&
      parameterAttr.getConfig().contains("offset") &&
      !parameterAttr.getConfig().getAs<IntegerAttr>("offset")) {
    warn() << "the parameter offset is not an integer";
    return std::nullopt;
  }

  // Only types stored at their natural width, so storage, raw data and shard
  // boundaries all agree byte for byte.
  Type elementType = type.getElementType();
  unsigned bitWidth =
      elementType.isIntOrFloat() ? elementType.getIntOrFloatBitWidth() : 0;
  if (!llvm::is_contained({8u, 16u, 32u, 64u}, bitWidth)) {
    warn() << "element type " << elementType
           << " is not stored at a natural width of 8, 16, 32 or 64 bits";
    return std::nullopt;
  }
  int64_t elementBytes = bitWidth / 8;
  int64_t expectedBytes = type.getNumElements() * elementBytes;
  if (rawBytes && !rawBytes->empty() &&
      static_cast<int64_t>(rawBytes->size()) != expectedBytes) {
    warn() << "the initial value has " << rawBytes->size()
           << " bytes, expected " << expectedBytes;
    return std::nullopt;
  }
  if (parameterAttr && parameterAttr.getStorageSize() != expectedBytes) {
    warn() << "the parameter has " << parameterAttr.getStorageSize()
           << " bytes, expected " << expectedBytes;
    return std::nullopt;
  }

  SplitPlan plan;
  plan.type = type;
  ArrayRef<int64_t> shape = type.getShape();
  const auto *it = llvm::find_if(shape, [](int64_t dim) { return dim != 1; });
  if (it == shape.end()) {
    warn() << "there is no dimension to split";
    return std::nullopt;
  }
  plan.splitDim = std::distance(shape.begin(), it);
  int64_t extent = shape[plan.splitDim];
  plan.sliceBytes = elementBytes;
  for (int64_t dim : shape.drop_front(plan.splitDim + 1)) {
    plan.sliceBytes *= dim;
  }
  if (plan.sliceBytes > maxSize) {
    warn() << "one index of dimension " << plan.splitDim << " alone has "
           << plan.sliceBytes << " bytes";
    return std::nullopt;
  }

  // As few shards as possible, of near-equal size. Balancing the shards never
  // exceeds the extent that fits into one allocation.
  int64_t maxShardExtent = maxSize / plan.sliceBytes;
  int64_t numShards = llvm::divideCeil(extent, maxShardExtent);
  int64_t shardExtent = llvm::divideCeil(extent, numShards);
  for (int64_t offset = 0; offset < extent; offset += shardExtent) {
    plan.offsets.push_back(offset);
    plan.sizes.push_back(std::min(shardExtent, extent - offset));
  }
  return plan;
}

/// Checks that every use of `value` (a load of the global being split) can be
/// rewritten to use the shards. Emits a warning at the first one that cannot.
/// `seenOps` holds the ops already planned for a rewrite. It is shared by all
/// loads of the global, as rewriting an op replaces it, so it can be rewritten
/// for only one use.
static FailureOr<UsePlan>
analyzeUses(Value value, const SplitPlan &plan, IREE::Util::GlobalOp globalOp,
            llvm::SmallDenseSet<Operation *> &seenOps) {
  UsePlan usePlan;
  auto unsupported = [&](Operation *user, const Twine &reason) {
    InFlightDiagnostic diag = user->emitWarning();
    diag << "cannot split the use of oversized constant @"
         << globalOp.getGlobalName().getValue() << ": " << reason;
    diag.attachNote(globalOp.getLoc()) << "constant declared here";
    return failure();
  };

  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();

    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(user)) {
      if (!seenOps.insert(user).second) {
        return unsupported(user, "used by more than one operand");
      }
      if (!linalgOp.hasPureTensorSemantics() || !linalgOp.isDpsInput(&use) ||
          !isa<TilingInterface>(user)) {
        return unsupported(user, "not an input of a tensor op that can tile");
      }
      AffineMap map = linalgOp.getMatchingIndexingMap(&use);
      auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(plan.splitDim));
      if (!dimExpr) {
        return unsupported(user, "the split dimension is not indexed by a "
                                 "single loop");
      }
      unsigned loopDim = dimExpr.getPosition();
      for (auto [index, expr] : llvm::enumerate(map.getResults())) {
        if (index != plan.splitDim && expr.isFunctionOfDim(loopDim)) {
          return unsupported(user, "the split loop indexes more than one "
                                   "dimension of the constant");
        }
      }
      utils::IteratorType iteratorType =
          linalgOp.getIteratorTypesArray()[loopDim];
      bool isReduction = iteratorType == utils::IteratorType::reduction;
      for (OpOperand &init : linalgOp.getDpsInitsMutable()) {
        AffineMap initMap = linalgOp.getMatchingIndexingMap(&init);
        int64_t numUses = llvm::count_if(initMap.getResults(), [&](auto e) {
          return e.isFunctionOfDim(loopDim);
        });
        bool isPlainDim =
            llvm::is_contained(initMap.getResults(),
                               getAffineDimExpr(loopDim, user->getContext()));
        if (isReduction ? numUses != 0 : (numUses != 1 || !isPlainDim)) {
          return unsupported(user, "the split loop is not a plain dimension "
                                   "of the results");
        }
      }
      usePlan.operandUses.push_back({cast<TilingInterface>(user),
                                     use.getOperandNumber(), loopDim,
                                     isReduction});
      continue;
    }

    if (auto extractOp = dyn_cast<tensor::ExtractOp>(user)) {
      auto genericOp = dyn_cast<linalg::GenericOp>(user->getParentOp());
      if (!genericOp || user->getBlock() != genericOp.getBody()) {
        return unsupported(user, "not read directly in a linalg.generic body");
      }
      if (!seenOps.insert(genericOp).second) {
        return unsupported(user, "read more than once in the same op");
      }
      if (!genericOp.hasPureTensorSemantics() ||
          genericOp.getNumDpsInits() != 1 ||
          genericOp.getNumParallelLoops() != genericOp.getNumLoops() ||
          llvm::is_contained(genericOp->getOperands(), value)) {
        return unsupported(user, "the reading op is not an elementwise "
                                 "linalg.generic with one result");
      }
      usePlan.gatherUses.push_back({genericOp, extractOp});
      continue;
    }

    return unsupported(user, "unsupported user");
  }
  return usePlan;
}

/// Builds a parameter that refers to `byteLength` bytes at `byteOffset` within
/// `parameterAttr`: the same scope and key, with the offset added to the one
/// the parameter already has. Any other config entries are kept.
static IREE::Flow::NamedParameterAttr
buildShardParameter(IREE::Flow::NamedParameterAttr parameterAttr,
                    RankedTensorType shardType, int64_t byteOffset,
                    int64_t byteLength) {
  MLIRContext *context = parameterAttr.getContext();
  Builder builder(context);
  NamedAttrList config;
  int64_t baseOffset = 0;
  if (DictionaryAttr configAttr = parameterAttr.getConfig()) {
    config.append(configAttr.getValue());
    if (auto offsetAttr = configAttr.getAs<IntegerAttr>("offset")) {
      baseOffset = offsetAttr.getInt();
    }
  }
  config.set("offset", builder.getI64IntegerAttr(baseOffset + byteOffset));
  config.set("length", builder.getI64IntegerAttr(byteLength));
  return IREE::Flow::NamedParameterAttr::get(
      context, shardType, parameterAttr.getScope(), parameterAttr.getKey(),
      config.getDictionary(context));
}

/// Builds the initial value of one shard.
static TypedAttr buildShardValue(Attribute value, const SplitPlan &plan,
                                 int64_t shard, StringRef blobName) {
  RankedTensorType shardType = plan.getShardType(shard);
  int64_t byteOffset = plan.offsets[shard] * plan.sliceBytes;
  int64_t byteLength = plan.sizes[shard] * plan.sliceBytes;
  if (auto parameterAttr = dyn_cast<IREE::Flow::NamedParameterAttr>(value)) {
    return buildShardParameter(parameterAttr, shardType, byteOffset,
                               byteLength);
  }
  ArrayRef<char> rawBytes = *getRawBytes(value);
  if (rawBytes.empty()) {
    auto splatAttr = cast<DenseElementsAttr>(value);
    return DenseElementsAttr::get(shardType,
                                  splatAttr.getSplatValue<Attribute>());
  }
  ArrayRef<char> shardBytes = rawBytes.slice(byteOffset, byteLength);
  if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(value)) {
    // The shard is a view into the original blob rather than a copy. Erasing
    // the original global does not free its blob: the blob manager keeps it
    // until the context is destroyed, so a copy would double the memory.
    // This relies on nothing replacing the original blob (e.g., registering a
    // new blob using the same name) while the shards live.
    // The view is aligned to the original alignment only as far as its offset
    // allows; it stays aligned to the element type, as the offset is a
    // multiple of a slice of the split dimension.
    llvm::Align alignment = llvm::commonAlignment(
        llvm::Align(resourceAttr.getRawHandle().getBlob()->getDataAlignment()),
        byteOffset);
    return DenseResourceElementsAttr::get(
        shardType, blobName,
        UnmanagedAsmResourceBlob::allocateWithAlign(shardBytes,
                                                    alignment.value()));
  }
  return DenseElementsAttr::getFromRawBuffer(shardType, shardBytes);
}

/// Splits an op that reads the constant as an input operand into one op per
/// shard, iterating over that shard's range of the split loop. For a parallel
/// loop the results are inserted into the original destination; for a
/// reduction loop each op accumulates into the result of the previous one.
static LogicalResult splitOperandUse(RewriterBase &rewriter,
                                     const OperandUse &use, Value value,
                                     const SplitPlan &plan,
                                     ArrayRef<Value> shards) {
  TilingInterface op = use.op;
  Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);
  SmallVector<Range> domain = op.getIterationDomain(rewriter);
  auto dpsOp = cast<DestinationStyleOpInterface>(op.getOperation());
  SmallVector<Value> results = llvm::to_vector(dpsOp.getDpsInits());

  for (int64_t shard = 0; shard < plan.getNumShards(); ++shard) {
    SmallVector<OpFoldResult> offsets, sizes;
    for (const Range &range : domain) {
      offsets.push_back(range.offset);
      sizes.push_back(range.size);
    }
    offsets[use.loopDim] = rewriter.getIndexAttr(plan.offsets[shard]);
    sizes[use.loopDim] = rewriter.getIndexAttr(plan.sizes[shard]);
    FailureOr<TilingResult> tiled =
        op.getTiledImplementation(rewriter, offsets, sizes);
    if (failed(tiled) || tiled->tiledOps.size() != 1) {
      return op.emitError("failed to split along loop ") << use.loopDim;
    }
    Operation *tiledOp = tiled->tiledOps.front();

    // Read the shard instead of the matching slice of the whole constant.
    OpOperand &operand = tiledOp->getOpOperand(use.operandNumber);
    auto sliceOp = operand.get().getDefiningOp<tensor::ExtractSliceOp>();
    if (!sliceOp || sliceOp.getSource() != value) {
      return op.emitError("expected the split op to read a slice of the "
                          "constant");
    }
    // Create the ops that read the shard before the tiled op that uses them.
    rewriter.setInsertionPoint(tiledOp);
    SmallVector<OpFoldResult> sliceOffsets = sliceOp.getMixedOffsets();
    OpFoldResult &offset = sliceOffsets[plan.splitDim];
    if (std::optional<int64_t> constOffset = getConstantIntValue(offset)) {
      offset = rewriter.getIndexAttr(*constOffset - plan.offsets[shard]);
    } else {
      Value base =
          arith::ConstantIndexOp::create(rewriter, loc, plan.offsets[shard]);
      offset = arith::SubIOp::create(rewriter, loc, cast<Value>(offset), base)
                   .getResult();
    }
    Value shardSlice = shards[shard];
    bool isWholeShard = sliceOp.getType() == shardSlice.getType() &&
                        llvm::all_of(sliceOffsets, isZeroInteger) &&
                        llvm::all_of(sliceOp.getMixedStrides(), isOneInteger);
    if (!isWholeShard) {
      shardSlice = tensor::ExtractSliceOp::create(
          rewriter, loc, sliceOp.getType(), shards[shard], sliceOffsets,
          sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
    }
    rewriter.modifyOpInPlace(tiledOp, [&] { operand.set(shardSlice); });
    if (sliceOp.use_empty()) {
      rewriter.eraseOp(sliceOp);
    }
    rewriter.setInsertionPointAfter(tiledOp);

    if (use.isReduction) {
      // The results do not depend on the split loop, so each shard's op takes
      // the whole destination: chain them.
      if (shard > 0) {
        auto tiledDpsOp = cast<DestinationStyleOpInterface>(tiledOp);
        for (auto [init, result] :
             llvm::zip_equal(tiledDpsOp.getDpsInitsMutable(), results)) {
          if (init.get().getType() != result.getType()) {
            return op.emitError("expected a reduction to keep the whole "
                                "destination");
          }
          Value oldInit = init.get();
          rewriter.modifyOpInPlace(tiledOp, [&, &init = init, result = result] {
            init.set(result);
          });
          if (auto oldSlice = oldInit.getDefiningOp<tensor::ExtractSliceOp>();
              oldSlice && oldSlice.use_empty()) {
            rewriter.eraseOp(oldSlice);
          }
        }
      }
      results = llvm::to_vector(tiled->tiledValues);
      continue;
    }
    for (auto [index, tiledValue] : llvm::enumerate(tiled->tiledValues)) {
      SmallVector<OpFoldResult> resultOffsets, resultSizes;
      if (failed(op.getResultTilePosition(rewriter, index, offsets, sizes,
                                          resultOffsets, resultSizes))) {
        return op.emitError("failed to compute the position of result ")
               << index;
      }
      SmallVector<OpFoldResult> strides(resultOffsets.size(),
                                        rewriter.getIndexAttr(1));
      results[index] = tensor::InsertSliceOp::create(
          rewriter, loc, tiledValue, results[index], resultOffsets, resultSizes,
          strides);
    }
  }
  rewriter.replaceOp(op, results);
  return success();
}

/// Rewrites an elementwise op that reads the constant at a computed index (a
/// gather) into a chain of ops, one per shard. Shard k reads its own shard at
/// the clamped, rebased index and keeps the previous op's element unless the
/// index falls into shard k. Elementwise fusion may merge the chain into one
/// dispatch again; each shard stays a binding within the allocation limit.
static void splitGatherUse(RewriterBase &rewriter, GatherUse use,
                           const SplitPlan &plan, ArrayRef<Value> shards) {
  linalg::GenericOp op = use.op;
  Location loc = op.getLoc();
  unsigned numInputs = op.getNumDpsInputs();
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  AffineMap initMap = maps[numInputs];
  Type elementType = getElementTypeOrSelf(op.getResult(0).getType());

  Value previous;
  for (int64_t shard = 0; shard < plan.getNumShards(); ++shard) {
    rewriter.setInsertionPoint(op);
    SmallVector<Value> inputs(op.getDpsInputs());
    SmallVector<AffineMap> shardMaps(maps.begin(), maps.begin() + numInputs);
    if (previous) {
      inputs.push_back(previous);
      shardMaps.push_back(initMap);
    }
    shardMaps.push_back(initMap);
    auto shardOp = linalg::GenericOp::create(
        rewriter, loc, op.getResultTypes(), inputs, op.getDpsInits(), shardMaps,
        op.getIteratorTypesArray());
    shardOp->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    IRMapping mapping;
    rewriter.cloneRegionBefore(op.getRegion(), shardOp.getRegion(),
                               shardOp.getRegion().end(), mapping);
    Block *body = shardOp.getBody();
    BlockArgument previousArg;
    if (previous) {
      previousArg = body->insertArgument(numInputs, elementType, loc);
    }

    auto extractOp = cast<tensor::ExtractOp>(
        mapping.lookup(use.extractOp.getResult()).getDefiningOp());
    rewriter.setInsertionPoint(extractOp);
    SmallVector<Value> indices(extractOp.getIndices());
    Value index = indices[plan.splitDim];
    if (plan.offsets[shard] != 0) {
      Value base =
          arith::ConstantIndexOp::create(rewriter, loc, plan.offsets[shard]);
      index = arith::SubIOp::create(rewriter, loc, index, base);
    }
    Value last =
        arith::ConstantIndexOp::create(rewriter, loc, plan.sizes[shard] - 1);
    indices[plan.splitDim] = arith::MinUIOp::create(rewriter, loc, index, last);
    Value inShard;
    if (previous) {
      Value size =
          arith::ConstantIndexOp::create(rewriter, loc, plan.sizes[shard]);
      inShard = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                                      index, size);
    }
    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(extractOp, shards[shard],
                                                   indices);
    if (previous) {
      auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
      rewriter.setInsertionPoint(yieldOp);
      Value selected = arith::SelectOp::create(
          rewriter, loc, inShard, yieldOp.getOperand(0), previousArg);
      rewriter.modifyOpInPlace(yieldOp,
                               [&] { yieldOp->setOperand(0, selected); });
    }

    previous = shardOp.getResult(0);
  }
  rewriter.replaceOp(op, previous);
}

struct SplitLargeConstantsPass
    : impl::SplitLargeConstantsPassBase<SplitLargeConstantsPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);
    IRRewriter rewriter(&getContext());
    for (auto globalOp :
         llvm::to_vector(moduleOp.getOps<IREE::Util::GlobalOp>())) {
      if (failed(splitGlobal(rewriter, symbolTable, globalOp))) {
        return signalPassFailure();
      }
    }
  }

  LogicalResult splitGlobal(RewriterBase &rewriter, SymbolTable &symbolTable,
                            IREE::Util::GlobalOp globalOp) {
    std::optional<SplitPlan> plan = planSplit(globalOp);
    if (!plan) {
      return success();
    }

    // Check every use before changing anything, so that a constant is either
    // fully split or left alone.
    SmallVector<IREE::Util::GlobalLoadOp> loadOps;
    if (auto uses = SymbolTable::getSymbolUses(globalOp, getOperation())) {
      for (const SymbolTable::SymbolUse &use : *uses) {
        auto loadOp = dyn_cast<IREE::Util::GlobalLoadOp>(use.getUser());
        if (!loadOp) {
          use.getUser()->emitWarning("cannot split the use of oversized "
                                     "constant @")
              << globalOp.getGlobalName().getValue()
              << ": only util.global.load is supported";
          return success();
        }
        loadOps.push_back(loadOp);
      }
    }
    SmallVector<UsePlan> usePlans;
    llvm::SmallDenseSet<Operation *> seenOps;
    for (IREE::Util::GlobalLoadOp loadOp : loadOps) {
      FailureOr<UsePlan> usePlan =
          analyzeUses(loadOp.getResult(), *plan, globalOp, seenOps);
      if (failed(usePlan)) {
        return success();
      }
      usePlans.push_back(std::move(*usePlan));
    }
    LLVM_DEBUG(llvm::dbgs()
               << "splitting @" << globalOp.getGlobalName() << " into "
               << plan->getNumShards() << " shards along dimension "
               << plan->splitDim << "\n");

    SmallVector<IREE::Util::GlobalOp> shardOps;
    rewriter.setInsertionPointAfter(globalOp);
    for (int64_t shard = 0; shard < plan->getNumShards(); ++shard) {
      auto shardOp = cast<IREE::Util::GlobalOp>(rewriter.clone(*globalOp));
      SymbolTable::setSymbolName(shardOp, (globalOp.getGlobalName().getValue() +
                                           "_shard" + Twine(shard))
                                              .str());
      symbolTable.insert(shardOp);
      shardOp.setType(plan->getShardType(shard));
      shardOp.setInitialValueAttr(
          buildShardValue(globalOp.getInitialValueAttr(), *plan, shard,
                          shardOp.getGlobalName().getValue()));
      shardOps.push_back(shardOp);
      rewriter.setInsertionPointAfter(shardOp);
    }

    for (auto [loadOp, usePlan] : llvm::zip_equal(loadOps, usePlans)) {
      rewriter.setInsertionPoint(loadOp);
      SmallVector<Value> shards;
      for (IREE::Util::GlobalOp shardOp : shardOps) {
        auto shardLoadOp = IREE::Util::GlobalLoadOp::create(
            rewriter, loadOp.getLoc(), shardOp);
        shardLoadOp.setIsImmutable(loadOp.getIsImmutable());
        shards.push_back(shardLoadOp.getResult());
      }
      for (const OperandUse &use : usePlan.operandUses) {
        if (failed(splitOperandUse(rewriter, use, loadOp.getResult(), *plan,
                                   shards))) {
          return failure();
        }
      }
      for (const GatherUse &use : usePlan.gatherUses) {
        splitGatherUse(rewriter, use, *plan, shards);
      }
      for (Value shard : shards) {
        if (shard.use_empty()) {
          rewriter.eraseOp(shard.getDefiningOp());
        }
      }
      rewriter.eraseOp(loadOp);
    }
    symbolTable.erase(globalOp);
    return success();
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
