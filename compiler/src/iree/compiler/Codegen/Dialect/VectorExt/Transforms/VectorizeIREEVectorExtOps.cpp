// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::VectorExt {

#define GEN_PASS_DEF_VECTORIZEIREEVECTOREXTOPSPASS
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h.inc"

namespace {

struct VectorizeToLayoutOpPattern final
    : OpRewritePattern<IREE::VectorExt::ToLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  vector::TransferReadOp
  createReadOp(PatternRewriter &rewriter,
               IREE::VectorExt::ToLayoutOp toLayoutOp) const {
    Location loc = toLayoutOp.getLoc();
    ShapedType inputTy = toLayoutOp.getType();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto identityMap = rewriter.getMultiDimIdentityMap(inputTy.getRank());
    SmallVector<int64_t> readShape =
        toLayoutOp.getLayout().getUndistributedShape();
    Value mask = nullptr;
    if (!toLayoutOp.getType().hasStaticShape()) {
      SmallVector<OpFoldResult> mixedSourceDims =
          tensor::getMixedSizes(rewriter, loc, toLayoutOp.getInput());
      auto maskType = VectorType::get(readShape, rewriter.getI1Type());
      mask =
          rewriter.create<vector::CreateMaskOp>(loc, maskType, mixedSourceDims);
    }
    VectorType vectorType =
        VectorType::get(readShape, inputTy.getElementType());
    auto inBounds = rewriter.getBoolArrayAttr(
        SmallVector<bool>(vectorType.getRank(), true));
    auto padValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(inputTy.getElementType()));
    auto read = rewriter.create<vector::TransferReadOp>(
        loc,
        /*type=*/vectorType,
        /*source=*/toLayoutOp.getInput(),
        /*indices=*/ValueRange{SmallVector<Value>(readShape.size(), zero)},
        /*permutation_map=*/identityMap,
        /*padding=*/padValue,
        /*mask=*/mask,
        /*in_bounds=*/inBounds);
    return read;
  }

  vector::TransferWriteOp
  createWriteOp(PatternRewriter &rewriter,
                IREE::VectorExt::ToLayoutOp tensorLayoutOp,
                Value vectorLayoutOp, Value mask) const {
    Location loc = tensorLayoutOp.getLoc();
    ShapedType tensorTy = tensorLayoutOp.getType();
    auto resType =
        RankedTensorType::get(tensorTy.getShape(), tensorTy.getElementType());
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    int64_t rank = tensorTy.getShape().size();
    auto inBounds = rewriter.getBoolArrayAttr(SmallVector<bool>(rank, true));
    auto identityMap = rewriter.getMultiDimIdentityMap(tensorTy.getRank());
    auto empty = rewriter.create<tensor::EmptyOp>(
        loc, tensor::getMixedSizes(rewriter, loc, tensorLayoutOp.getInput()),
        tensorTy.getElementType());
    return rewriter.create<vector::TransferWriteOp>(
        loc,
        /*result=*/resType,
        /*vector=*/vectorLayoutOp,
        /*source=*/empty,
        /*indices=*/ValueRange{SmallVector<Value>(rank, zero)},
        /*permutation_map=*/identityMap,
        /*mask=*/mask,
        /*inBounds=*/inBounds);
  }

  LogicalResult matchAndRewrite(IREE::VectorExt::ToLayoutOp toLayoutOp,
                                PatternRewriter &rewriter) const override {
    if (!toLayoutOp.hasTensorSemantics()) {
      return failure();
    }
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(toLayoutOp);
    Location loc = toLayoutOp.getLoc();
    vector::TransferReadOp readOp = createReadOp(rewriter, toLayoutOp);
    // Create the toLayout operation but with vector types instead.
    auto newLayoutOp = rewriter.create<IREE::VectorExt::ToLayoutOp>(
        loc, readOp, toLayoutOp.getLayout(), toLayoutOp.getMmaKindAttr(),
        toLayoutOp.getSharedMemoryConversion());
    // Create the write back to a tensor.
    vector::TransferWriteOp writeOp =
        createWriteOp(rewriter, toLayoutOp, newLayoutOp, readOp.getMask());
    rewriter.replaceOp(toLayoutOp, writeOp);
    return success();
  }
};

struct VectorizeIREEVectorExtOpsPass final
    : impl::VectorizeIREEVectorExtOpsPassBase<VectorizeIREEVectorExtOpsPass> {
  void runOnOperation() override {

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<VectorizeToLayoutOpPattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

static linalg::GenericOp
buildPartialGenericOp(RewriterBase &rewriter, linalg::GenericOp fullOp,
                      ArrayRef<int64_t> vectorSizes,
                      SmallVector<Operation *> partial,
                      DenseMap<Value, std::pair<Value, AffineMap>> &tmap) {
  // Each value used in the partial body is either outside the operation (use as
  // is), or is defined inside the block (including block arguements). For
  // values defined inside the block, the value will have a tensor and an
  // AffineMap to access the tensor in tmap;

  // Find all values used in partial that are defined inside the block.
  SetVector<Value> newInputs;
  SetVector<Value> newOutputs;
  for (Operation *op : partial) {
    for (Value operand : op->getOperands()) {
      if (operand.getParentBlock() != fullOp.getBody()) {
        continue;
      }

      if (tmap.contains(operand)) {
        newInputs.insert(operand);
      }
    }

    // If a user of the operation is not in partial, it needs to be a result.
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (llvm::find(partial, user) == partial.end()) {
          newOutputs.insert(result);
        }
      }
    }
  }

  SmallVector<Value> ins, outs;
  SmallVector<AffineMap> indexingMaps;
  AffineMap ident = rewriter.getMultiDimIdentityMap(fullOp.getNumLoops());

  for (Value val : newInputs) {
    auto [in, map] = tmap[val];
    ins.push_back(in);
    indexingMaps.push_back(map);
  }

  for (Value val : newOutputs) {
    Value out = rewriter.create<tensor::EmptyOp>(fullOp.getLoc(), vectorSizes,
                                                 getElementTypeOrSelf(val));
    outs.push_back(out);
    indexingMaps.push_back(ident);
  }

  // If the last operation is a yield, add the out operands.
  bool hasYield = !partial.empty() && isa<linalg::YieldOp>(partial.back());
  if (hasYield) {
    for (OpOperand &operand : fullOp.getDpsInitsMutable()) {
      outs.push_back(operand.get());
      indexingMaps.push_back(fullOp.getMatchingIndexingMap(&operand));
    }
  }

  auto newOp = rewriter.create<linalg::GenericOp>(
      fullOp.getLoc(), TypeRange(outs), ins, outs, indexingMaps,
      fullOp.getIteratorTypesArray(),
      [&](OpBuilder &b, Location loc, ValueRange blockArgs) {
        IRMapping localMap;
        for (auto [oldVal, newVal] : llvm::zip_equal(
                 newInputs, blockArgs.take_front(newInputs.size()))) {
          localMap.map(oldVal, newVal);
        }

        if (!hasYield) {
          for (auto [oldVal, newVal] : llvm::zip_equal(
                   newOutputs, blockArgs.take_back(newOutputs.size()))) {
            localMap.map(oldVal, newVal);
          }
        }

        // Clone partial into this region.
        for (Operation *op : partial) {
          b.clone(*op, localMap);
        }

        if (!hasYield) {
          SmallVector<Value> yieldValues = llvm::map_to_vector(
              newOutputs, [&](Value val) { return localMap.lookup(val); });
          b.create<linalg::YieldOp>(loc, yieldValues);
        }
      });

  // Add a entry in tmap for each value in newOutputs.
  for (auto [index, val] : llvm::enumerate(newOutputs)) {
    Value tensor = newOp.getResult(index);
    AffineMap map = indexingMaps[newInputs.size() + index];
    tmap[val] = {tensor, map};
  }

  return newOp;
}

/// Try to find a mapping from the memory space of tensor.extract to the loop
/// iteration space, if possible.
static AffineMap
getIterationSpaceToMemorySpaceMap(linalg::GenericOp genericOp,
                                  tensor::ExtractOp extractOp) {
  // Try to find a mapping from the memory space to the input iteration space.
  // Find backward slice for each index argument to tensor.extract.
  MLIRContext *ctx = extractOp.getContext();
  AffineMap map = AffineMap::get(genericOp.getNumLoops(), 0, ctx);
  for (Value val : extractOp.getIndices()) {
    Value trace = val;
    bool found = false;
    while (true) {
      // If the index is a linalg.index op, this index maps to a loop iteration
      // variable.
      if (auto indexOp = trace.getDefiningOp<linalg::IndexOp>()) {
        map = map.insertResult(getAffineDimExpr(indexOp.getDim(), ctx),
                               map.getNumResults());
        found = true;
        break;
      }

      // If this index is a value defined outside the linalg.generic block, it
      // is a constant value in the loop.
      Operation *defOp = trace.getDefiningOp();
      if (defOp && !genericOp.getBody()->findAncestorOpInBlock(*defOp)) {
        map = map.insertResult(getAffineConstantExpr(0, ctx),
                               map.getNumResults());
        found = true;
        break;
      }

      // If this index is a block argument, the index is a loop iterated
      // variable, and the iteration space of the index is same as the loop
      // iterated variable. However, we restrict to cases where the variable
      // iteration space is 1-D or 0-D, otherwise the Memory -> Loop Iteration
      // Space map will contain mods/floordivS.
      if (auto blockArg = dyn_cast<BlockArgument>(trace)) {
        // If this block argument isn't part of the linalg.generic body, then
        // this is a constant value in the loop.
        if (blockArg.getParentBlock() != genericOp.getBody()) {
          map = map.insertResult(getAffineConstantExpr(0, ctx),
                                 map.getNumResults());
          found = true;
        } else {
          // Allow 0-D and 1-D maps.
          AffineMap indexMap =
              genericOp.getIndexingMapsArray()[blockArg.getArgNumber()];
          if (indexMap.getNumResults() == 0) {
            map = map.insertResult(getAffineConstantExpr(0, ctx),
                                   map.getNumResults());
            found = true;
          } else if (indexMap.getNumResults() == 1) {
            if (auto dimExpr = dyn_cast<AffineDimExpr>(indexMap.getResult(0))) {
              map =
                  map.insertResult(getAffineDimExpr(dimExpr.getPosition(), ctx),
                                   map.getNumResults());
              found = true;
            }
          }
        }
        break;
      }

      // Trace back arith.index_cast ops. This is a commonly occuring case,
      // there may be other cases which we can trace back here (For example,
      // addition with a loop invariant constant).
      if (isa<arith::IndexCastOp>(defOp)) {
        trace = defOp->getOperand(0);
      } else {
        break;
      }
    }

    if (!found) {
      return AffineMap();
    }
  }

  return map;
}

static AffineMap inversePerm(AffineMap map) {
  MLIRContext *context = map.getContext();
  AffineExpr zero = mlir::getAffineConstantExpr(0, context);
  // Start with all the results as 0.
  SmallVector<AffineExpr, 4> exprs(map.getNumInputs(), zero);
  for (unsigned i : llvm::seq(unsigned(0), map.getNumResults())) {
    // Skip zeros from input map. 'exprs' is already initialized to zero.
    if (auto constant = dyn_cast<AffineConstantExpr>(map.getResult(i))) {
      if (constant.getValue() != 0) {
        return AffineMap();
      }
      continue;
    }

    if (isa<AffineDimExpr>(map.getResult(i))) {
      // Reverse each dimension existing in the original map result.
      exprs[map.getDimPosition(i)] = getAffineDimExpr(i, context);
      continue;
    }

    // Fail if the expr is not a constant or a dim expr.
    return AffineMap();
  }
  return AffineMap::get(map.getNumResults(), /*symbolCount=*/0, exprs, context);
}

} // namespace

LogicalResult vectorizeGatherLikeGenericToTransferGather(
    RewriterBase &rewriter, linalg::GenericOp linalgOp,
    ArrayRef<int64_t> vectorSizes, ArrayRef<bool> scalableVecDims,
    bool vectorizeNDExtract) {

  // Since upstream vectorization does not support hooks to vectorize individual
  // operations inside a linalg.generic, we take an alternate approach here,
  // by splitting the generic into 3 operations, anchored around the first
  // tensor.extract operation:
  //
  // 1. pre-extract-generic
  // 2. extract-as-transfer-gather
  // 3. post-extract-generic

  // Find the first tensor.extract operation and use it as a cut-off point for
  // gather vectorization.
  SmallVector<Operation *> preExtract;
  tensor::ExtractOp extractOp;
  SmallVector<Operation *> postExtract;

  for (Operation &op : linalgOp.getBody()->getOperations()) {
    if (extractOp) {
      // Already found extract, add to postExtract.
      postExtract.push_back(&op);
    } else {
      if (auto candidate = dyn_cast<tensor::ExtractOp>(op)) {
        extractOp = candidate;
        continue;
      }
      preExtract.push_back(&op);
    }
  }

  // If no extract op was found, call generic vectorization.
  if (!extractOp) {
    return linalg::vectorize(rewriter, linalgOp, vectorSizes, scalableVecDims,
                             vectorizeNDExtract);
  }

  Location loc = linalgOp->getLoc();
  SmallVector<int64_t> canonicalVectorSizes(vectorSizes);
  SmallVector<bool> canonicalScalableDims(scalableVecDims);

  // If vector sizes are not provided, assume static vector sizes and use loop
  // ranges.
  if (vectorSizes.empty()) {
    assert(canonicalScalableDims.empty() &&
           "vector sizes not provided but scalable vector sizes provided");
    canonicalVectorSizes = linalgOp.getStaticLoopRanges();
    canonicalScalableDims.append(linalgOp.getNumLoops(), false);

    // loop ranges must be static to infer vector sizes.
    if (ShapedType::isDynamicShape(canonicalVectorSizes)) {
      return failure();
    }
  }

  AffineMap itSpaceToExtract =
      getIterationSpaceToMemorySpaceMap(linalgOp, extractOp);
  if (!itSpaceToExtract) {
    return failure();
  }

  // Create a mapping from values used inside the linalg body to newly created
  // tensors.
  DenseMap<Value, std::pair<Value, AffineMap>> tmap;
  for (OpOperand &operand : linalgOp->getOpOperands()) {
    AffineMap map = linalgOp.getMatchingIndexingMap(&operand);
    Value blockArg = linalgOp.getMatchingBlockArgument(&operand);
    tmap[blockArg] = {operand.get(), map};
  }

  rewriter.setInsertionPointAfter(linalgOp);

  // Build the preExtract linalg.generic and vectorize it.
  linalg::GenericOp preOp = buildPartialGenericOp(
      rewriter, linalgOp, canonicalVectorSizes, preExtract, tmap);

  // Build the iree_vector_ext.transfer_gather operation.
  SmallVector<Value> baseIndices;
  SmallVector<Value> indexVecs;
  SmallVector<bool> indexed;
  SmallVector<AffineMap> indexedMaps;

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

  for (auto [i, index] : llvm::enumerate(extractOp.getIndices())) {
    if (!tmap.contains(index)) {
      baseIndices.push_back(index);
      indexed.push_back(false);
      continue;
    }

    auto [tensor, map] = tmap[index];

    Type elemType = getElementTypeOrSelf(index);
    AffineMap readMap = inverseAndBroadcastProjectedPermutation(map);
    VectorType readType = VectorType::get(canonicalVectorSizes, elemType);

    SmallVector<Value> operandIndices(map.getNumResults(), zero);

    // TODO: Mask the operation here. It's really hard to do that here though
    // because we don't have access to the vectorization infra, but maybe there
    // are easier ways to do it here.
    auto read = rewriter.create<vector::TransferReadOp>(
        loc, readType, tensor, operandIndices, readMap);

    baseIndices.push_back(zero);
    indexed.push_back(true);
    indexedMaps.push_back(inversePerm(itSpaceToExtract));
    indexVecs.push_back(read.getResult());
  }

  auto gatherTy = VectorType::get(canonicalVectorSizes, extractOp.getType());
  Value padding = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(extractOp.getType()));
  // tensor.extract always produces in-bounds accesses.
  SmallVector<Attribute> inBounds(gatherTy.getRank(),
                                  rewriter.getBoolAttr(true));

  auto transferGatherOp = rewriter.create<IREE::VectorExt::TransferGatherOp>(
      loc, gatherTy, extractOp.getTensor(), baseIndices, indexVecs,
      rewriter.getBoolArrayAttr(indexed),
      rewriter.getAffineMapArrayAttr(indexedMaps),
      inversePerm(itSpaceToExtract), padding,
      /*mask=*/Value(), rewriter.getArrayAttr(inBounds));

  // Create a empty tensor to write to.
  auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, canonicalVectorSizes,
                                                  gatherTy.getElementType());
  SmallVector<Value> writeIndices(canonicalVectorSizes.size(), zero);

  auto writeOp = rewriter.create<vector::TransferWriteOp>(
      loc, transferGatherOp.getResult(), emptyOp, writeIndices);

  tmap[extractOp.getResult()] = {
      writeOp.getResult(),
      rewriter.getMultiDimIdentityMap(canonicalVectorSizes.size())};

  // Build the postExtract linalg.generic.
  linalg::GenericOp postOp = buildPartialGenericOp(
      rewriter, linalgOp, canonicalVectorSizes, postExtract, tmap);

  rewriter.replaceOp(linalgOp, postOp);

  if (failed(vectorizeGatherLikeGenericToTransferGather(
          rewriter, preOp, vectorSizes, scalableVecDims, vectorizeNDExtract))) {
    return failure();
  };

  if (failed(vectorizeGatherLikeGenericToTransferGather(
          rewriter, postOp, vectorSizes, scalableVecDims,
          vectorizeNDExtract))) {
    return failure();
  }

  return success();
}

LogicalResult
vectorizeLinalgExtGatherToTransferGather(RewriterBase &rewriter,
                                         IREE::LinalgExt::GatherOp gatherOp) {

  // TODO: need to split the innermost dim of `indices` into `indexDepth`
  // vectors so that each independent index can be passed to the
  // iree_vector_ext.transfer_gather op.
  if (gatherOp.getIndexDepth() != 1) {
    return failure();
  }

  // TODO: There is no 1-to-1 conversion between `iree_linalg_ext.gather` and
  // `iree_vector_ext.transfer_gather` if the batch rank is > 1. Maybe support
  // unrolling the batch dimension in the future.
  if (gatherOp.getBatchRank() > 1) {
    return failure();
  }

  auto loc = gatherOp.getLoc();
  RewriterBase::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(gatherOp);

  ShapedType indicesTy = gatherOp.getIndicesType();
  ShapedType gatherTy = gatherOp.getOutputType();
  ShapedType sourceTy = gatherOp.getSourceType();

  auto gatherVectorTy =
      VectorType::get(gatherTy.getShape(), gatherTy.getElementType());
  // Rank-reduced to remove the innermost unit dim.
  auto indicesVecTy =
      VectorType::get(indicesTy.getShape().take_front(gatherOp.getBatchRank()),
                      rewriter.getIndexType());

  // Read `indices` tensor via `vector.transfer_read` and cast from int to
  // index.
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value indicesVec = rewriter.create<vector::TransferReadOp>(
      loc, indicesVecTy.clone(indicesTy.getElementType()),
      gatherOp.getIndices(), SmallVector<Value>(indicesTy.getRank(), zero));
  indicesVec =
      rewriter.create<arith::IndexCastOp>(loc, indicesVecTy, indicesVec);

  // Create transfer_gather op
  SmallVector<Value> baseIndices(sourceTy.getRank(), zero);
  SmallVector<bool> indexed(sourceTy.getRank(), false);
  indexed[0] = true;
  auto inBounds =
      rewriter.getBoolArrayAttr(SmallVector<bool>(sourceTy.getRank(), true));
  auto indexedMaps = rewriter.getAffineMapArrayAttr(SmallVector<AffineMap>(
      1,
      rewriter.getMultiDimIdentityMap(sourceTy.getRank()).getMajorSubMap(1)));
  Value padding = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(gatherTy.getElementType()));

  auto transferGatherOp = rewriter.create<IREE::VectorExt::TransferGatherOp>(
      loc, gatherVectorTy, gatherOp.getSource(), baseIndices,
      ValueRange{indicesVec}, rewriter.getBoolArrayAttr(indexed), indexedMaps,
      rewriter.getMultiDimIdentityMap(gatherTy.getRank()), padding,
      /*mask=*/Value(), inBounds);

  // Write back into tensor.
  auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, gatherTy.getShape(),
                                                  gatherTy.getElementType());
  SmallVector<Value> writeIndices(gatherTy.getRank(), zero);
  auto writeOp = rewriter.create<vector::TransferWriteOp>(
      loc, transferGatherOp.getResult(), emptyOp, writeIndices);
  rewriter.replaceOp(gatherOp, writeOp);
  return success();
}

} // namespace mlir::iree_compiler::IREE::VectorExt
