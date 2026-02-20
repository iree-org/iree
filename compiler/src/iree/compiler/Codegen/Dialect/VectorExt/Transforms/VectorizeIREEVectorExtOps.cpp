// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::VectorExt {

#define GEN_PASS_DEF_VECTORIZEIREEVECTOREXTOPSPASS
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h.inc"

namespace {

struct VectorizeToLayoutOpPattern final
    : OpRewritePattern<IREE::VectorExt::ToLayoutOp> {
  using Base::Base;

  vector::TransferReadOp
  createReadOp(PatternRewriter &rewriter,
               IREE::VectorExt::ToLayoutOp toLayoutOp) const {
    Location loc = toLayoutOp.getLoc();
    ShapedType inputTy = toLayoutOp.getType();
    auto zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto identityMap = rewriter.getMultiDimIdentityMap(inputTy.getRank());
    SmallVector<int64_t> readShape =
        toLayoutOp.getLayout().getUndistributedShape();
    Value mask = nullptr;
    if (!toLayoutOp.getType().hasStaticShape()) {
      SmallVector<OpFoldResult> mixedSourceDims =
          tensor::getMixedSizes(rewriter, loc, toLayoutOp.getInput());
      auto maskType = VectorType::get(readShape, rewriter.getI1Type());
      mask = vector::CreateMaskOp::create(rewriter, loc, maskType,
                                          mixedSourceDims);
    }
    VectorType vectorType =
        VectorType::get(readShape, inputTy.getElementType());
    auto inBounds = rewriter.getBoolArrayAttr(
        SmallVector<bool>(vectorType.getRank(), true));
    auto padValue =
        ub::PoisonOp::create(rewriter, loc, inputTy.getElementType());
    auto read = vector::TransferReadOp::create(
        rewriter, loc,
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
    auto zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    int64_t rank = tensorTy.getShape().size();
    auto inBounds = rewriter.getBoolArrayAttr(SmallVector<bool>(rank, true));
    auto identityMap = rewriter.getMultiDimIdentityMap(tensorTy.getRank());
    return vector::TransferWriteOp::create(
        rewriter, loc,
        /*result=*/resType,
        /*vector=*/vectorLayoutOp,
        /*source=*/tensorLayoutOp.getInput(),
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
    auto newLayoutOp = IREE::VectorExt::ToLayoutOp::create(
        rewriter, loc, readOp, toLayoutOp.getLayout(),
        toLayoutOp.getMmaKindAttr(), toLayoutOp.getSharedMemoryConversion());
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
    Value out = tensor::EmptyOp::create(rewriter, fullOp.getLoc(), vectorSizes,
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

  auto newOp = linalg::GenericOp::create(
      rewriter, fullOp.getLoc(), TypeRange(outs), ins, outs, indexingMaps,
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
          linalg::YieldOp::create(b, loc, yieldValues);
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
    FailureOr<linalg::VectorizationResult> result = linalg::vectorize(
        rewriter, linalgOp, vectorSizes, scalableVecDims, vectorizeNDExtract);
    if (failed(result)) {
      return failure();
    }
    rewriter.replaceOp(linalgOp, result->replacements);
    return success();
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
  SmallVector<Value> baseOffsets;
  SmallVector<Value> indexVecs;
  SmallVector<AffineMap> indexVecMaps;

  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);

  // Build the source indexing map. Every index is treated as gathered (symbol).
  // Canonicalization (foldTransferGatherFromStep, FoldSingleElementIndexVec,
  // etc.) will recover contiguous dims where possible.
  MLIRContext *ctx = rewriter.getContext();
  SmallVector<AffineExpr> sourceMapExprs;
  int64_t numSymbols = 0;
  for (auto [i, index] : llvm::enumerate(extractOp.getIndices())) {
    if (!tmap.contains(index)) {
      // Value defined outside the block — loop-invariant (constant/broadcast).
      baseOffsets.push_back(index);
      sourceMapExprs.push_back(getAffineConstantExpr(0, ctx));
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
    auto read = vector::TransferReadOp::create(
        rewriter, loc, readType, tensor, operandIndices,
        /*padding=*/std::nullopt, readMap);

    baseOffsets.push_back(zero);
    // This source dim is gathered: use a symbol.
    int64_t symIdx = numSymbols++;
    sourceMapExprs.push_back(getAffineSymbolExpr(symIdx, ctx));
    // The index vec map: the read result has shape canonicalVectorSizes, so
    // it's indexed by all vector dims (identity map).
    indexVecMaps.push_back(
        rewriter.getMultiDimIdentityMap(canonicalVectorSizes.size()));
    indexVecs.push_back(read.getResult());
  }

  AffineMap sourceMap = AffineMap::get(
      /*dimCount=*/canonicalVectorSizes.size(), numSymbols, sourceMapExprs,
      ctx);

  // Build the full indexing_maps array: [sourceMap, indexVecMap0, ...]
  SmallVector<AffineMap> indexingMaps;
  indexingMaps.push_back(sourceMap);
  for (AffineMap &m : indexVecMaps) {
    // Add symbols to each index vec map to match the source map.
    m = AffineMap::get(m.getNumDims(), numSymbols, m.getResults(), ctx);
    indexingMaps.push_back(m);
  }

  auto gatherTy = VectorType::get(canonicalVectorSizes, extractOp.getType());
  Value padding = ub::PoisonOp::create(rewriter, loc, extractOp.getType());

  auto transferGatherOp = IREE::VectorExt::TransferGatherOp::create(
      rewriter, loc, gatherTy, extractOp.getTensor(), baseOffsets, indexVecs,
      rewriter.getAffineMapArrayAttr(indexingMaps), padding,
      /*mask=*/Value());

  // Create a empty tensor to write to.
  auto emptyOp = tensor::EmptyOp::create(rewriter, loc, canonicalVectorSizes,
                                         gatherTy.getElementType());
  SmallVector<Value> writeIndices(canonicalVectorSizes.size(), zero);

  auto writeOp = vector::TransferWriteOp::create(
      rewriter, loc, transferGatherOp.getResult(), emptyOp, writeIndices);

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

Value maskOperation(RewriterBase &rewriter, Operation *op, Value mask) {
  Value maskedOp =
      cast<vector::MaskOp>(mlir::vector::maskOperation(rewriter, op, mask))
          .getResult(0);
  return maskedOp;
}

LogicalResult
vectorizeLinalgExtGatherToTransferGather(RewriterBase &rewriter,
                                         IREE::LinalgExt::GatherOp gatherOp,
                                         ArrayRef<int64_t> vectorSizes) {

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

  if (vectorSizes.empty()) {
    vectorSizes = gatherTy.getShape();
  }

  auto gatherVectorTy = VectorType::get(vectorSizes, gatherTy.getElementType());
  // Rank-reduced to remove the innermost unit dim.
  auto indicesVecTy = VectorType::get(
      vectorSizes.take_front(gatherOp.getBatchRank()), rewriter.getIndexType());

  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto indicesVecRead = vector::TransferReadOp::create(
      rewriter, loc, indicesVecTy.clone(indicesTy.getElementType()),
      gatherOp.getIndices(), SmallVector<Value>(indicesTy.getRank(), zero),
      std::nullopt);
  VectorType indicesMaskType = indicesVecTy.clone(rewriter.getI1Type());
  SmallVector<OpFoldResult> gatherDims =
      tensor::getMixedSizes(rewriter, loc, gatherOp.getOutput());
  Value indicesMask = vector::CreateMaskOp::create(
      rewriter, loc, indicesMaskType,
      ArrayRef(gatherDims).take_front(gatherOp.getBatchRank()));
  Value indicesVec = maskOperation(rewriter, indicesVecRead, indicesMask);
  indicesVec =
      arith::IndexCastOp::create(rewriter, loc, indicesVecTy, indicesVec);

  SmallVector<Value> baseOffsets(sourceTy.getRank(), zero);
  Value padding =
      ub::PoisonOp::create(rewriter, loc, gatherTy.getElementType());

  // Build indexing_maps for the transfer_gather.
  // Source map: (vector_dims)[s0] -> (s0, d_batch+1, ..., d_N)
  // First source dim is gathered (s0), rest are contiguous.
  MLIRContext *ctx = rewriter.getContext();
  int64_t vectorRank = vectorSizes.size();
  int64_t sourceRank = sourceTy.getRank();
  SmallVector<AffineExpr> sourceMapExprs;
  sourceMapExprs.push_back(getAffineSymbolExpr(0, ctx)); // gathered dim 0
  for (int64_t i = 1; i < sourceRank; ++i) {
    // Map remaining source dims to corresponding vector dims.
    // The batch dims come first, so source dim i maps to vector dim
    // (i - 1 + batchRank). For batchRank <= 1 and indexDepth == 1,
    // source dim i maps to vector dim i.
    sourceMapExprs.push_back(getAffineDimExpr(i, ctx));
  }
  AffineMap sourceMap =
      AffineMap::get(vectorRank, /*symbolCount=*/1, sourceMapExprs, ctx);

  // Index vec map: (vector_dims)[s0] -> (d0) for batch_rank == 1
  AffineMap indexVecMap = AffineMap::get(vectorRank, /*symbolCount=*/1,
                                         {getAffineDimExpr(0, ctx)}, ctx);

  SmallVector<AffineMap> indexingMaps = {sourceMap, indexVecMap};

  auto transferGatherOp = IREE::VectorExt::TransferGatherOp::create(
      rewriter, loc, gatherVectorTy, gatherOp.getSource(), baseOffsets,
      ValueRange{indicesVec}, rewriter.getAffineMapArrayAttr(indexingMaps),
      padding, /*mask=*/Value());

  VectorType gatherMaskType = gatherVectorTy.clone(rewriter.getI1Type());
  Value gatherMask =
      vector::CreateMaskOp::create(rewriter, loc, gatherMaskType, gatherDims);
  Value maskedGather = maskOperation(rewriter, transferGatherOp, gatherMask);
  SmallVector<Value> writeIndices(gatherTy.getRank(), zero);
  auto writeOp = vector::TransferWriteOp::create(
      rewriter, loc, maskedGather, gatherOp.getOutput(), writeIndices);
  Value maskedWrite = maskOperation(rewriter, writeOp, gatherMask);

  rewriter.replaceOp(gatherOp, maskedWrite);
  return success();
}

LogicalResult
vectorizeLinalgExtArgCompare(RewriterBase &rewriter,
                             IREE::LinalgExt::ArgCompareOp argCompareOp,
                             ArrayRef<int64_t> vectorSizes) {
  Location loc = argCompareOp.getLoc();
  RewriterBase::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(argCompareOp);

  auto inputValTy = cast<ShapedType>(argCompareOp.getInputValue().getType());
  // Only static shapes are supported. Dynamic shapes would require masking.
  // Check input shape (includes reduction dimension) to catch dynamic reduction
  // dimensions that wouldn't appear in the static output shape.
  if (!inputValTy.hasStaticShape()) {
    return failure();
  }

  ShapedType outValTy = argCompareOp.getOutputValueType();
  ShapedType outIdxTy = argCompareOp.getOutputIndexType();

  if (vectorSizes.empty()) {
    vectorSizes = outValTy.getShape();
  }

  // Ensure full tiles - partial tiles would require masking support.
  if (vectorSizes != outValTy.getShape()) {
    return failure();
  }

  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);

  // Vectorize the DPS input operands (input_value and optional input_index).
  // We explicitly access these by name to avoid accidentally including
  // index_base.
  SmallVector<Value> inputVecs;
  auto vectorizeInput = [&](Value input) {
    auto inputTy = cast<ShapedType>(input.getType());
    SmallVector<Value> readIndices(inputTy.getRank(), zero);
    auto inputVecTy =
        VectorType::get(inputTy.getShape(), inputTy.getElementType());
    // TODO(Bangtian): Add masking/padding support for partial tiles.
    // Currently passes std::nullopt, assuming vector size matches tensor shape.
    auto readOp = vector::TransferReadOp::create(
        rewriter, loc, inputVecTy, input, readIndices, std::nullopt);
    inputVecs.push_back(readOp);
  };

  vectorizeInput(argCompareOp.getInputValue());
  if (Value inputIndex = argCompareOp.getInputIndex()) {
    vectorizeInput(inputIndex);
  }

  SmallVector<Value> initVecs;
  for (Value init : argCompareOp.getDpsInits()) {
    auto initTy = cast<ShapedType>(init.getType());
    SmallVector<Value> readIndices(vectorSizes.size(), zero);
    auto initVecTy = VectorType::get(vectorSizes, initTy.getElementType());
    auto readOp = vector::TransferReadOp::create(rewriter, loc, initVecTy, init,
                                                 readIndices, std::nullopt);
    initVecs.push_back(readOp);
  }

  auto outValVecTy = VectorType::get(vectorSizes, outValTy.getElementType());
  auto outIdxVecTy = VectorType::get(vectorSizes, outIdxTy.getElementType());

  Region &srcRegion = argCompareOp.getRegion();

  // Create the vector arg_compare operation using the builder that takes
  // individual operands (this properly handles AttrSizedOperandSegments).
  Value inputIndex = (inputVecs.size() > 1) ? inputVecs[1] : Value();
  Value indexBase = argCompareOp.getIndexBase();

  auto vectorArgCompareOp = IREE::VectorExt::ArgCompareOp::create(
      rewriter, loc, TypeRange{outValVecTy, outIdxVecTy},
      /*input_value=*/inputVecs[0],
      /*input_index=*/inputIndex,
      /*init_value=*/initVecs[0],
      /*init_index=*/initVecs[1],
      /*index_base=*/indexBase,
      /*dimension=*/argCompareOp.getDimension());

  Block *srcBlock = &srcRegion.front();
  Region &dstRegion = vectorArgCompareOp.getRegion();
  rewriter.modifyOpInPlace(vectorArgCompareOp,
                           [&]() { dstRegion.getBlocks().clear(); });
  SmallVector<Location> argLocs(srcBlock->getNumArguments(), loc);
  Block *dstBlock = rewriter.createBlock(&dstRegion, dstRegion.end(),
                                         srcBlock->getArgumentTypes(), argLocs);

  // Clone operations from source block to destination block using rewriter.
  IRMapping mapper;
  for (auto [srcArg, dstArg] :
       llvm::zip_equal(srcBlock->getArguments(), dstBlock->getArguments())) {
    mapper.map(srcArg, dstArg);
  }

  rewriter.setInsertionPointToStart(dstBlock);
  for (Operation &op : srcBlock->getOperations()) {
    auto yieldOp = dyn_cast<IREE::LinalgExt::YieldOp>(op);
    if (!yieldOp) {
      rewriter.clone(op, mapper);
      continue;
    }
    // Replace LinalgExt::YieldOp with VectorExt::YieldOp.
    SmallVector<Value> mappedOperands;
    for (Value operand : yieldOp.getOperands()) {
      mappedOperands.push_back(mapper.lookup(operand));
    }
    IREE::VectorExt::YieldOp::create(rewriter, yieldOp.getLoc(),
                                     mappedOperands);
  }

  // Set insertion point to after the vectorArgCompareOp for subsequent
  // operations.
  rewriter.setInsertionPointAfter(vectorArgCompareOp);

  SmallVector<Value> results;
  for (auto [result, output] : llvm::zip_equal(vectorArgCompareOp.getResults(),
                                               argCompareOp.getDpsInits())) {
    SmallVector<Value> writeIndices(vectorSizes.size(), zero);
    auto writeOp = vector::TransferWriteOp::create(rewriter, loc, result,
                                                   output, writeIndices);
    results.push_back(writeOp.getResult());
  }

  rewriter.replaceOp(argCompareOp, results);
  return success();
}

/// Lowers vector.mask %mask { iree_vector_ext.transfer_gather }
///  into
/// iree_vector_ext.transfer_gather %mask
///
/// Ideally, the mask should have just been put on transfer_gather directly,
/// but this is done this way to match upstream vector.transfer_read masking.
struct MaskedTransferGatherOpPattern : public OpRewritePattern<vector::MaskOp> {
public:
  using Base::Base;

  LogicalResult matchAndRewrite(vector::MaskOp maskOp,
                                PatternRewriter &rewriter) const override {
    auto gatherOp = dyn_cast<TransferGatherOp>(maskOp.getMaskableOp());
    if (!gatherOp) {
      return failure();
    }
    // TODO: The 'vector.mask' passthru is a vector and 'transfer_gather'
    // expects a scalar. We could only lower one to the other for cases where
    // the passthru is a broadcast of a scalar.
    if (maskOp.hasPassthru()) {
      return rewriter.notifyMatchFailure(
          maskOp, "can't lower passthru to transfer_gather");
    }
    // Add a mask indexing map (identity) to the existing indexing_maps.
    SmallVector<AffineMap> indexingMaps = gatherOp.getIndexingMapsArray();
    int64_t vectorRank = gatherOp.getVector().getType().getRank();
    int64_t numSymbols = gatherOp.getIndexVecs().size();
    AffineMap maskMap =
        AffineMap::getMultiDimIdentityMap(vectorRank, rewriter.getContext());
    maskMap = AffineMap::get(vectorRank, numSymbols, maskMap.getResults(),
                             rewriter.getContext());
    indexingMaps.push_back(maskMap);

    rewriter.replaceOpWithNewOp<TransferGatherOp>(
        maskOp, gatherOp.getVector().getType(), gatherOp.getBase(),
        gatherOp.getOffsets(), gatherOp.getIndexVecs(),
        rewriter.getAffineMapArrayAttr(indexingMaps), gatherOp.getPadding(),
        maskOp.getMask());
    return success();
  }
};

void populateVectorMaskLoweringPatterns(RewritePatternSet &patterns) {
  patterns.add<MaskedTransferGatherOpPattern>(patterns.getContext());
}

} // namespace mlir::iree_compiler::IREE::VectorExt
