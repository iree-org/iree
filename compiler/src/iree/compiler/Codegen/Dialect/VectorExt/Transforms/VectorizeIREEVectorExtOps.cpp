// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-generic-vectorization"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

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

} // namespace

namespace {
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
} // namespace

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

LogicalResult vectorizeGatherToTransferGather(RewriterBase &rewriter,
                                              Operation *op,
                                              ArrayRef<int64_t> vectorSizes,
                                              ArrayRef<bool> scalableVecDims) {
  assert(LinalgExt::isGatherlikeOp(op) &&
         "Expected operation to be gather like");

  Location loc = op->getLoc();
  auto linalgOp = cast<linalg::GenericOp>(op);

  SmallVector<int64_t> canonicalVectorSizes(vectorSizes);
  SmallVector<bool> canonicalScalableDims(scalableVecDims);

  if (vectorSizes.empty()) {
    canonicalVectorSizes = linalgOp.getStaticLoopRanges();
    canonicalScalableDims.append(linalgOp.getNumLoops(), false);
  }

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

  // Mapping from values used inside the linalg body to newly created tensors.
  DenseMap<Value, std::pair<Value, AffineMap>> tmap;
  for (OpOperand &operand : linalgOp->getOpOperands()) {
    AffineMap map = linalgOp.getMatchingIndexingMap(&operand);
    Value blockArg = linalgOp.getMatchingBlockArgument(&operand);
    tmap[blockArg] = {operand.get(), map};
  }

  rewriter.setInsertionPointAfter(linalgOp);

  // Build the preExtract linalg.generic.
  linalg::GenericOp preOp = buildPartialGenericOp(
      rewriter, linalgOp, canonicalVectorSizes, preExtract, tmap);

  // Build the iree_vector_ext.transfer_gather.
  SmallVector<Value> baseIndices;
  SmallVector<Value> indexVecs;
  SmallVector<int64_t> indexed;
  SmallVector<AffineMap> indexedMaps;

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

  for (auto [i, index] : llvm::enumerate(extractOp.getIndices())) {
    if (!tmap.contains(index)) {
      baseIndices.push_back(index);
      continue;
    }

    auto [tensor, map] = tmap[index];

    Type elemType = getElementTypeOrSelf(index);
    AffineMap readMap = inverseAndBroadcastProjectedPermutation(map);
    VectorType readType = VectorType::get(canonicalVectorSizes, elemType);

    SmallVector<Value> operandIndices(map.getNumResults(), zero);

    // TODO: Mask the operation here. It's really hard to do that here though
    // because we don't have access to the vectorization infra.
    auto read = rewriter.create<vector::TransferReadOp>(
        loc, readType, tensor, operandIndices, readMap);

    baseIndices.push_back(zero);
    indexed.push_back(i);
    indexedMaps.push_back(readMap);

    indexVecs.push_back(read.getResult());
  }

  auto gatherTy = VectorType::get(canonicalVectorSizes, extractOp.getType());
  Value padding = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(extractOp.getType()));
  SmallVector<Attribute> inBounds(gatherTy.getRank(),
                                  rewriter.getBoolAttr(true));

  auto transferGatherOp = rewriter.create<IREE::VectorExt::TransferGatherOp>(
      loc, gatherTy, extractOp.getTensor(), baseIndices, indexVecs, indexed,
      rewriter.getAffineMapArrayAttr(indexedMaps),
      rewriter.getMultiDimIdentityMap(gatherTy.getRank()), padding,
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

  postOp->getParentOp()->dump();

  (void)linalg::vectorize(rewriter, preOp, canonicalVectorSizes,
                          canonicalScalableDims, true);
  (void)linalg::vectorize(rewriter, postOp, canonicalVectorSizes,
                          canonicalScalableDims, true);

  return success();
}

} // namespace mlir::iree_compiler::IREE::VectorExt
