// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUVECTORLOWERINGPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

//====---------------------------------------------------------------------===//
// Patterns for late vector op lowering.
//====---------------------------------------------------------------------===//

namespace {

struct PromoteContractOperands final
    : public vector::MaskableOpRewritePattern<vector::ContractionOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  FailureOr<Value>
  matchAndRewriteMaskableOp(vector::ContractionOp contractOp,
                            vector::MaskingOpInterface maskOp,
                            PatternRewriter &rewriter) const override {
    Type operandElType = getElementTypeOrSelf(contractOp.getLhsType());
    Type resultElType = getElementTypeOrSelf(contractOp.getResultType());

    if (operandElType == resultElType) {
      return failure();
    }

    Location loc = contractOp.getLoc();
    Value lhs =
        promoteToElementType(loc, rewriter, contractOp.getLhs(), resultElType);
    Value rhs =
        promoteToElementType(loc, rewriter, contractOp.getRhs(), resultElType);

    auto replacement = rewriter.create<vector::ContractionOp>(
        loc, lhs, rhs, contractOp.getAcc(), contractOp.getIndexingMaps(),
        contractOp.getIteratorTypes());

    if (!maskOp) {
      return replacement.getResult();
    }
    auto maskedOp = vector::maskOperation(
        rewriter, replacement, maskOp.getMask(), maskOp.getPassthru());
    return maskedOp->getResult(0);
  }

  Value promoteToElementType(Location loc, RewriterBase &rewriter, Value v,
                             Type dstElementType) const {
    Type elementType = getElementTypeOrSelf(v.getType());
    if (elementType == dstElementType)
      return v;

    // vector.contract only allows extension on operands.
    assert(elementType.getIntOrFloatBitWidth() <=
               dstElementType.getIntOrFloatBitWidth() &&
           "vector.contract does not allow truncation of operands");

    Type promotedType = dstElementType;
    if (auto vecType = dyn_cast<VectorType>(v.getType()))
      promotedType = vecType.clone(promotedType);

    if (isa<FloatType>(dstElementType))
      return rewriter.create<arith::ExtFOp>(loc, promotedType, v);
    // For integer types, vector.contract only supports signless integer types
    // and promotion happens via sign extension.
    return rewriter.create<arith::ExtSIOp>(loc, promotedType, v);
  }
};


void processBlock(Block *block, IRRewriter & rewriter){

  llvm::errs() << "Processing block" << "\n";


  // A map from vectors in the values that are inserted into them.
  DenseMap<Value, SmallVector<std::pair<Value, int>>> values;

  auto processSlice = [&](vector::InsertStridedSliceOp insertSlice) {
    llvm::errs() << "Processing slice " << insertSlice << "\n";
    VectorType smallType = insertSlice.getSourceVectorType();
    VectorType largeType = insertSlice.getDestVectorType();
    if (smallType.getRank() != 1 || largeType.getRank() != 1)
      return;

    Value dst = insertSlice.getDest();
    Value src = insertSlice.getValueToStore();
    Value res = insertSlice.getResult();

    auto offsets = insertSlice.getOffsets();
    if (offsets.size() != 1)
      return;
    mlir::Attribute offsetAttr = *offsets.begin();
    auto intAttr = dyn_cast<IntegerAttr>(offsetAttr);
    if (!intAttr) {
      return;
    }
    int offset = intAttr.getInt();

    int64_t nElmsSmall = smallType.getNumElements();
    int64_t nElmsLarge = largeType.getNumElements();

    auto it = values.find(dst);
    if (it != values.end()) {
      // TODO(newling) efficiency here
      values.insert({res, it->second});
      // values.erase(dst);
      // values.erase(it);
    } else {
      values.insert(
          {res, SmallVector<std::pair<Value, int>>(nElmsLarge, {nullptr, -1})});
    }

    auto & vec = values[res];
    for (int i = 0; i < nElmsSmall; ++i) {
      vec[i + offset] = {src, i};
    }

    llvm::errs() << "processed\n";
  };

  auto processShuffle = [&](vector::ShuffleOp shuffleOp) {
    llvm::errs() << "Processing shuffle " << shuffleOp << "\n";
    VectorType outType = shuffleOp.getType();
    VectorType lhsType = shuffleOp.getV1VectorType();
    if (outType.getRank() != 1 || lhsType.getRank() != 1)
      return;

    int64_t nLhsElms = lhsType.getNumElements();
    ArrayRef<int64_t> indices = shuffleOp.getMask();
    if (std::any_of(indices.begin(), indices.end(),
                    [&](int64_t i) { return i >= nLhsElms; }))
      return;

    auto it = values.find(shuffleOp.getV1());
    if (it == values.end())
      return;

    auto &vec = it->second;
    if (std::any_of(indices.begin(), indices.end(),
                    [&](int64_t i) { return !vec[i].first; }))
      return;

    SmallVector<Value> fromElms;
    fromElms.reserve(indices.size());

    llvm::errs() << "fromElms, reserved " << indices.size() << "\n";

    for (auto index : indices) {
      auto backtracked = vec[index];
      Value backtracedSrc = backtracked.first;
      assert(backtracedSrc && "already checked no?");
      auto localIndex = backtracked.second;
      rewriter.setInsertionPoint(shuffleOp);
      // Create a vector.extract from the rank-1 source to a scalar.
      auto scalar = rewriter.create<vector::ExtractOp>(
          shuffleOp.getLoc(), backtracedSrc, SmallVector<int64_t>{localIndex});

      llvm::errs() << "extract created : " << scalar << "\n";
      fromElms.push_back(scalar);
    }

    auto replacement = rewriter.create<vector::FromElementsOp>(shuffleOp.getLoc(), outType,
                                            fromElms);

    llvm::errs() << "replacement is " << replacement << "\n";
    rewriter.replaceOp(shuffleOp, replacement.getResult());

    llvm::errs() << "shuffle processed" << "\n";
  };


  SmallVector<Operation *> toProcess;

  for (const Operation &op : block->getOperations()) {
    if (isa<vector::InsertStridedSliceOp, vector::ShuffleOp>(op)){
      toProcess.push_back(const_cast<Operation *>(&op));
    }
  }
  for (auto op : toProcess) {
    if (auto slice = dyn_cast<vector::InsertStridedSliceOp>(op)) {
      processSlice(slice);
    }
    if (auto shuffleOp = dyn_cast<vector::ShuffleOp>(op)) {
      processShuffle(shuffleOp);
    }
  }
}


// clang-format off
//       %511 = vector.insert_strided_slice %480, %cst {offsets = [0], strides = [1]} : vector<4xf32> into vector<64xf32>
//       %512 = vector.insert_strided_slice %482, %511 {offsets = [4], strides = [1]} : vector<4xf32> into vector<64xf32>
//       %513 = vector.insert_strided_slice %484, %512 {offsets = [8], strides = [1]} : vector<4xf32> into vector<64xf32>
//       %514 = vector.insert_strided_slice %486, %513 {offsets = [12], strides = [1]} : vector<4xf32> into vector<64xf32>
//       %515 = vector.insert_strided_slice %488, %514 {offsets = [16], strides = [1]} : vector<4xf32> into vector<64xf32>
//       %516 = vector.insert_strided_slice %490, %515 {offsets = [20], strides = [1]} : vector<4xf32> into vector<64xf32>
//       %517 = vector.insert_strided_slice %492, %516 {offsets = [24], strides = [1]} : vector<4xf32> into vector<64xf32>
//       %518 = vector.insert_strided_slice %494, %517 {offsets = [28], strides = [1]} : vector<4xf32> into vector<64xf32>
//       %519 = vector.insert_strided_slice %496, %518 {offsets = [32], strides = [1]} : vector<4xf32> into vector<64xf32>
//       %520 = vector.insert_strided_slice %498, %519 {offsets = [36], strides = [1]} : vector<4xf32> into vector<64xf32>
//       %521 = vector.insert_strided_slice %500, %520 {offsets = [40], strides = [1]} : vector<4xf32> into vector<64xf32>
//       %522 = vector.insert_strided_slice %502, %521 {offsets = [44], strides = [1]} : vector<4xf32> into vector<64xf32>
//       %523 = vector.insert_strided_slice %504, %522 {offsets = [48], strides = [1]} : vector<4xf32> into vector<64xf32>
//       %524 = vector.insert_strided_slice %506, %523 {offsets = [52], strides = [1]} : vector<4xf32> into vector<64xf32>
//       %525 = vector.insert_strided_slice %508, %524 {offsets = [56], strides = [1]} : vector<4xf32> into vector<64xf32>
//       %526 = vector.insert_strided_slice %510, %525 {offsets = [60], strides = [1]} : vector<4xf32> into vector<64xf32>
// clang-format on
//
// replace 
//       %511 = vector.insert_strided_slice %480, %cst {offsets = [0], strides = [1]} : vector<4xf32> into vector<64xf32>
//       %512 = vector.insert_strided_slice %482, %511 {offsets = [4], strides = [1]} : vector<4xf32> into vector<64xf32>
//
// with 
//       %511 = vector.shuffle %480, %482 [0, ... 7] vector<4xf32>, vector<4xf32> 
//       %512 = vector.insert_strided_slice %511 ... 
//
// 

// LogicalResult insertTree(Operation * funcOp){
// 
// }


LogicalResult localExtractInsertOptimizations(Operation * funcOp){

  IRRewriter rewriter(funcOp->getContext());



  // Let's start surgically, go straight for what we know is the optimization.
  SmallVector<vector::ShuffleOp> shuffleOps;
  DenseSet<Block *> blocksWithShuffles;
  funcOp->walk([&](vector::ShuffleOp shuffleOp) {
    blocksWithShuffles.insert(shuffleOp->getBlock());
    // // Check that it is rank-1:
    // if (shuffleOp.getType().getRank() != 1) {
    //   return WalkResult::advance();
    // }
    // auto indices = shuffleOp.getMask();
    // shuffleOps.push_back(shuffleOp);
    // return WalkResult::advance();
  });

  for (auto block : blocksWithShuffles){
    processBlock(block, rewriter);
    llvm::errs() << "block processed " << "\n";
  }

  llvm::errs() << "all blocks processed " << "\n";

  return success();
}

std::optional<SmallVector<int64_t>> getNativeVectorShape(Operation *op) {

  // Elementwise operations.
  // Set the native shape by setting all but the final non-1 to 1.
  //
  // Example: <100x100x4x1xf32> -> <1x1x4x1xf32>
  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1) {
    if (auto vectorType = llvm::dyn_cast<VectorType>(op->getResultTypes()[0])) {
      int64_t rank = vectorType.getRank();
      ArrayRef<int64_t> shape = vectorType.getShape();
     //  auto bitWidth  = vectorType.getElementTypeBitWidth();
     //  int factor = 1;
     //  if (bitWidth == 32) factor = 1; 
     //  else if (bitWidth == 16) factor = 2;
     //  else if (bitWidth == 8) factor = 4;

      auto iter = std::find_if_not(shape.rbegin(), shape.rend(),
                                   [](int64_t dim) { return dim == 1; });
      SmallVector<int64_t> nativeSize(rank, 1);
      if (iter != shape.rend()) {
        // Found a non-1 dimension, so we can keep it.
        auto val = *iter;
        val = 1;
       //  if (val % factor == 0) val = factor;
       //  else val = 1;
        nativeSize[rank - 1 - std::distance(shape.rbegin(), iter)] = val;
      }
      return nativeSize;
    }
  }

  // Unroll vector.transpose in all but the inner-most dimension of result.
  // Example: A transpose `vector<2x4xf32> to vector<4x2xf32>` results in 4
  // extract->insert_strided_slice pairs.
  //
  // An alternative is to use `populateVectorTransposeLoweringPatterns`
  // which always creates scalar extract-insert pairs.
  //
  // TODO(newling) reconsider the optimal strategy for this.
  if (auto transposeOp = llvm::dyn_cast<vector::TransposeOp>(op)) {
    VectorType vectorType = transposeOp.getType();
    SmallVector<int64_t> nativeSize(vectorType.getRank(), 1);
    if (vectorType.getRank() > 0) {
      nativeSize.back() = vectorType.getShape().back();
    }
    return nativeSize;
  }
  return std::nullopt;
}

struct LLVMGPUVectorLoweringPass final
    : impl::LLVMGPUVectorLoweringPassBase<LLVMGPUVectorLoweringPass> {
  using impl::LLVMGPUVectorLoweringPassBase<
      LLVMGPUVectorLoweringPass>::LLVMGPUVectorLoweringPassBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<scf::SCFDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();

    auto addVectorCanonicalizationPatterns = [](RewritePatternSet &patterns) {
      MLIRContext *ctx = patterns.getContext();
      vector::BroadcastOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ExtractStridedSliceOp::getCanonicalizationPatterns(patterns, ctx);
      vector::InsertOp::getCanonicalizationPatterns(patterns, ctx);
      vector::InsertStridedSliceOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ShapeCastOp::getCanonicalizationPatterns(patterns, ctx);
      vector::TransposeOp::getCanonicalizationPatterns(patterns, ctx);
      populateConvertToShapeCastPatterns(patterns);
    };

    MLIRContext *context = funcOp.getContext();

    // Remove permutation_map, replace with explict broadcast and transpose ops
    // (which we immediately try to canonicalize away).
    {
      RewritePatternSet patterns(context);
      vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
      addVectorCanonicalizationPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // vector->vector conversions, and unrolling.
    {
      RewritePatternSet patterns(context);

      // Unroll broadcast, leaving rank-1 broadcast.
      vector::populateVectorBroadcastLoweringPatterns(patterns);

      // Unroll gather, leaving rank-1 gathers.
      vector::populateVectorGatherLoweringPatterns(patterns);

      // Unroll create_mask, leaving rank-1 create_masks.
      vector::populateVectorMaskOpLoweringPatterns(patterns);

      // Convert contract to fma.
      vector::populateVectorContractLoweringPatterns(
          patterns, vector::VectorContractLowering::OuterProduct);
      patterns.add<PromoteContractOperands>(context);

      // Convert multi_reduce to arith ops.
      vector::populateVectorMultiReductionLoweringPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerParallel);

      // Unroll remaining vops. Currently transpose and elementwise ops are
      // handled here.
      auto opts = vector::UnrollVectorOptions().setNativeShapeFn(
          [=](auto op) { return getNativeVectorShape(op); });
      vector::populateVectorUnrollPatterns(patterns, opts);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    llvm::errs() << "\n\n==================================\nState just before "
                    "the start of the lowering of the transfer ops"
                 << "\n==================================\n";
    llvm::errs() << funcOp << "\n";


    // transfer_read -> load and transfer_write -> store.
    {
      RewritePatternSet patterns(context);
      VectorTransferToSCFOptions vectorToSCFOptions;
      vectorToSCFOptions.enableFullUnroll();
      populateVectorToSCFConversionPatterns(patterns, vectorToSCFOptions);
      memref::populateFoldMemRefAliasOpPatterns(patterns);
      amdgpu::populateAmdgpuTransferReadToLoadPatterns(patterns);
      vector::populateVectorTransferLoweringPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
   //  {
   //    RewritePatternSet patterns(context);
   //    vector::populateVectorTransferLoweringPatterns(patterns);
   //    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
   //      return signalPassFailure();
   //    }
   //  }

    // Canonicalize.
    {
      RewritePatternSet patterns(context);
      vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
      addVectorCanonicalizationPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    llvm::errs() << "\n\n==================================\nState just before "
                    "the start of flattening process"
                 << "\n==================================\n";
    llvm::errs() << funcOp << "\n";


    // TODO(newling) it's the flattening which is causing the increase in memory. 
     // Flatten!
     {
       RewritePatternSet patterns(context);
       GreedyRewriteConfig config;
       config.fold = false;
 
       // TODO(newling) this is very clearly defined set of patterns --
 
       // energy function that the patterns try to minimize is
       // sum(operations in vector and arith dialects of) energy(op)
       // - energy(shape_cast) = 0
       // - energy(other_op) = sum of ranks of vector operands.
 
       // IREE uses this as a late stage canonicaliazation before lowering to
       // LLVM, only after unrolling of single-threaded code. Any pattern which
       // decreases this object should be added. Any pattern that increases this
       // objective should definitely not be added to avoid cycles. Any pattern
       // that leaves the energy unchanged -- the energy function can be extended
       // (lexicographically).
 
       populateFlattenVectorExtractInsertPatterns(patterns);
       populateForOpInductionVarShapePatterns(patterns);
       if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
         return signalPassFailure();
       }
     }

    // Canonicalize.
    {
      RewritePatternSet patterns(context);
      addVectorCanonicalizationPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }


    if (failed(localExtractInsertOptimizations(funcOp))) 
      return signalPassFailure();

   // if (failed(insertTree(funcOp)))
   //   return signalPassFailure();


     bool shapesRemain = false;
     funcOp->walk([&](vector::ShapeCastOp shapeCastOp) { shapesRemain = true; });
     if (shapesRemain) {
       llvm::errs() << "\n\nfuncOp at this point is \n\n" << funcOp << "\n\n";
       return signalPassFailure();
     }


    // Less desirable unrolls, delayed till here in case previous
    // canonicalization can eliminate them.
    {
      RewritePatternSet patterns(context);
      // shape_cast to extract-like and insert-like ops.
      vector::populateVectorShapeCastLoweringPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Canonicalize.
    {
      RewritePatternSet patterns(context);
      addVectorCanonicalizationPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
