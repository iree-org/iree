// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVVectorize.cpp -------------------------------------------------===//
//
// This pass vectorizes Linalg ops with buffer semantics.
//
//===----------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using mlir::iree_compiler::IREE::LinalgExt::LinalgVectorizationPattern;
using mlir::iree_compiler::IREE::LinalgExt::VectorizationPatterns;

#define DEBUG_TYPE "iree-spirv-vectorize"

namespace mlir {
namespace iree_compiler {
namespace {

int getComputeVectorSize(int64_t size) {
  // Try to use 4 first, and then 2, and then 1.
  return size % 4 == 0 ? 4 : (size % 2 == 0 ? 2 : 1);
}

int getMemoryVectorSize(Value source, Type scalarType, int64_t size) {
  int bitwidth = scalarType.getIntOrFloatBitWidth();
  while (auto sliceOp = source.getDefiningOp<tensor::ExtractSliceOp>())
    source = sliceOp.getSource();
  if (!matchPattern(source, m_Constant())) {
    // If we are not reading from a constant array that is embedded in the
    // kernel, try to use a large vector size matching the bitwidth to read in
    // 128-bit chunks. This helps with memory access performance. Such vector
    // sizes are not native in SPIR-V though; this relies on following passes to
    // bitcast them to 32-bit 4-element vectors to be valid.
    if (bitwidth <= 8 && size % 16 == 0) return 16;
    if (bitwidth <= 16 && size % 8 == 0) return 8;
  }
  if (bitwidth <= 32 && size % 4 == 0) return 4;
  return size % 2 == 0 ? 2 : 1;
}

SmallVector<int64_t> getNativeVectorShapeImpl(VectorTransferOpInterface op) {
  auto vecType = op.getVectorType();
  SmallVector<int64_t> nativeSize(vecType.getRank(), 1);
  for (const auto &[index, dim] :
       llvm::enumerate(op.permutation_map().getResults())) {
    if (auto dimExpr = dim.dyn_cast<AffineDimExpr>()) {
      if (dimExpr.getPosition() == op.permutation_map().getNumDims() - 1) {
        nativeSize[index] = getMemoryVectorSize(
            op.source(), vecType.getElementType(), vecType.getShape()[index]);
      }
    }
  }
  return nativeSize;
}

SmallVector<int64_t> getNativeVectorShapeImpl(vector::ContractionOp op) {
  // Find the contract output's innermost dimension. It is guaranteed to be a
  // parallel dimension due to contract definition. Unroll it with compute
  // size.
  AffineMap resultMap = op.getIndexingMapsArray().back();
  unsigned lastParallelDim =
      resultMap.getDimPosition(resultMap.getNumResults() - 1);
  SmallVector<int64_t> nativeSize(op.getIteratorTypes().size(), 1);
  SmallVector<int64_t, 4> bounds;
  op.getIterationBounds(bounds);
  nativeSize[lastParallelDim] = getComputeVectorSize(bounds[lastParallelDim]);
  return nativeSize;
}

SmallVector<int64_t> getNativeVectorShapeImpl(vector::MultiDimReductionOp op) {
  // Unroll all reduction dimensions by size 1 for vector.multi_reduction.
  VectorType srcVectorType = op.getSourceVectorType();
  auto nativeSize = llvm::to_vector(srcVectorType.getShape());
  auto dims = op.getReductionDims().getAsValueRange<IntegerAttr>();
  for (const auto &dimAttr : dims) {
    nativeSize[dimAttr.getZExtValue()] = 1;
  }
  return nativeSize;
}

SmallVector<int64_t> getNativeVectorShapeImpl(vector::ReductionOp op) {
  VectorType srcVectorType = op.getSourceVectorType();
  assert(srcVectorType.getRank() == 1);  // Guaranteed by semantics
  int64_t vectorSize = getComputeVectorSize(srcVectorType.getDimSize(0));
  return {vectorSize};
}

SmallVector<int64_t> getNativeVectorShapeImpl(vector::TransposeOp op) {
  VectorType vectorType = op.getResultVectorType();
  SmallVector<int64_t> nativeSize(vectorType.getRank(), 1);
  nativeSize.back() = getComputeVectorSize(vectorType.getShape().back());
  return nativeSize;
}

Optional<SmallVector<int64_t>> getNativeVectorShape(Operation *op) {
  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1) {
    if (auto vecType = op->getResultTypes()[0].dyn_cast<VectorType>()) {
      SmallVector<int64_t> nativeSize(vecType.getRank(), 1);
      nativeSize.back() = getComputeVectorSize(vecType.getShape().back());
      return nativeSize;
    }
  }

  return TypeSwitch<Operation *, Optional<SmallVector<int64_t>>>(op)
      .Case<VectorTransferOpInterface, vector::ContractionOp,
            vector::MultiDimReductionOp, vector::ReductionOp,
            vector::TransposeOp>(
          [](auto typedOp) { return getNativeVectorShapeImpl(typedOp); })
      .Default([](Operation *) { return std::nullopt; });
}

/// Add patterns to vectorize any supported Linalg ops.
void populateVectorizationPatterns(RewritePatternSet &patterns) {
  IREE::LinalgExt::LinalgTransformationFilter f;
  IREE::LinalgExt::LinalgVectorizationOptions vectorizationOptions;
  VectorizationPatterns<linalg::FillOp, linalg::GenericOp>::insert(
      patterns, vectorizationOptions, f);
  linalg::populateConvolutionVectorizationPatterns(patterns);
  patterns.add<LinalgVectorizationPattern>(
      patterns.getContext(), vectorizationOptions,
      f.addOpFilter<linalg::ContractionOpInterface>());
}

/// Adds patterns to unroll vector ops to SPIR-V native vector size.
void populateVectorUnrollPatterns(RewritePatternSet &patterns) {
  auto options =
      vector::UnrollVectorOptions().setNativeShapeFn(getNativeVectorShape);
  vector::populateVectorUnrollPatterns(patterns, options);
}

/// Vectorizes Linalg ops on buffer semantics.
class SPIRVVectorizePass : public SPIRVVectorizeBase<SPIRVVectorizePass> {
 public:
  SPIRVVectorizePass() = default;
  SPIRVVectorizePass(const SPIRVVectorizePass &pass) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    func::FuncOp funcOp = getOperation();

    // First apply vectorization to generate vectors of the original tensor
    // shape.
    {
      RewritePatternSet patterns(context);
      populateVectorizationPatterns(patterns);
      // Pull in additional vectorization patterns in IREE.
      populateVectorizePadPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After vectorization ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Special peephole optimizations to clean up IR before further processing.
    {
      RewritePatternSet patterns(context);
      // Pull in patterns to shuffle broadcast/transpose ops around in order to
      // cancel them or embed into contract ops. Embedding in the flexible
      // contract ops will help to sustain the structure through various
      // transformations.
      vector::populateVectorReductionToContractPatterns(patterns);
      // Pull in patterns to canonicalize transfer ops.
      vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
      // Fold consumer add ops into the contraction op itself.
      vector::ContractionOp::getCanonicalizationPatterns(patterns, context);
      // Fold transpose ops if possible as we cannot unroll it later.
      vector::TransposeOp::getCanonicalizationPatterns(patterns, context);

      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After peephole optimization ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Fold tensor.extract_slice/insert_slice ops into transfer ops. This helps
    // to remove those tensor slice ops so that we can enable further vector op
    // transformations.
    {
      RewritePatternSet patterns(context);
      vector::TransferReadOp::getCanonicalizationPatterns(patterns, context);
      vector::TransferWriteOp::getCanonicalizationPatterns(patterns, context);

      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After folding tensor extract/insert slice ops ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Lower vector.multi_dimension early if any operand is a transpose op.
    // The lowering itself generates transpose ops. This helps to cancel
    // transpose ops. vector.multi_reduction is arguably a higher level op and
    // the lowering also unrolls the multi_reduction op, so it makes sense to
    // happen before normal unrolling.
    {
      SmallVector<Operation *> reductionOps;
      funcOp.walk([&](vector::MultiDimReductionOp reductionOp) {
        if (llvm::any_of(reductionOp->getOperands(), [](Value operand) {
              return operand.getDefiningOp<vector::TransposeOp>();
            }))
          reductionOps.push_back(reductionOp);
        return WalkResult::advance();
      });
      RewritePatternSet patterns(context);
      vector::populateVectorMultiReductionLoweringPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerParallel);
      if (failed(applyOpPatternsAndFold(reductionOps, std::move(patterns)))) {
        funcOp.emitOpError("vector lowering failed");
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After lowering multi_reduction ops ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Then unroll vectors to native vector size. We try to use 128-bit
    // vectors for memory access and 4/2/1 vector sizes for computation.
    {
      RewritePatternSet patterns(context);
      populateVectorUnrollPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After unrolling vector ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Lower reduction-unrolled vector contract ops. Such contract ops have
    // their reduction dimensions all be one, so we can convert them into
    // elementwise ops.
    {
      RewritePatternSet patterns(context);
      auto options =
          vector::VectorTransformsOptions().setVectorTransformsOptions(
              vector::VectorContractLowering::ParallelArith);
      vector::populateVectorContractLoweringPatterns(patterns, options);
      // The pattern can generate transpose ops. Try to fold it if possible to
      // avoid lowering them into extract/insert later.
      vector::TransposeOp::getCanonicalizationPatterns(patterns, context);
      // It also generates broadcast/extract ops. Clean up them too.
      vector::BroadcastOp::getCanonicalizationPatterns(patterns, context);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, context);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After lowering size-1 reduction contract ops ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Now lower vector transpose given we have handled vector patterns that may
    // generate transpose ops in previous steps. This converts transpose ops
    // into extract and insert pairs, in preparation of further transformations
    // to canonicalize/cancel.
    {
      RewritePatternSet patterns(context);
      auto options =
          vector::VectorTransformsOptions().setVectorTransposeLowering(
              vector::VectorTransposeLowering::EltWise);
      vector::populateVectorTransposeLoweringPatterns(patterns, options);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After lowering transpose ops ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Next run canonicalization to cast away leading size-1 dimensions. They
    // can be generated from vector unrolling and generally cause issues to
    // cancel corresponding read/write or insert/extract op pairs. This also
    // need to happen before hoisting, where we would make certain vectors loop
    // carried. Once that's done, it's hard to handle the leading size-1
    // dimensions across regions.
    {
      RewritePatternSet patterns(context);

      // We need to pull in casting way leading one dims to allow cancelling
      // some read/write ops.
      vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);

      // We may have vector.insert_strided_slice inserting 1-D native vectors
      // into n-D larger vectors with the above. Break that down too. This is a
      // companion transformation of unrolling.
      vector::populateVectorInsertExtractStridedSliceDecompositionPatterns(
          patterns);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, context);

      // Trimming leading unit dims may generate broadcast/shape_cast ops. Clean
      // them up.
      vector::BroadcastOp::getCanonicalizationPatterns(patterns, context);
      vector::ShapeCastOp::getCanonicalizationPatterns(patterns, context);

      vector::TransferReadOp::getCanonicalizationPatterns(patterns, context);
      vector::TransferWriteOp::getCanonicalizationPatterns(patterns, context);

      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After trimming leading unit dims ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Next perform hoisting. This would analyze transfer read/write ops into
    // tensors and hoist them out of loop nests. So after it we have
    // loop-carried vectors, not loop-carried tensors anymore.
    linalg::hoistRedundantVectorTransfersOnTensor(funcOp);
    linalg::hoistRedundantVectorTransfers(funcOp);

    LLVM_DEBUG({
      llvm::dbgs() << "--- After hoisting transfers ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Lower vector transfer permutation map.
    {
      RewritePatternSet patterns(context);
      vector::ExtractStridedSliceOp::getCanonicalizationPatterns(patterns,
                                                                 context);
      vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After lowering transfer ops ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Lower vector broadcast/transpose and contraction.
    {
      RewritePatternSet patterns(context);
      auto options = vector::VectorTransformsOptions()
                         .setVectorTransformsOptions(
                             vector::VectorContractLowering::OuterProduct)
                         .setVectorTransposeLowering(
                             vector::VectorTransposeLowering::EltWise);
      vector::populateVectorBroadcastLoweringPatterns(patterns);
      vector::populateVectorContractLoweringPatterns(patterns, options);
      vector::populateVectorMultiReductionLoweringPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerParallel);
      vector::populateVectorTransposeLoweringPatterns(patterns, options);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After lowering various vector ops ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Run all sorts of canonicalization patterns to clean up again.
    {
      RewritePatternSet patterns(context);
      vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
      vector::InsertOp::getCanonicalizationPatterns(patterns, context);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, context);
      vector::TransferReadOp::getCanonicalizationPatterns(patterns, context);
      vector::TransferWriteOp::getCanonicalizationPatterns(patterns, context);
      vector::ReductionOp::getCanonicalizationPatterns(patterns, context);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVVectorizePass() {
  return std::make_unique<SPIRVVectorizePass>();
}

}  // namespace iree_compiler
}  // namespace mlir
