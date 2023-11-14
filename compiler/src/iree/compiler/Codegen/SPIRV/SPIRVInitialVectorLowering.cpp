// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVInitialVectorLowering.cpp -------------------------------------===//
//
// This pass hosts initial steps towards lowering vectors ops to meet SPIR-V
// requirements--it applies vector lowering patterns to unroll large n-D vectors
// to 1-D ones that are directly in SPIR-V.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/SPIRV/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-initial-vector-lowering"

namespace mlir {
namespace iree_compiler {
namespace {

void debugPrint(func::FuncOp funcOp, const char *message) {
  LLVM_DEBUG({
    llvm::dbgs() << "//--- " << message << " ---//\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
}

int getComputeVectorSize(int64_t size) {
  for (int i : {4, 3, 2}) {
    if (size % i == 0)
      return i;
  }
  return 1;
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
    if (bitwidth <= 8 && size % 16 == 0)
      return 16;
    if (bitwidth <= 16 && size % 8 == 0)
      return 8;
  }
  if (bitwidth <= 32 && size % 4 == 0)
    return 4;
  return size % 2 == 0 ? 2 : 1;
}

SmallVector<int64_t> getNativeVectorShapeImpl(VectorTransferOpInterface op) {
  auto vecType = op.getVectorType();
  SmallVector<int64_t> nativeSize(vecType.getRank(), 1);
  for (const auto &[index, dim] :
       llvm::enumerate(op.getPermutationMap().getResults())) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(dim)) {
      if (dimExpr.getPosition() == op.getPermutationMap().getNumDims() - 1) {
        nativeSize[index] = getMemoryVectorSize(
            op.source(), vecType.getElementType(), vecType.getShape()[index]);
      }
    }
  }
  return nativeSize;
}

Operation *stripElementBitPatternPreservingParents(Value op) {
  while (Operation *parentOp = op.getDefiningOp()) {
    Value source =
        TypeSwitch<Operation *, Value>(parentOp)
            .Case<vector::BroadcastOp>([](vector::BroadcastOp broadcast) {
              return broadcast.getVector();
            })
            .Case<vector::ExtractOp, vector::ExtractElementOp,
                  vector::ExtractStridedSliceOp>(
                [](auto extract) { return extract.getVector(); })
            .Case<vector::InsertOp, vector::InsertElementOp,
                  vector::InsertStridedSliceOp>(
                [](auto insert) { return insert.getSource(); })
            .Case<vector::TransposeOp>([](vector::TransposeOp transpose) {
              return transpose.getVector();
            })
            .Default([](Operation *) { return nullptr; });

    if (!source)
      break;
    op = source;
  }

  return op.getDefiningOp();
}

/// Returns true when |op| has the i32 element type that is likely to be result
/// of a zero/sign extension from i8.
bool mayExtI8ToI32(Value op) {
  if (!getElementTypeOrSelf(op.getType()).isInteger(32))
    return false;

  // Look through vector operations created by vector unrolling patterns,
  // hoping to find a zero/sign extension op. Note that we do not need to find
  // the exact definition for |op| as the final extension will be matched by
  // other patterns -- we only need a good enough proxy to know that one is
  // likely to be found after canonicalization.
  // TODO(#12543): Implement integer narrowing patterns to be able to tell for
  // sure.
  Operation *def = stripElementBitPatternPreservingParents(op);
  Type inTy;

  if (auto ext = dyn_cast_or_null<arith::ExtSIOp>(def)) {
    inTy = getElementTypeOrSelf(ext.getIn().getType());
  } else if (auto ext = dyn_cast_or_null<arith::ExtUIOp>(def)) {
    inTy = getElementTypeOrSelf(ext.getIn().getType());
  } else {
    return false;
  }

  return inTy.isInteger(8);
}

/// Succeeds when |contract| is a i32 matmul whose LHS and RHS operands may be
/// result of zero/sign extension of i8 inputs.
LogicalResult detectI8ToI32Matmul(vector::ContractionOp contract) {
  if (contract.getKind() != vector::CombiningKind::ADD)
    return failure();

  if (!mayExtI8ToI32(contract.getLhs()) || !mayExtI8ToI32(contract.getRhs()))
    return failure();

  ArrayRef<Attribute> iteratorTypes = contract.getIteratorTypes().getValue();
  if (iteratorTypes.size() != 3)
    return failure();

  return success(vector::isParallelIterator(iteratorTypes[0]) &&
                 vector::isParallelIterator(iteratorTypes[1]) &&
                 vector::isReductionIterator(iteratorTypes[2]));
}

/// Returns the index of the reduction dimension.
unsigned getReductionDim(vector::ContractionOp contract) {
  AffineMap resultMap = contract.getIndexingMapsArray().back();
  ArrayRef<Attribute> iteratorTypes = contract.getIteratorTypes().getValue();
  for (auto [idx, it] : llvm::enumerate(iteratorTypes)) {
    if (vector::isReductionIterator(it)) {
      return idx;
    }
  }

  // Return the last index as a fallback.
  return resultMap.getNumDims() - 1;
}

unsigned getInnermostParallelDim(vector::ContractionOp contract) {
  AffineMap resultMap = contract.getIndexingMapsArray().back();
  return resultMap.getDimPosition(resultMap.getNumResults() - 1);
}

SmallVector<int64_t> getNativeVectorShapeImpl(vector::ContractionOp op,
                                              bool targetSupportsDotProd) {
  // Find the contract dimension to unroll. This depends on whether we use the
  // outer product or inner product lowering. Outer product is the default
  // strategy.
  bool lowerToInnerProd =
      targetSupportsDotProd && succeeded(detectI8ToI32Matmul(op));
  unsigned unrollDim =
      lowerToInnerProd ? getReductionDim(op) : getInnermostParallelDim(op);
  auto iteratorTypes = op.getIteratorTypes().getValue();
  SmallVector<int64_t> nativeSize(iteratorTypes.size(), 1);
  SmallVector<int64_t> bounds;
  op.getIterationBounds(bounds);
  nativeSize[unrollDim] = getComputeVectorSize(bounds[unrollDim]);
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
  assert(srcVectorType.getRank() == 1); // Guaranteed by semantics
  int64_t vectorSize = getComputeVectorSize(srcVectorType.getDimSize(0));
  return {vectorSize};
}

SmallVector<int64_t> getNativeVectorShapeImpl(vector::TransposeOp op) {
  VectorType vectorType = op.getResultVectorType();
  SmallVector<int64_t> nativeSize(vectorType.getRank(), 1);
  nativeSize.back() = getComputeVectorSize(vectorType.getShape().back());
  return nativeSize;
}

SmallVector<int64_t> getNativeVectorShapeImpl(vector::GatherOp op) {
  VectorType vectorType = op.getVectorType();
  SmallVector<int64_t> nativeSize(vectorType.getRank(), 1);
  nativeSize.back() = getComputeVectorSize(vectorType.getShape().back());
  return nativeSize;
}

std::optional<SmallVector<int64_t>>
getNativeVectorShape(Operation *op, bool targetSupportsDotProd) {
  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1) {
    if (auto vecType = llvm::dyn_cast<VectorType>(op->getResultTypes()[0])) {
      SmallVector<int64_t> nativeSize(vecType.getRank(), 1);
      nativeSize.back() = getComputeVectorSize(vecType.getShape().back());
      return nativeSize;
    }
  }

  return TypeSwitch<Operation *, std::optional<SmallVector<int64_t>>>(op)
      .Case<VectorTransferOpInterface, vector::MultiDimReductionOp,
            vector::ReductionOp, vector::TransposeOp, vector::GatherOp>(
          [](auto typedOp) { return getNativeVectorShapeImpl(typedOp); })
      .Case<vector::ContractionOp>([=](auto contract) {
        return getNativeVectorShapeImpl(contract, targetSupportsDotProd);
      })
      .Default([](Operation *) { return std::nullopt; });
}

/// Adds patterns to unroll vector ops to SPIR-V native vector size.
void populateVectorUnrollPatterns(RewritePatternSet &patterns,
                                  bool targetSupportsDotProd) {
  auto options = vector::UnrollVectorOptions().setNativeShapeFn(
      [=](auto op) { return getNativeVectorShape(op, targetSupportsDotProd); });
  vector::populateVectorUnrollPatterns(patterns, options);
}

/// Returns true when the target environment support integer dot product ops.
bool supportsIntegerDotProductOps(func::FuncOp fn) {
  spirv::TargetEnvAttr targetEnvAttr = getSPIRVTargetEnvAttr(fn);
  if (!targetEnvAttr) {
    // Alternatively, check if the function op itself has a target env
    // attribute. This may be preferred in tests.
    targetEnvAttr =
        fn->getAttrOfType<spirv::TargetEnvAttr>(spirv::getTargetEnvAttrName());
    if (!targetEnvAttr)
      return false;
  }

  spirv::TargetEnv targetEnv(targetEnvAttr);
  if (!targetEnv.allows(spirv::Extension::SPV_KHR_integer_dot_product))
    return false;

  // Query all the dot prod capabilities except for the packed one -- none of
  // the vectorization patterns need it.
  if (!targetEnv.allows(spirv::Capability::DotProduct))
    return false;
  if (!targetEnv.allows(spirv::Capability::DotProductInput4x8Bit))
    return false;
  if (!targetEnv.allows(spirv::Capability::DotProductInputAll))
    return false;

  return true;
}

class SPIRVInitialLoweringPass
    : public SPIRVInitialVectorLoweringBase<SPIRVInitialLoweringPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    // vector.gather lowering patterns target scf ops.
    registry.insert<linalg::LinalgDialect, vector::VectorDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    func::FuncOp funcOp = getOperation();

    bool emitIntegerDotProdOps = supportsIntegerDotProductOps(funcOp);

    // First apply vectorization to generate vectors of the original tensor
    // shape for tensor.pad ops.
    {
      RewritePatternSet patterns(context);
      // Pull in additional vectorization patterns in IREE.
      populateVectorizePadPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    debugPrint(funcOp, "after vectorizing tensor.pad");

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

    debugPrint(funcOp, "after peephole optimization");

    // High dimension contraction can appear after vectorizing ops like 1-D
    // convolution. Those 1-D convolution ops typically have a leading unit
    // batch dimension. Try to drop that to map to matmul dimensions better.
    SmallVector<vector::ContractionOp> contractOps;
    funcOp.walk([&](vector::ContractionOp op) {
      if (op.getIteratorTypes().size() > 3)
        contractOps.push_back(op);
    });
    for (vector::ContractionOp op : contractOps) {
      OpBuilder builder(op);
      IRRewriter rewriter(builder);
      (void)vector::castAwayContractionLeadingOneDim(op, rewriter);
    }

    debugPrint(funcOp, "after trimming contract leading unit dims");

    // Fold tensor.extract_slice/insert_slice ops into transfer ops. This helps
    // to remove those tensor slice ops so that we can enable further vector op
    // transformations.
    {
      RewritePatternSet patterns(context);
      vector::TransferReadOp::getCanonicalizationPatterns(patterns, context);
      vector::TransferWriteOp::getCanonicalizationPatterns(patterns, context);
      populateVectorTransferTensorSliceTransforms(patterns);

      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    debugPrint(funcOp, "after folding tensor extract/insert slice ops");

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

    debugPrint(funcOp, "after lowering multi reduction ops");

    // Prepare for SPIR-V integer dot product lowering.
    if (emitIntegerDotProdOps) {
      RewritePatternSet patterns(context);
      vector::populateVectorContractCanonicalizeMatmulToMMT(
          patterns, detectI8ToI32Matmul);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }

      debugPrint(funcOp, "after preparing for SPIR-V dot product lowering");
    }

    // Then unroll vectors to native vector size. We try to use 128-bit
    // vectors for memory access and 4/2/1 vector sizes for computation.
    {
      RewritePatternSet patterns(context);
      populateVectorUnrollPatterns(patterns, emitIntegerDotProdOps);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    debugPrint(funcOp, "after unrolling vector ops");

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

    debugPrint(funcOp, "after lowering size-1 reduction contract ops");

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

    debugPrint(funcOp, "after lowering transpose ops");

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
      populateVectorTransferTensorSliceTransforms(patterns);

      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    debugPrint(funcOp, "after trimming leading unit dims");

    // Lower vector reduction to SPIR-V integer dot product.
    if (emitIntegerDotProdOps) {
      RewritePatternSet patterns(context);
      populateVectorReductionToSPIRVDotProductPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }

      debugPrint(funcOp, "after lowering to SPIR-V dot product");
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVInitialVectorLoweringPass() {
  return std::make_unique<SPIRVInitialLoweringPass>();
}

} // namespace iree_compiler
} // namespace mlir
