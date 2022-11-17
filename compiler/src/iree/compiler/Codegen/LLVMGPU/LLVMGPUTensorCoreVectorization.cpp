// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Common/GPUPatterns.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "mlir/Dialect/NVGPU/Utils/MMAUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <iostream>

#define DEBUG_LEVEL 0

using mlir::iree_compiler::IREE::LinalgExt::LinalgVectorizationPattern;
using mlir::iree_compiler::IREE::LinalgExt::VectorizationPatterns;

namespace mlir {
namespace iree_compiler {

/// Flag defined in Passes.cpp.
extern llvm::cl::opt<bool> llvmgpuUseMMASync;

//====---------------------------------------------------------------------===//
// Patterns for vectorization
//====---------------------------------------------------------------------===//

static void populateVectorizationPatterns(RewritePatternSet &patterns) {
  IREE::LinalgExt::LinalgTransformationFilter f(
      StringAttr::get(patterns.getContext(), getVectorizeMarker()));
  IREE::LinalgExt::LinalgVectorizationOptions vectorizationOptions;
  VectorizationPatterns<linalg::FillOp, linalg::GenericOp>::insert(
      patterns, vectorizationOptions, f);
  patterns.add<LinalgVectorizationPattern>(
      patterns.getContext(), vectorizationOptions,
      f.addOpFilter<linalg::ContractionOpInterface>());
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
  vector::populateVectorReductionToContractPatterns(patterns);
}

static Optional<SmallVector<int64_t>> unrollOrder(Operation *op) {
  auto contract = dyn_cast<vector::ContractionOp>(op);
  if (!contract) return std::nullopt;
  SmallVector<int64_t> order;
  // Pick an unrolling order that will allow tensorcore operation to reuse LHS
  // register. This is needed to get good performance on sm_80 target.
  // First make reduction the outer dimensions.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isReductionIterator(iter)) {
      order.push_back(index);
    }
  }

  llvm::SmallDenseSet<int64_t> dims;
  for (AffineExpr expr : contract.getIndexingMapsArray()[0].getResults()) {
    dims.insert(expr.cast<AffineDimExpr>().getPosition());
  }
  // Then parallel dimensions that are part of Lhs as we want to re-use Lhs.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isParallelIterator(iter) && dims.count(index)) {
      order.push_back(index);
    }
  }
  // Then the remaining parallel loops.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isParallelIterator(iter) && !dims.count(index)) {
      order.push_back(index);
    }
  }
  return order;
}


// Helper function to return native size for WMMA-based operations.
static Optional<SmallVector<int64_t, 4>> getWmmaNativeVectorSize(    
  Operation *op) {
  // Currently hardcode the size of wmma operation. When more cases are
  // supported this should be picked based on what the backend supports.
  int64_t m = 16;
  int64_t n = 16;
  if (auto contract = dyn_cast<vector::ContractionOp>(op)) {
    int64_t k = contract.getLhsType().getElementType().isF16() ? 16 : 8;
    SmallVector<int64_t> nativeSize(contract.getIteratorTypes().size() - 3, 1);
    nativeSize.append({m, n, k});
    return nativeSize;
  }
  if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
    SmallVector<int64_t> nativeSize(writeOp.getVectorType().getRank() - 2, 1);
    nativeSize.append({m, n});
    return nativeSize;
  }
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    // Transfer read ops may need different shapes based on how they are being
    // used. For simplicity just match the shape used by the extract strided op.
    VectorType sliceType;
    for (Operation *users : op->getUsers()) {
      auto extract = dyn_cast<vector::ExtractStridedSliceOp>(users);
      if (!extract) return std::nullopt;
      auto vecType = extract.getResult().getType().cast<VectorType>();
      if (sliceType && sliceType != vecType) return std::nullopt;
      sliceType = vecType;
    }
    return llvm::to_vector<>(sliceType.getShape());
  }
  if ((OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1)) {
    if (auto vecType = op->getResultTypes()[0].dyn_cast<VectorType>()) {
      SmallVector<int64_t> nativeSize(vecType.getRank() - 2, 1);
      // Map elementwise ops to the output shape.
      nativeSize.append({m, n});
      return nativeSize;
    }
  }
  return std::nullopt;
}

static int getVectorContractOpOperandId(
  vector::ContractionOp contractOp, Operation *op) {
  if (contractOp.getLhs() == op->getResult(0)) return 0;
  if (contractOp.getRhs() == op->getResult(0)) return 1;
  if (contractOp.getAcc() == op->getResult(0)) return 2;
  return -1;
}

// Returns operand id for vector::ContractOp where the vector::TransferReadOp
// is consumed. 
static int getVectorContractOpOperandIdForVectorReadOp(Operation *op) {
  vector::ContractionOp contractOp;

  Operation *firstLevelUser = *((op->getUsers()).begin());
  if (auto contractOp = dyn_cast<vector::ContractionOp>(firstLevelUser))
      return getVectorContractOpOperandId(contractOp, op);
  Operation *secondLevelUser = *((firstLevelUser->getUsers()).begin());
  if (auto contractOp = dyn_cast<vector::ContractionOp>(secondLevelUser))
      return getVectorContractOpOperandId(contractOp, firstLevelUser);
  return -1;
}

// Helper function to return native size for MMA.SYNC-based operations.
static Optional<SmallVector<int64_t>> getMmaNativeVectorSize(
    Operation *op) {
  
  // Shape of native Tensor Core GPU mma.sync operations.
  int64_t mmaShapeM = 16;
  int64_t mmaShapeN = 8;
  int64_t mmaShapeK;

  // Shape the mma.sync warp-level operation.
  if (auto contract = dyn_cast<vector::ContractionOp>(op)) {
    auto sourceType = contract.getLhsType().getElementType();

    // Set mmaShapeK based on sourceType.
    if (sourceType.isInteger(4))
      mmaShapeK = 64;  
    else if (sourceType.isInteger(8))
      mmaShapeK = 32;   
    else if (sourceType.isF16() || sourceType.isBF16())
       mmaShapeK = 16;  
    else if (sourceType.isF32())
      mmaShapeK = 8;   
    else
      return std::nullopt;

    // Initialize/set the starting dims of the ranked shape, such as batch, to 1. 
    SmallVector<int64_t> mmaShape(contract.getIteratorTypes().size() - 3,
                                       1);
    mmaShape.append({mmaShapeM, mmaShapeN, mmaShapeK});
    return mmaShape;
  }

  // Shape of warp-level vector write operation.
  if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
    SmallVector<int64_t> outputShape(writeOp.getVectorType().getRank() - 2,
                                       1);
    outputShape.append({mmaShapeM, mmaShapeN});
    return outputShape;
  }

  // Shape of warp-level vector read (load) operation.
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    auto resultVectorType = readOp.getVector().getType().cast<VectorType>();
    auto resultElementType = resultVectorType.getElementType();

    int operandId = getVectorContractOpOperandIdForVectorReadOp(op);
    if (operandId == -1) {
      op->emitError() << "Cannot determine operandId this "
        "vector::TransferReadOp is used as in the vector::TransferContractOp";
      return std::nullopt;
    }
    
    // Loading F16 values from Shared Memory to Registers.
    if (resultElementType.isF16() || resultElementType.isBF16()) {
      // For matrixC.
      if (operandId == 2) {
        SmallVector<int64_t> readShape;
        readShape.append({mmaShapeM, mmaShapeN});
        // std::cout << "OperandC " << readShape[0] << "x" << readShape[1] << std::endl;
        return readShape;
      } 

      // For matrixA and matrixB.
      if (operandId == 0 || operandId == 1) {
        // MmaSyncOp input operands: matrixA and matrixB. 
        // LDSMx1, x2, x4:
        // - LDSMx1 loads a 1 tile  of 8x8.
        // - LDSMx2 loads a 2 tiles of 8x8.
        // - LDSMx4 loads a 4 tiles of 8x8. (in use)
        // IREE uses the largest tiled load, i.e., LDSMx4. 

        // MmaSyncOp source operand: matrixC.
        // matrixC is also read/written in tiled block of 16x16. In the pass 
        // OptimizeVectorTransfer, matrixC reads are moved above the mainloop 
        // and writes are moved below the mainloop. Thus, mma.sync read/write 
        // accumulator inplace. 

        SmallVector<int64_t> readShape;
        readShape.append({16, 16});
        // std::cout << "OperandA/B " << readShape[0] << "x" << readShape[1] << std::endl;
        return readShape;
      }
    }
    
    // Loading F32 values from Shared Memory to Registers.
    if (resultElementType.isF32()) {
    
    }
  }
  return std::nullopt;
}

static Optional<SmallVector<int64_t>> getGPUTensorCoreNativeVectorSize(
    Operation *op) {
  if (llvmgpuUseMMASync)
    return getMmaNativeVectorSize(op);
  
  return getWmmaNativeVectorSize(op);
}

static void populateVectorUnrollPatterns(RewritePatternSet &patterns) {
  vector::populateVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShapeFn(getGPUTensorCoreNativeVectorSize)
                    .setUnrollTraversalOrderFn(unrollOrder));
}

namespace {
struct LLVMGPUTensorCoreVectorizationPass
    : public LLVMGPUTensorCoreVectorizationBase<
          LLVMGPUTensorCoreVectorizationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();
    {

#if DEBUG_LEVEL
      std::cout << "// ---- Before  LLVMGPUTensorCoreVectorization" << std::endl;
      funcOp->dump();
      std::cout << std::endl;
#endif

      // Step 1. Vectorize.
      RewritePatternSet vectorizationPatterns(context);
      populateVectorizationPatterns(vectorizationPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(vectorizationPatterns)))) {
        return signalPassFailure();
      }

#if DEBUG_LEVEL
      std::cout << "// ---- LLVMGPUTensorCoreVectorization (Vectorize)" << std::endl;
      funcOp->dump();
      std::cout << std::endl;
#endif

      // Step 2. Fold consumer add ops into the contraction op itself.
      RewritePatternSet canonicalizationPatterns(context);
      vector::ContractionOp::getCanonicalizationPatterns(
          canonicalizationPatterns, context);
      populateCombineVectorTransferReadBroadcastPatterns(
          canonicalizationPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(canonicalizationPatterns)))) {
        return signalPassFailure();
      }

#if DEBUG_LEVEL
      std::cout << "// ---- Step 3. LLVMGPUTensorCoreVectorization (Fold consumer add ops)" << std::endl;
      funcOp->dump();
      std::cout << std::endl;
#endif
      // Step 3. Prepare vector operations to be lowered to GPU ops.
      RewritePatternSet vectorContractPatterns(funcOp.getContext());
      mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(vectorContractPatterns);
      mlir::populatePrepareVectorToMMAPatterns(vectorContractPatterns, llvmgpuUseMMASync);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(vectorContractPatterns)))) {
        return signalPassFailure();
      }

#if DEBUG_LEVEL
      std::cout << "// ---- Step 3.  LLVMGPUTensorCoreVectorization (Before Break and unroll)" << std::endl;
      funcOp->dump();
      std::cout << std::endl;
#endif

      // Step 4. Break and unroll warp tile size to native math and load sizes.
      RewritePatternSet vectorUnrollPatterns(context);
      populateVectorUnrollPatterns(vectorUnrollPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(vectorUnrollPatterns)))) {
        return signalPassFailure();
      }

#if DEBUG_LEVEL
      std::cout << "// ---- After LLVMGPUTensorCoreVectorization " << std::endl;
      funcOp->dump();
      std::cout << std::endl;
#endif
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMGPUTensorCoreVectorizationPass() {
  return std::make_unique<LLVMGPUTensorCoreVectorizationPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
