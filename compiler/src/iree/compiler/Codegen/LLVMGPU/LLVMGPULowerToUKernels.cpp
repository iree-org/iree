// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>
#include <string>

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/Dialect/UKernelOps.h"
#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/CUDA/CUDATarget.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmgpu-lower-to-ukernels"

namespace mlir {
namespace iree_compiler {
namespace {

struct LLVMGPULowerToUKernelsPass
    : LLVMGPULowerToUKernelsBase<LLVMGPULowerToUKernelsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect>();
    registry.insert<mlir::bufferization::BufferizationDialect>();
  }
  void runOnOperation() override;
};

static void generateMicrokernelName(std::string &fnName,
                                    ArrayRef<StringRef> strTypes,
                                    ArrayRef<int64_t> tiles, int numStages,
                                    bool hasFill, bool hasConsumer) {
  fnName.append("__iree_matmul_");
  fnName.append(strTypes[0]);
  fnName.append("_");
  fnName.append(strTypes[1]);
  fnName.append("_");
  fnName.append(strTypes[2]);
  fnName.append("_");
  fnName.append(std::to_string(tiles[0]));
  fnName.append("_");
  fnName.append(std::to_string(tiles[1]));
  fnName.append("_");
  fnName.append(std::to_string(tiles[2]));
  fnName.append("_");
  fnName.append(std::to_string(numStages));
  fnName.append("_");
  fnName.append(hasFill ? "true" : "false");
  fnName.append("_");
  fnName.append(hasConsumer ? "true" : "false");
}

static LogicalResult returnCtypes(Type lhsType, Type rhsType, Type resType,
                                  SmallVectorImpl<StringRef> &types) {
  if (lhsType.isF32() && rhsType.isF32() && resType.isF32()) {
    types.push_back("tf32"); /* lhs */
    types.push_back("tf32"); /* rhs */
    types.push_back("f32");  /* output */
    return success();
  } else if (lhsType.isF16() && rhsType.isF16()) {
    types.push_back("f16"); /* lhs */
    types.push_back("f16"); /* rhs */
    if (resType.isF32())
      types.push_back("f32"); /* output */
    else
      types.push_back("f16"); /* output */
    return success();
  }  // other types are not implemented yet
  return failure();
}

}  // namespace

FailureOr<Operation *> lowerMatmulToMicrokernel(RewriterBase &rewriter,
                                                linalg::LinalgOp matmulOp,
                                                ArrayRef<int64_t> tiles,
                                                int64_t stages,
                                                Optional<StringRef> ukName) {
  Location loc = matmulOp.getLoc();

  Value lhs = matmulOp.getDpsInputOperand(0)->get();
  Value rhs = matmulOp.getDpsInputOperand(1)->get();
  Value out = matmulOp.getDpsInitOperand(0)->get();

  Type lhsElementType = lhs.getType().cast<TensorType>().getElementType();
  Type rhsElementType = rhs.getType().cast<TensorType>().getElementType();
  Type resElementType = out.getType().cast<TensorType>().getElementType();

  SmallVector<StringRef, 3> strTypes;
  LogicalResult maybeStrTypes =
      returnCtypes(lhsElementType, rhsElementType, resElementType, strTypes);
  if (maybeStrTypes.failed()) {
    return rewriter.notifyMatchFailure(matmulOp, "Not supported data type");
  }

  // Step 1. Find out the tile sizes
  if (tiles.size() < 2) {
    return rewriter.notifyMatchFailure(matmulOp, "Tiling is not sufficient");
  }

  // Step 2. Make sure pipeline stages is valid
  if (stages <= 0) {
    return matmulOp->emitError(
        "Software pipeline depth must be positive integer\n");
  }

  // Step 3. Fuse linalg.fill with matmul
  Optional<Value> fillValue = std::nullopt;
  auto fillOp = dyn_cast<linalg::FillOp>(out.getDefiningOp());
  bool hasFill = false;
  if (fillOp) {
    hasFill = true;
    fillValue = fillOp.getDpsInputOperand(0)->get();
    out = fillOp.getDpsInitOperand(0)->get();
  } else {
    fillValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resElementType));
  }

  // Step 4. Find out if there is any consumer. Consumer needs the result, so
  // microkernel stores it back to the shared memory, otherwise, it could
  // store to global memory for performance reasons
  bool hasConsumer = true;
  if (matmulOp->use_empty()) hasConsumer = false;
  if (matmulOp->hasOneUse()) {
    if (isa<IREE::Flow::DispatchTensorStoreOp>(
            matmulOp->getUses().begin()->getOwner()))
      hasConsumer = false;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Lowers microkernel [Tile = " << tiles[0] << "x" << tiles[1]
                 << "x" << tiles[2] << ", stages = " << stages
                 << ", lhs = " << strTypes[0].str()
                 << ", rhs = " << strTypes[1].str()
                 << ", result = " << strTypes[2].str() << "] \n";
  });

  // Step 5. Generate a name for microkernel
  std::string fnName;
  if (!ukName.has_value()) {
    generateMicrokernelName(fnName, strTypes, tiles, stages, hasFill,
                            hasConsumer);
  } else {
    fnName = ukName->str();
  }

  // Step 6. Allocate shared memory
  Optional<Value> shmemBufferOut = std::nullopt;
  Optional<Value> shmemBufferRemaining = std::nullopt;
  int shmemSizeTotal = 0;
  if (!out.getDefiningOp<tensor::EmptyOp>()) {
    // Step 6.1 For output
    int shmemSizeOut = tiles[0] * tiles[1];
    shmemBufferOut = rewriter.create<bufferization::AllocTensorOp>(
        loc, RankedTensorType::get({tiles[0], tiles[1]}, resElementType),
        ValueRange{});

    // Step 6.2 For inputs: Here we reuse outputs shared memory, but if
    // the inputs needs large space, we allocate the remaining.
    shmemSizeTotal = ((tiles[0] * tiles[2]) + (tiles[1] * tiles[2])) * stages;
    const int shmemSizeRemaining = shmemSizeTotal - shmemSizeOut;
    if (shmemSizeRemaining > 0) {
      shmemBufferRemaining = rewriter.create<bufferization::AllocTensorOp>(
          loc, RankedTensorType::get({shmemSizeRemaining}, resElementType),
          ValueRange{});
    }
  }
  if (!shmemBufferOut.has_value()) shmemBufferOut = out;
  if (!shmemBufferRemaining.has_value()) {
    // Just pass something to match the ABI
    shmemBufferRemaining = rewriter.create<bufferization::AllocTensorOp>(
        loc, RankedTensorType::get({0}, resElementType), ValueRange{});
  }

  // Step 8. Fill the operands
  SmallVector<Value> ins = {lhs, rhs};
  SmallVector<Value> others, outs;
  if (hasConsumer) {
    ins.push_back(out);
    outs.push_back(shmemBufferOut.value());
  } else {
    outs.push_back(out);
    others.push_back(shmemBufferOut.value());
  }
  others.push_back(shmemBufferRemaining.value());
  others.push_back(fillValue.value());

  // Step 9. Generate the microkernel op
  Operation *ukernelOp =
      rewriter.replaceOpWithNewOp<IREE::Codegen::UKernelGenericOp>(
          matmulOp, matmulOp->getResultTypes(), fnName, ins, outs, others,
          rewriter.getDictionaryAttr({}), rewriter.getIndexAttr(0));

  LLVM_DEBUG({
    llvm::dbgs() << "Calling Microkernel `" << fnName << "`, allocated "
                 << shmemSizeTotal * lhsElementType.getIntOrFloatBitWidth() /
                        8 / 1024
                 << " Kb Shared Memory \n";
  });

  return ukernelOp;
}

namespace {
/// Lowers linalg's matmul op into micro kernel call op.
struct MatmulConversion : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  Optional<unsigned> stages = std::nullopt;

  MatmulConversion(MLIRContext *context, Optional<unsigned> softwarePipeline)
      : OpRewritePattern<linalg::MatmulOp>(context) {
    stages = softwarePipeline;
  }

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    // Step 1. Find out the tile sizes
    SmallVector<int64_t> tiles = getTileSizes(matmulOp, 0);

    // Step 2. Check whether stages is present
    if (!stages.has_value())
      return rewriter.notifyMatchFailure(matmulOp,
                                         "Expects software pipeline depth\n");

    return lowerMatmulToMicrokernel(rewriter, matmulOp, tiles, stages.value());
  }
};
}  // namespace

static Optional<unsigned> findPipelineStages(func::FuncOp op) {
  IREE::Codegen::TranslationInfoAttr translation = getTranslationInfo(op);
  if (translation) return translation.getSoftwarePipelineDepth();
  return std::nullopt;
}

void LLVMGPULowerToUKernelsPass::runOnOperation() {
  Optional<unsigned> stages = findPipelineStages(getOperation());
  RewritePatternSet patterns(&getContext());
  patterns.insert<MatmulConversion>(&getContext(), stages);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMGPULowerToUKernelsPass() {
  return std::make_unique<LLVMGPULowerToUKernelsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir