// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/Dialect/UKernelOps.h"
#include "iree/compiler/Codegen/Microkernels/CUDA/uCUDAContract.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/CUDA/CUDATarget.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
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
  }
  void runOnOperation() override;
};

/// Generate microkernel names based on combinedOps
static std::string generateMicrokernelName(ArrayRef<Operation *> combinedOps,
                                           StringRef typeName, int TILE_M,
                                           int TILE_N, int TILE_K,
                                           int numstages, bool has_fill,
                                           bool writeback_to_global) {
  uGPUKernel ukernel(typeName.str(), typeName.str(), {TILE_M, TILE_N, TILE_K},
                     numstages, has_fill, writeback_to_global);
  return ukernel.generate_ukernel_name();
}

static FailureOr<StringRef> returnCtype(Type type) {
  if (type.isF32()) return StringRef("float");
  if (type.isInteger(32)) return StringRef("int");
  if (type.isUnsignedInteger(32)) return StringRef("unsigned");
  return failure();
}

/// Lowers linalg's matmul op into micro kernel call op.
struct MatmulConversion : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  Optional<unsigned> stages = std::nullopt;

  MatmulConversion(MLIRContext *context, unsigned softwarePipeline)
      : OpRewritePattern<linalg::MatmulOp>(context) {
    stages = softwarePipeline;
  }

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Operation *> combinedOps = {matmulOp};
    Location loc = matmulOp.getLoc();

    Type elementType = matmulOp.getInputs()
                           .front()
                           .getType()
                           .cast<TensorType>()
                           .getElementType();
    FailureOr<StringRef> strType = returnCtype(elementType);

    if (failed(strType))
      return rewriter.notifyMatchFailure(matmulOp, "Not supported data type");

    Value lhs = matmulOp.getDpsInputOperand(0)->get();
    Value rhs = matmulOp.getDpsInputOperand(1)->get();
    Value out = matmulOp.getDpsInitOperand(0)->get();

    // Step 1. Find out the tile sizes
    SmallVector<int64_t> tiles = getTileSizes(matmulOp, 0);
    if (tiles.size() < 2)
      return rewriter.notifyMatchFailure(matmulOp, "Tiling is not sufficient");

    // Step 2. Find out and make sure pipeline stages is present
    if (!stages.has_value()) {
      return matmulOp->emitError("Expects software pipeline depth\n");
    }

    // Step 3. Fuse linalg.fill with matmul
    Optional<Value> fillValue = std::nullopt;
    auto fillOp = dyn_cast<linalg::FillOp>(out.getDefiningOp());
    bool hasFill;
    if (fillOp) {
      hasFill = true;
      fillValue = fillOp.getDpsInputOperand(0)->get();
      out = fillOp.getDpsInitOperand(0)->get();
    } else {
      hasFill = false;
      fillValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(elementType), elementType);
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

    // Step 5. Generate a name for microkernel
    auto fnName = generateMicrokernelName(
        combinedOps, strType.value(), tiles[0], tiles[1], tiles[2],
        stages.value(), hasFill, !hasConsumer);

    // Step 6. Allocate shared memory for output
    const int shmemSizeOut = tiles[0] * tiles[1];
    Value shmemBufferOut = rewriter.create<bufferization::AllocTensorOp>(
        loc, RankedTensorType::get({tiles[0], tiles[1]}, elementType),
        ValueRange{});

    // Step 7. Allocate shared memory for inputs. Here we reuse outputs shared
    // memory, so we allocate only the remaining.
    Value shmemBufferRemaining = out;
    const int shmemSizeTotal =
        ((tiles[0] * tiles[2]) + (tiles[1] * tiles[2])) * stages.value();
    const int shmemSizeRemaining = shmemSizeTotal - shmemSizeOut;
    if (shmemSizeRemaining > 0) {
      shmemBufferRemaining = rewriter.create<bufferization::AllocTensorOp>(
          loc, RankedTensorType::get({shmemSizeRemaining}, elementType),
          ValueRange{});
    }
    //  todo(guray) Verify that we have sufficient shared memory here

    llvm::errs() << "Tile size: " << tiles[0] << ", " << tiles[1] << ", "
                 << tiles[2] << ", stages = " << stages.value()
                 << " ==> shmem_out:" << shmemSizeOut
                 << ", shmem_remaining: " << shmemSizeRemaining << ", total = "
                 << shmemSizeTotal * elementType.getIntOrFloatBitWidth() / 1024
                 << "kb\n";

    // Step 8. Fill the operands
    SmallVector<Value> ins = {lhs, rhs};
    SmallVector<Value> others, outs;
    if (hasConsumer) {
      ins.push_back(out);
      outs.push_back(shmemBufferOut);
    } else {
      outs.push_back(out);
      others.push_back(shmemBufferOut);
    }
    others.push_back(shmemBufferRemaining);
    others.push_back(fillValue.value());

    // Step 8. Generate the op
    rewriter.replaceOpWithNewOp<IREE::Codegen::UKernelGenericOp>(
        matmulOp, matmulOp.getResultTypes(), StringRef(fnName), ins, outs,
        others);

    return success();
  }
};

}  // namespace

void LLVMGPULowerToUKernelsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());

  IREE::Codegen::TranslationInfoAttr translation =
      getTranslationInfo(getOperation());
  if (!translation) {
    getOperation()->emitError("Expects software pipeline depth\n");
  }
  unsigned stages = translation.getSoftwarePipelineDepth();

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
