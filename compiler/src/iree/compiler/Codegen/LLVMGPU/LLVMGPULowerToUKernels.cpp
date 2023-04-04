// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/Dialect/UKernelOps.h"
#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"
#include "iree/compiler/Codegen/Microkernels/CUDA/uCUDAContract.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/CUDA/CUDATarget.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
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

llvm::cl::opt<bool> clGPUTF32("iree-codegen-llvmgpu-tf32",
                              llvm::cl::desc("use tf32 for 10 bit mantissa"),
                              llvm::cl::init(false));

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
                                           StringRef lhsType, StringRef rhsType,
                                           StringRef resType, int TILE_M,
                                           int TILE_N, int TILE_K,
                                           int numstages, bool has_fill,
                                           bool writeback_to_global) {
  return generate_ukernel_name(lhsType.str(), rhsType.str(), resType.str(),
                               TILE_M, TILE_N, TILE_K, numstages, has_fill,
                               writeback_to_global);
}

static LogicalResult returnCtypes(Type lhsType, Type rhsType, Type resType,
                                  SmallVectorImpl<StringRef> &types,
                                  bool hasTf32) {
  if (lhsType.isF32() && rhsType.isF32() && resType.isF32()) {
    if (hasTf32 && clGPUTF32) {
      types.push_back("tf32"); /* lhs */
      types.push_back("tf32"); /* rhs */
    } else {
      types.push_back("float"); /* lhs */
      types.push_back("float"); /* rhs */
    }
    types.push_back("float"); /* output */
    return success();
  }
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

    Value lhs = matmulOp.getDpsInputOperand(0)->get();
    Value rhs = matmulOp.getDpsInputOperand(1)->get();
    Value out = matmulOp.getDpsInitOperand(0)->get();

    Type lhsElementType = lhs.getType().cast<TensorType>().getElementType();
    Type rhsElementType = rhs.getType().cast<TensorType>().getElementType();
    Type resElementType = out.getType().cast<TensorType>().getElementType();

    FailureOr<TargetInfo> tinfo =
        mlir::iree_compiler::getTargetInfoFromAnyOp(matmulOp);
    if (failed(tinfo))
      return rewriter.notifyMatchFailure(matmulOp,
                                         "Could not find target information");

    SmallVector<StringRef, 3> strTypes;
    LogicalResult maybeStrTypes =
        returnCtypes(lhsElementType, rhsElementType, resElementType, strTypes,
                     tinfo->hasTF32TensorCore);
    if (maybeStrTypes.failed())
      return rewriter.notifyMatchFailure(matmulOp, "Not supported data type");

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
          loc, rewriter.getZeroAttr(resElementType), resElementType);
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
        combinedOps, strTypes[0], strTypes[1], strTypes[2], tiles[0], tiles[1],
        tiles[2], stages.value(), hasFill, !hasConsumer);

    // Step 6. Allocate shared memory for output
    const int shmemSizeOut = tiles[0] * tiles[1];
    Value shmemBufferOut = rewriter.create<bufferization::AllocTensorOp>(
        loc, RankedTensorType::get({tiles[0], tiles[1]}, resElementType),
        ValueRange{});

    // Step 7. Allocate shared memory for inputs. Here we reuse outputs shared
    // memory, so we allocate only the remaining.
    Value shmemBufferRemaining = out;
    const int shmemSizeTotal =
        ((tiles[0] * tiles[2]) + (tiles[1] * tiles[2])) * stages.value();
    const int shmemSizeRemaining = shmemSizeTotal - shmemSizeOut;
    if (shmemSizeRemaining > 0) {
      shmemBufferRemaining = rewriter.create<bufferization::AllocTensorOp>(
          loc, RankedTensorType::get({shmemSizeRemaining}, resElementType),
          ValueRange{});
    } else {
      // Just pass something to match the ABI
      shmemBufferRemaining = rewriter.create<bufferization::AllocTensorOp>(
          loc, RankedTensorType::get({0}, resElementType), ValueRange{});
    }
    //  todo(guray) Verify that we have sufficient shared memory here

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

    LLVM_DEBUG({
      llvm::dbgs() << "Calling Microkernel `" << fnName << "`, needs "
                   << shmemSizeTotal * lhsElementType.getIntOrFloatBitWidth() /
                          8 / 1024
                   << " Kb Shared Memory \n";
    });
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
