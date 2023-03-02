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
static llvm::SmallString<512> generateMicrokernelName(
    ArrayRef<Operation *> combinedOps, StringRef typeName, int TILE_M,
    int TILE_N, bool has_fill) {
  uGPUKernel ukernel(typeName.str(), typeName.str(), {TILE_M, TILE_N},
                     has_fill);

  std::string ukernelname = ukernel.generate_ukernel_name();
  return llvm::SmallString<512>(ukernelname);
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
  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Operation *> combinedOps = {matmulOp};

    Type elementType = matmulOp.getInputs()
                           .front()
                           .getType()
                           .cast<TensorType>()
                           .getElementType();
    FailureOr<StringRef> strType = returnCtype(elementType);

    if (failed(strType))
      return rewriter.notifyMatchFailure(matmulOp, "Not supported data type");

    SmallVector<int64_t> tiles = getTileSizes(matmulOp, 0);
    if (tiles.size() < 2)
      return rewriter.notifyMatchFailure(matmulOp, "Tiling is not sufficient");

    Location loc = matmulOp.getLoc();
    Value lhs = matmulOp.getDpsInputOperand(0)->get();
    Value rhs = matmulOp.getDpsInputOperand(1)->get();
    Value out = matmulOp.getDpsInitOperand(0)->get();

    // fuse fill into matmul
    Value fillValue;
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

    StringRef fnName = generateMicrokernelName(combinedOps, strType.value(),
                                               tiles[0], tiles[1], hasFill);

    SmallVector<Value> ins = {lhs, rhs};
    SmallVector<Value> others, outs;

    // create shared memory allocation
    const int kStages = 3;
    int totalShmem = kStages * tiles[0] * tiles[1] * 2 / 4;
    totalShmem = totalShmem - (tiles[0] * tiles[1]);
    RankedTensorType totalShmemType =
        RankedTensorType::get({totalShmem}, elementType);
    Value shmemBuffer = rewriter.create<bufferization::AllocTensorOp>(
        loc, totalShmemType, ValueRange{});

    ins.push_back(out);
    outs.push_back(out);
    others.push_back(shmemBuffer);
    others.push_back(fillValue);
    // } else {
    //   ins.push_back(out);
    //   RankedTensorType rType =
    //       RankedTensorType::get({tiles[0], tiles[1]}, elementType);
    //   Value dummy =
    //       rewriter.create<tensor::EmptyOp>(loc, rType.getShape(),
    //       elementType);
    //   outs.push_back(dummy);
    //   others.push_back(shmemBuffer);
    //   others.push_back(fillValue);
    // }
    rewriter.replaceOpWithNewOp<IREE::Codegen::UKernelGenericOp>(
        matmulOp, matmulOp.getResultTypes(), fnName, ins, outs, others);

    return success();
  }
};

}  // namespace

void LLVMGPULowerToUKernelsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.insert<MatmulConversion>(&getContext());
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
