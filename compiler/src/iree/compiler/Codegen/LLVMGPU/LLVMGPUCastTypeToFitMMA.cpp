// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/VectorOpUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {

struct UpcastContractOutput : OpRewritePattern<vector::ContractionOp> {
  UpcastContractOutput(MLIRContext *context, IREE::GPU::MmaAttr intrinsic,
                       PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), intrinsic(intrinsic) {}

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    VectorContractOpInfo opInfo(contractOp);
    if (opInfo.getOpKind() == VectorContractOpInfo::OpKind::UNKNOWN) {
      return rewriter.notifyMatchFailure(contractOp, "unhandled contract kind");
    }

    auto srcCType = dyn_cast<VectorType>(contractOp.getAccType());
    if (!srcCType) {
      return rewriter.notifyMatchFailure(contractOp, "unhandled scalar case");
    }
    auto srcAType = contractOp.getLhsType();
    auto srcBType = contractOp.getRhsType();

    auto [dstAElemType, dstBElemType, dstCElemType] =
        intrinsic.getABCElementTypes();
    auto [dstM, dstN, dstK] = intrinsic.getMNKShape();

    auto srcCElemFType = dyn_cast<FloatType>(srcCType.getElementType());
    auto dstCElemFType = dyn_cast<FloatType>(dstCElemType);
    if (!srcCElemFType || !dstCElemFType ||
        srcCElemFType.getWidth() >= dstCElemFType.getWidth()) {
      return rewriter.notifyMatchFailure(contractOp, "not upcast case");
    }

    if (srcAType.getElementType() != dstAElemType ||
        srcBType.getElementType() != dstBElemType) {
      return rewriter.notifyMatchFailure(contractOp, "a/b type mismatch");
    }

    auto [srcCMIndex, srcCNIndex] = *opInfo.getResultMNIndex();
    auto [srcAKIndex, srcBKIndex] = *opInfo.getOperandKIndex();
    int64_t srcM = srcCType.getShape()[srcCMIndex];
    int64_t srcN = srcCType.getShape()[srcCNIndex];
    int64_t srcK = srcAType.getShape()[srcAKIndex];

    if (srcM % dstM != 0 || srcN % dstN != 0 || srcK % dstK != 0) {
      return rewriter.notifyMatchFailure(contractOp, "shape cannot divide");
    }

    Location loc = contractOp.getLoc();
    auto dstCType = srcCType.clone(dstCElemFType);
    auto extOp =
        rewriter.create<arith::ExtFOp>(loc, dstCType, contractOp.getAcc());
    auto newContractOp = rewriter.create<vector::ContractionOp>(
        loc, contractOp.getLhs(), contractOp.getRhs(), extOp,
        contractOp.getIndexingMaps(), contractOp.getIteratorTypes());
    rewriter.replaceOpWithNewOp<arith::TruncFOp>(contractOp, srcCType,
                                                 newContractOp);
    return success();
  }

private:
  IREE::GPU::MmaAttr intrinsic;
};

struct LLVMGPUCastTypeToFitMMAPass
    : public LLVMGPUCastTypeToFitMMABase<LLVMGPUCastTypeToFitMMAPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();

    llvm::StringLiteral scheduleAttrName =
        IREE::GPU::MMAScheduleAttr::getMnemonic();
    auto scheduleAttr =
        func->getAttrOfType<IREE::GPU::MMAScheduleAttr>(scheduleAttrName);
    if (!scheduleAttr) {
      DictionaryAttr configDict = getTranslationInfo(func).getConfiguration();
      scheduleAttr = dyn_cast_or_null<IREE::GPU::MMAScheduleAttr>(
          configDict.get(scheduleAttrName));
    }
    if (!scheduleAttr) {
      func.emitError() << "mising mma_schedule\n";
      return signalPassFailure();
    }

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<UpcastContractOutput>(context, scheduleAttr.getIntrinsic());

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLLVMGPUCastTypeToFitMMAPass() {
  return std::make_unique<LLVMGPUCastTypeToFitMMAPass>();
}

} // namespace mlir::iree_compiler
