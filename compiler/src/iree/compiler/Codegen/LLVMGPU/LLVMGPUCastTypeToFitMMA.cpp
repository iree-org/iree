// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/VectorOpUtils.h"
#include "mlir/Analysis/SliceAnalysis.h"
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

#define GEN_PASS_DEF_LLVMGPUCASTTYPETOFITMMAPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

struct UpcastContractOutput final : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    auto maybeOpInfo = VectorContractOpInfo::inferFromIndexingMaps(
        contractOp.getIndexingMapsArray());
    if (failed(maybeOpInfo)) {
      return rewriter.notifyMatchFailure(contractOp, "not a contraction");
    }
    VectorContractOpInfo opInfo = maybeOpInfo.value();

    auto srcCType = dyn_cast<VectorType>(contractOp.getAccType());
    if (!srcCType) {
      return rewriter.notifyMatchFailure(contractOp, "unhandled scalar case");
    }
    auto srcAType = contractOp.getLhsType();
    auto srcBType = contractOp.getRhsType();

    auto intrinsic = contractOp->getAttrOfType<IREE::GPU::MmaInterfaceAttr>(
        "iree.amdgpu.mma");
    if (!intrinsic) {
      return rewriter.notifyMatchFailure(
          contractOp, "could not find iree.amdgpu.mma attribute on contract");
    }
    auto [dstAElemType, dstBElemType, dstCElemType] =
        intrinsic.getABCElementTypes();

    auto srcCElemFType = dyn_cast<FloatType>(srcCType.getElementType());
    auto dstCElemFType = dyn_cast<FloatType>(dstCElemType);
    if (!srcCElemFType || !dstCElemFType ||
        srcCElemFType.getWidth() >= dstCElemFType.getWidth()) {
      return rewriter.notifyMatchFailure(
          contractOp, "unhandled non-floating point or non-upcasting case");
    }

    if (srcAType.getElementType() != dstAElemType ||
        srcBType.getElementType() != dstBElemType) {
      return rewriter.notifyMatchFailure(contractOp, "a/b type mismatch");
    }

    Location loc = contractOp.getLoc();
    auto dstCType = srcCType.clone(dstCElemFType);
    auto extOp =
        rewriter.create<arith::ExtFOp>(loc, dstCType, contractOp.getAcc());
    auto newContractOp = rewriter.create<vector::ContractionOp>(
        loc, contractOp.getLhs(), contractOp.getRhs(), extOp,
        contractOp.getIndexingMaps(), contractOp.getIteratorTypes());
    newContractOp->setDiscardableAttrs(
        contractOp->getDiscardableAttrDictionary());
    rewriter.replaceOpWithNewOp<arith::TruncFOp>(contractOp, srcCType,
                                                 newContractOp);
    return success();
  }
};

static void inferMmaKind(vector::ContractionOp contract) {
  SetVector<Operation *> slice;
  getForwardSlice(contract.getResult(), &slice);

  // Operations in slice are ordered in topological order, so the first
  // to_layout operation we encounter is setting the layout.
  IREE::VectorExt::ToLayoutOp toLayout;
  for (Operation *op : slice) {
    auto candidate = dyn_cast<IREE::VectorExt::ToLayoutOp>(op);
    if (candidate) {
      toLayout = candidate;
      break;
    }
  }

  if (!toLayout) {
    return;
  }

  auto intrinsic =
      dyn_cast_or_null<IREE::GPU::MmaInterfaceAttr>(toLayout.getMmaKindAttr());
  if (!intrinsic) {
    return;
  }

  contract->setAttr("iree.amdgpu.mma", intrinsic);
}

struct LLVMGPUCastTypeToFitMMAPass final
    : impl::LLVMGPUCastTypeToFitMMAPassBase<LLVMGPUCastTypeToFitMMAPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();

    // Set MMA type from config embedded in toLayoutOp of contraction.
    func.walk([&](vector::ContractionOp contract) {
      inferMmaKind(contract);
      if (!contract->hasAttr("iree.amdgpu.mma")) {
        func.emitOpError("Failed to detect valid to_layout consumer of "
                         "vector.contract to infer MMA kind.");
        return signalPassFailure();
      }
    });

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<UpcastContractOutput>(context);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
