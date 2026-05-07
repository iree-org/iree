// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/ArmNeon/Transforms.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-virtual-vector-lowering"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUVIRTUALVECTORLOWERINGPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {

// Returns true if `type` is a row-major contiguous memref. For such memrefs the
// per-dimension form `[i0, ..., i_{k-1} + idx]` and the linearize/delinearize
// form address the same byte for each lane. Identity-layout memrefs (with
// static or dynamic shape) qualify by construction; an explicit strided layout
// qualifies only when its strides form the row-major contiguous sequence,
// which currently requires a static shape.
static bool isRowMajorContiguousMemRef(MemRefType type) {
  if (type.getLayout().isIdentity()) {
    return true;
  }
  return memref::isStaticShapeAndContiguousRowMajor(type);
}

// IREE-local override of upstream `Gather1DToConditionalLoads` (in
// mlir/lib/Dialect/Vector/Transforms/LowerVectorGather.cpp). Registered with
// higher benefit alongside the upstream pattern so that this version takes
// precedence on its target case (a rank > 1 row-major contiguous memref base)
// and the upstream pattern handles every other case.
//
// Upstream emits, per lane, a
// `delinearize(linearize(offsets, shape) + idx, shape)` to produce N-D load
// indices. It recovers the flatten index behavior from vectorization, which
// corrects the indices for strided memrefs. Because the "OOB access" needs to
// take strides into account, and it recovers the logical indices. It is a
// correct form for loading ops. However, it adds additional computation, which
// is not easy to remove, when it is a contiguous memref. For such cases, the
// vector.load -> LLVM dialect lowering linearizes the indices based on the
// strides, which results in the same physical address. Thus, the correction is
// not needed, because we rely on the vector -> LLVM lowering for the fixup.
// From the upstream doc, the value is target specific, so it is hard to
// upstream this "optimization".
// The root cause is the flatten index behavior of vectorization, and it is not
// an easy fix today.
struct ContiguousMemrefGather1DToConditionalLoads
    : OpRewritePattern<vector::GatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::GatherOp op,
                                PatternRewriter &rewriter) const override {
    VectorType resultTy = op.getType();
    if (resultTy.getRank() != 1) {
      return rewriter.notifyMatchFailure(op, "unsupported rank");
    }
    if (resultTy.isScalable()) {
      return rewriter.notifyMatchFailure(op, "not a fixed-width vector");
    }

    auto memType = dyn_cast<MemRefType>(op.getBase().getType());
    if (!memType) {
      return rewriter.notifyMatchFailure(op, "non-memref base");
    }
    if (memType.getRank() <= 1) {
      return rewriter.notifyMatchFailure(op, "rank-1 memref handled upstream");
    }
    if (!isRowMajorContiguousMemRef(memType)) {
      return rewriter.notifyMatchFailure(op, "not row-major contiguous");
    }

    Location loc = op.getLoc();
    Type elemTy = resultTy.getElementType();
    VectorType elemVecTy = VectorType::get({1}, elemTy);

    Value condMask = op.getMask();
    Value base = op.getBase();
    Value indexVec = rewriter.createOrFold<arith::IndexCastOp>(
        loc, op.getIndexVectorType().clone(rewriter.getIndexType()),
        op.getIndices());
    auto loadOffsets = llvm::to_vector(op.getOffsets());
    Value lastLoadOffset = loadOffsets.back();

    Value result = op.getPassThru();
    BoolAttr nontemporalAttr = nullptr;
    IntegerAttr alignmentAttr = op.getAlignmentAttr();

    for (int64_t i = 0, e = resultTy.getNumElements(); i < e; ++i) {
      int64_t thisIdx[1] = {i};
      Value condition =
          vector::ExtractOp::create(rewriter, loc, condMask, thisIdx);
      Value index = vector::ExtractOp::create(rewriter, loc, indexVec, thisIdx);
      loadOffsets.back() =
          rewriter.createOrFold<arith::AddIOp>(loc, lastLoadOffset, index);

      auto loadBuilder = [&](OpBuilder &b, Location loc) {
        Value load =
            vector::LoadOp::create(b, loc, elemVecTy, base, loadOffsets,
                                   nontemporalAttr, alignmentAttr);
        int64_t zeroIdx[1] = {0};
        Value extracted = vector::ExtractOp::create(b, loc, load, zeroIdx);
        Value newResult =
            vector::InsertOp::create(b, loc, extracted, result, thisIdx);
        scf::YieldOp::create(b, loc, newResult);
      };
      auto passThruBuilder = [result](OpBuilder &b, Location loc) {
        scf::YieldOp::create(b, loc, result);
      };

      result = scf::IfOp::create(rewriter, loc, condition,
                                 /*thenBuilder=*/loadBuilder,
                                 /*elseBuilder=*/passThruBuilder)
                   .getResult(0);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

class LLVMCPUVirtualVectorLoweringPass
    : public impl::LLVMCPUVirtualVectorLoweringPassBase<
          LLVMCPUVirtualVectorLoweringPass> {
public:
  using Base::Base;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    vector::VectorDialect, arm_neon::ArmNeonDialect,
                    LLVM::LLVMDialect, IREE::Util::UtilDialect>();
  }
  void runOnOperation() override;
};

void LLVMCPUVirtualVectorLoweringPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

  auto vectorMultiReductionLowering =
      vector::VectorMultiReductionLowering::InnerReduction;
  auto vectorContractLowering = vector::VectorContractLowering::OuterProduct;
  auto vectorTransferSplit =
      llvm::StringSwitch<vector::VectorTransferSplit>(
          splitVectorTransfersTo.getValue())
          .Case("none", vector::VectorTransferSplit::None)
          .Case("linalg-copy", vector::VectorTransferSplit::LinalgCopy)
          .Case("vector-transfers", vector::VectorTransferSplit::VectorTransfer)
          .Default(vector::VectorTransferSplit::None);

  auto vectorTransformOptions =
      vector::VectorTransformsOptions()
          .setVectorTransformsOptions(vectorContractLowering)
          .setVectorMultiReductionLowering(vectorMultiReductionLowering)
          .setVectorTransferSplit(vectorTransferSplit);

  DictionaryAttr targetConfig;
  if (auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp)) {
    targetConfig = targetAttr.getConfiguration();
  }

  // Target-dependenet patterns.
  {
    if (enableArmI8mm) {
      RewritePatternSet patterns(ctx);
      arm_neon::populateLowerContractionToNeonI8MMPatterns(patterns);
      (void)applyPatternsGreedily(funcOp, std::move(patterns));
    }
  }

  // Target-independent patterns.
  {
    RewritePatternSet patterns(ctx);
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);

    // Unroll ND transfer_gather/scatter to rank 1, lower contiguous ones to
    // vector.transfer_read/write, lower rank-1 non-contiguous ones to
    // vector.gather/scatter.
    IREE::VectorExt::populateVectorTransferGatherScatterLoweringPatterns(
        patterns);
    IREE::VectorExt::TransferGatherOp::getCanonicalizationPatterns(patterns,
                                                                   ctx);
    IREE::VectorExt::TransferScatterOp::getCanonicalizationPatterns(patterns,
                                                                    ctx);
    IREE::VectorExt::populateLowerTransferGatherScatterToVectorPatterns(
        patterns);

    // RVV should be able to lower most of the gather / scatter with indexed
    // load / store.
    if (!targetConfig || !isRISCV(targetConfig) ||
        !hasAnyVFeature(targetConfig)) {
      vector::populateVectorGatherToConditionalLoadPatterns(patterns);
      // Higher benefit so this takes precedence over the upstream
      // `Gather1DToConditionalLoads` on the contiguous-memref case it targets;
      // upstream still owns every other case (rank-1, scalable, tensor base,
      // non-row-major-contiguous memref).
      patterns.add<ContiguousMemrefGather1DToConditionalLoads>(ctx,
                                                               /*benefit=*/100);
    }
    vector::populateVectorGatherLoweringPatterns(patterns);
    vector::populateVectorContractLoweringPatterns(
        patterns, vectorTransformOptions.vectorContractLowering,
        /*benefit=*/1,
        /*disableOuterProductLowering=*/false);
    // This pattern will transform vector loads whose elements are used in a
    // scalar fashion into scalar loads. This will let scalar loads to be folded
    // into broadcast/arithmetic operations and reduce register pressure.
    vector::populateScalarVectorTransferLoweringPatterns(
        patterns, /*benefit=*/1,
        /*allowMultipleUses=*/!(targetConfig && isAArch64(targetConfig)));
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    vector::populateVectorMultiReductionReorderPatterns(
        patterns, vectorMultiReductionLowering);
    vector::populateVectorMultiReductionFlatteningPatterns(
        patterns, vectorMultiReductionLowering);
    vector::populateVectorMultiReductionUnrollingPatterns(
        patterns, vectorMultiReductionLowering);
    populateVectorTransferFullPartialPatterns(patterns, vectorTransformOptions);

    // Lower vector-semantics `iree_codegen.inner_tiled` ops (e.g. data-tiled
    // matmul on CPU) the same way we lower other high-level vector ops: unroll
    // any non-unit iter dim, drop the (now-unit) iter domain, then replace the
    // op with the per-intrinsic ops emitted by its kind's
    // `buildUnderlyingOperations` (`llvm.call_intrinsic` for CPU's
    // `IREE::CPU::DataTiledMMAAttr`). Note that the preceding
    // `DropVectorUnitDimsPass` only handles `vector.*` ops; the op-specific
    // `populateDropInnerTiledUnitDimsPatterns` here handles the iter
    // dimensions of `inner_tiled` itself, so they don't conflict.
    IREE::Codegen::populateUnrollInnerTiledPatterns(patterns);
    IREE::Codegen::populateDropInnerTiledUnitDimsPatterns(patterns);
    IREE::Codegen::populateLowerInnerTiledPatterns(patterns);

    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }

  // The `inner_tiled` lowering patterns above wrap their per-intrinsic ACC
  // distribute/reassemble in `IREE::Util::HoistableConversionOp` pairs so
  // those conversions can sink/hoist out of any reduction loop wrapping the
  // intrinsic. Cancel the surviving inverse pairs in this function — anything
  // that didn't get hoisted out is a trivial round-trip and folds away here.
  if (failed(IREE::Util::eliminateHoistableConversions(funcOp))) {
    return signalPassFailure();
  }
}
} // namespace
} // namespace mlir::iree_compiler
