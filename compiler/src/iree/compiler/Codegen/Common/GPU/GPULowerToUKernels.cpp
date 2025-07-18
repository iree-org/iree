// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPUTileSwizzleUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPULOWERTOUKERNELSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

/// Matches generic that represent argmax and check if
/// we have the ukernel that matches it shape constraint, and types.
/// If we do, then we convert into iree_codegen.ukernel.argmax operation,
/// that is later lowered into a call to the microkernel.
static FailureOr<IREE::Codegen::UKernelOpInterface>
matchArgmaxDAGForUKernel(RewriterBase &rewriter, linalg::GenericOp op) {
  Value input = op.getDpsInputOperand(0)->get();
  Value index = op.getDpsInitOperand(1)->get();
  auto indexType = cast<ShapedType>(index.getType());
  auto loweringConfig = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
  if (!loweringConfig) {
    return rewriter.notifyMatchFailure(op, "no lowering_config on this op");
  }
  IREE::GPU::UKernelConfigAttr ukernelAttr =
      IREE::GPU::getUkernelSpec(loweringConfig);
  if (!ukernelAttr) {
    return rewriter.notifyMatchFailure(op, "no ukernel selected for this op");
  }

  Location loc = op.getLoc();
  // Currently only support 1D reduction, where reduc is on fastest dim.
  // Tiling argmax ukernel is also set to enforce this structure.
  const int kReductionDim = op.getNumLoops() - 1;
  Value reductionDimSize =
      rewriter.create<tensor::DimOp>(loc, input, kReductionDim);
  bool isPureArgmax = op.getResults()[0].use_empty();
  StringRef kernelName = ukernelAttr.getName();
  SmallVector<Type> resultTypes;
  SmallVector<Value> outputs;
  Value val = op.getDpsInitOperand(0)->get();
  Type valType = val.getType();
  outputs = {val, index};
  resultTypes = {valType, indexType};
  Value writeMaxValueFlag = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI1Type(), rewriter.getBoolAttr(!isPureArgmax));

  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, resultTypes, kernelName, ValueRange{input}, outputs,
      ValueRange{reductionDimSize, writeMaxValueFlag},
      ukernelAttr.getDefAttrs(), /*num_strided_outer_dims=*/0);
  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

struct LowerArgmaxToUKernelPattern : OpRewritePattern<linalg::GenericOp> {
  LowerArgmaxToUKernelPattern(MLIRContext *context)
      : OpRewritePattern<linalg::GenericOp>(context) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!IREE::LinalgExt::isArgmaxOp(op)) {
      return failure();
    }
    FailureOr<IREE::Codegen::UKernelOpInterface> ukernelOp =
        matchArgmaxDAGForUKernel(rewriter, op);
    if (failed(ukernelOp)) {
      return rewriter.notifyMatchFailure(
          op, "failed to find microkernel op to replace with");
    }

    bool isPureArgmax = op.getResults()[0].use_empty();

    if (isPureArgmax) {
      rewriter.replaceAllUsesWith(op.getResults()[1],
                                  ukernelOp.value()->getResults()[1]);
    } else {
      auto origResults = op.getResults();
      auto newResults = ukernelOp.value()->getResults();
      if (origResults.size() != newResults.size()) {
        return rewriter.notifyMatchFailure(op, "result count mismatch");
      }
      // rewriter.replaceAllUsesWith(origResults, newResults);
      rewriter.replaceAllUsesWith(op.getResults()[0],
                                  ukernelOp.value()->getResults()[0]);
      rewriter.replaceAllUsesWith(op.getResults()[1],
                                  ukernelOp.value()->getResults()[1]);
    }

    return success();
  }
};

static Value createSharedMemory(PatternRewriter &rewriter, Location loc,
                                int sharedMemoryBytes) {
  RankedTensorType tensorType =
      RankedTensorType::get({sharedMemoryBytes}, rewriter.getI8Type());
  ValueRange dynSizes{};
  if (!sharedMemoryBytes) {
    IREE::Codegen::NullPointerType nullPointerType =
        IREE::Codegen::NullPointerType::get(rewriter.getContext());
    return rewriter.create<IREE::Codegen::NullPointerOp>(loc, nullPointerType);
  }
  auto allocOp =
      rewriter.create<bufferization::AllocTensorOp>(loc, tensorType, dynSizes);
  Attribute sharedAddrSpace = gpu::AddressSpaceAttr::get(
      rewriter.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  allocOp.setMemorySpaceAttr(sharedAddrSpace);
  return allocOp;
}

/// Returns the index of the innermost CrossIntrinsic dimension of the C matrix,
/// if it is static, and std::nullopt if it is dynamic or if there are no
/// CrossIntrinsic dims.
static std::optional<unsigned>
getCInnermostStaticCrossIntrinsicDim(IREE::Codegen::InnerTiledOp op) {
  auto outputType = dyn_cast<ShapedType>(op.getResultTypes()[0]);
  if (!outputType) {
    return std::nullopt;
  }
  auto mma = cast<IREE::GPU::DataTiledMMAAttr>(op.getKind());
  IREE::Codegen::TileSwizzle accSwizzle =
      getSwizzle(mma, IREE::GPU::MMAFragment::Acc);
  SmallVector<IREE::Codegen::TileSwizzle::Dim> swizzleDims;
  for (IREE::Codegen::TileSwizzle::ExpandShapeDimVectorType group :
       accSwizzle.expandShape) {
    swizzleDims.append(group);
  }
  applyPermutationToVector(swizzleDims, accSwizzle.permutation);
  int rankDiff = outputType.getRank() - swizzleDims.size();
  auto crossIntrinsic = IREE::Codegen::TileSwizzle::Dim::Kind::CrossIntrinsic;
  for (size_t e = swizzleDims.size(), swizzleIdx = e - 1; swizzleIdx < e;
       --swizzleIdx) {
    if (swizzleDims[swizzleIdx].kind != crossIntrinsic) {
      continue;
    }
    int outputIdx = swizzleIdx + rankDiff;
    if (outputType.isDynamicDim(outputIdx)) {
      return std::nullopt;
    }
    return outputIdx;
  }
  return std::nullopt;
}

struct LowerInnerTiledMmaToUKernelPattern
    : OpRewritePattern<IREE::Codegen::InnerTiledOp> {
  LowerInnerTiledMmaToUKernelPattern(MLIRContext *context)
      : OpRewritePattern<IREE::Codegen::InnerTiledOp>(context) {}

  LogicalResult matchAndRewrite(IREE::Codegen::InnerTiledOp op,
                                PatternRewriter &rewriter) const override {
    auto loweringConfig = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
    if (!loweringConfig) {
      return rewriter.notifyMatchFailure(op, "no lowering_config on this op");
    }
    IREE::GPU::UKernelConfigAttr ukernelAttr =
        IREE::GPU::getUkernelSpec(loweringConfig);
    if (!ukernelAttr) {
      return rewriter.notifyMatchFailure(op, "no ukernel selected for this op");
    }
    auto mma = dyn_cast<IREE::GPU::DataTiledMMAAttr>(op.getKind());
    if (!mma) {
      return rewriter.notifyMatchFailure(op, "unhandled MMAInterfaceAttr");
    }
    std::optional<int64_t> innerCrossIntrinsicDim =
        getCInnermostStaticCrossIntrinsicDim(op);
    if (!innerCrossIntrinsicDim) {
      return rewriter.notifyMatchFailure(
          op, "inner cross-intrinsic dim is dynamic or not found");
    }
    Location loc = op->getLoc();
    Type I32Type = rewriter.getI32Type();
    auto castIndexToI32 = [&](Value val) {
      return rewriter.create<arith::IndexCastOp>(loc, I32Type, val);
    };
    auto constI32 = [&](int val) {
      return rewriter.create<arith::ConstantIntOp>(loc, I32Type, val);
    };
    int64_t sharedMemoryBytes = ukernelAttr.getSharedMemoryBytes();
    auto sharedMemory = createSharedMemory(rewriter, loc, sharedMemoryBytes);
    Value k = castIndexToI32(
        rewriter.create<tensor::DimOp>(op.getLoc(), op.getInputs()[0], 1));
    Value intrinsicsM = constI32(mma.getIntrinsicsM());
    Value subgroupsM = constI32(mma.getSubgroupsM());
    Value intrinsicsN = constI32(mma.getIntrinsicsN());
    Value subgroupsN = constI32(mma.getSubgroupsN());
    Value intrinsicsK = constI32(mma.getIntrinsicsK());
    // There are 3 shaped input/output operands (A/B/C matrices).
    SmallVector<SmallVector<int64_t>> stridedDims(3, {});
    // Only the C matrix gets strides, and we pass the stride of the innermost
    // CrossIntrinsic dim, because the ukernel needs to know where to store the
    // result vector from each unrolled intrinsic. Offsets into all other
    // dimensions are handled by the compiler, and passed as part of the base
    // pointer + offset. The A and B matrices don't get strides, because we
    // expect them to always be passed as global memory pointers, and the
    // strides can be inferred by the ukernel implementation.
    stridedDims[2].push_back(innerCrossIntrinsicDim.value());
    // The only additional shaped operand is the shared memory buffer. Only
    // create a stride list for it if we have shared memory. Otherwise, the
    // operand is an iree_codegen.null_pointer op.
    if (sharedMemoryBytes != 0) {
      // Shared memory does not need strides.
      stridedDims.push_back({});
    }
    rewriter.replaceOpWithNewOp<IREE::Codegen::UKernelGenericOp>(
        op, op.getOutputs().getTypes(), ukernelAttr.getName(), op.getInputs(),
        op.getOutputs(),
        ValueRange{sharedMemory, constI32(sharedMemoryBytes), k, intrinsicsM,
                   subgroupsM, intrinsicsN, subgroupsN, intrinsicsK},
        ukernelAttr.getDefAttrs(), stridedDims);
    return success();
  }
};

struct GPULowerToUKernelsPass final
    : impl::GPULowerToUKernelsPassBase<GPULowerToUKernelsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    // Enabling a lowering of an op to a microkernel is a trade-off between the
    // potential performance advantage of a microkernel over pure code
    // generation for that op, and the potential benefits of fusions. Indeed,
    // once an op lowered into a microkernel, it will never be fused at any MLIR
    // level. Since microkernels are linked as bitcode, they will still undergo
    // LTO-like optimization in their calling contexts, but we shouldn't expect
    // this to achieve similar results as fusing structured ops.

    // These patterns are unconditionally enabled, because we have strong
    // evidence that it is difficult for codegen to consistently approach
    // microkernels performance, and that consideration overrides the benefit of
    // fusions for these ops.
    patterns
        .add<LowerArgmaxToUKernelPattern, LowerInnerTiledMmaToUKernelPattern>(
            context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
