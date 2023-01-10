// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/EncodingInfo.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace {

/// Get the materialization information from a `iree_linalg_ext.pack` operation.
static FailureOr<IREE::LinalgExt::MaterializeEncodingInfo>
getMaterializationInfo(IREE::LinalgExt::PackOp packOp) {
  IREE::LinalgExt::MaterializeEncodingInfo encodingInfo;
  SmallVector<OpFoldResult> mixedTileSizes = packOp.getMixedTiles();
  encodingInfo.innerTileSizes.reserve(mixedTileSizes.size());
  for (auto tileSize : mixedTileSizes) {
    if (tileSize.is<Value>()) {
      encodingInfo.innerTileSizes.push_back(ShapedType::kDynamic);
    } else {
      encodingInfo.innerTileSizes.push_back(
          tileSize.get<Attribute>().cast<IntegerAttr>().getInt());
    }
  }
  encodingInfo.innerDimsPos = llvm::to_vector(packOp.getInnerDimsPos());
  encodingInfo.outerDimsPerm = llvm::to_vector(packOp.getOuterDimsPerm());
  encodingInfo.srcRank = packOp.getInputRank();
  return encodingInfo;
}

/// Pattern to lower a `flow.dispatch.workgroup_count_from_set_encoding` op.
/// At the Flow level this op uses the logical shape of the tensor
/// as the workload. This gets materialized into an physical tensor
/// Lower this operation accounting for the change of shape from
/// the logical shape to the physical shape. It lowers to
/// a `flow.dispatch.workgroup_count_from_root_dag` where the root
/// is the `pack` op that materialized the encoding.
struct LowerDispatchWorkgroupCountFromSetEncodingOp
    : public OpRewritePattern<
          IREE::Flow::DispatchWorkgroupCountFromSetEncodingOp> {
  LowerDispatchWorkgroupCountFromSetEncodingOp(
      MLIRContext *context,
      IREE::LinalgExt::MaterializeEncodingInfo encodingInfo,
      IREE::LinalgExt::MaterializeEncodingValueFn materializeEncodingValueFn,
      RankedTensorType inputType, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit),
        materializeEncodingInfo(std::move(encodingInfo)),
        materializeEncodingValueFn(materializeEncodingValueFn),
        inputType(inputType) {}

  LogicalResult matchAndRewrite(
      IREE::Flow::DispatchWorkgroupCountFromSetEncodingOp workgroupCountOp,
      PatternRewriter &rewriter) const override {
    ValueRange workload = workgroupCountOp.getOperands();
    // The workload represents the unpacked shape. Get the workload of the
    // packed shape.
    Location loc = workgroupCountOp.getLoc();
    auto innerTileSizes =
        getInnerTileSizesOfr(rewriter, loc, inputType, materializeEncodingInfo,
                             materializeEncodingValueFn);
    if (failed(innerTileSizes)) return failure();
    SmallVector<OpFoldResult> resultShape =
        IREE::LinalgExt::PackOp::getResultShape(
            rewriter, loc, getAsOpFoldResult(workload), *innerTileSizes,
            materializeEncodingInfo.innerDimsPos,
            materializeEncodingInfo.outerDimsPerm);
    resultShape.resize(materializeEncodingInfo.srcRank);

    rewriter
        .replaceOpWithNewOp<IREE::Flow::DispatchWorkgroupCountFromDagRootOp>(
            workgroupCountOp,
            getValueOrCreateConstantIndexOp(rewriter, loc, resultShape));
    return success();
  }

 private:
  IREE::LinalgExt::MaterializeEncodingInfo materializeEncodingInfo;
  IREE::LinalgExt::MaterializeEncodingValueFn materializeEncodingValueFn;
  RankedTensorType inputType;
};

struct TestLLVMCPUMaterializeDispatchWorkgroupCountPass
    : public TestLLVMCPUMaterializeDispatchWorkgroupCountBase<
          TestLLVMCPUMaterializeDispatchWorkgroupCountPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        IREE::Flow::FlowDialect, IREE::HAL::HALDialect, linalg::LinalgDialect,
        IREE::LinalgExt::IREELinalgExtDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};

void TestLLVMCPUMaterializeDispatchWorkgroupCountPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp innerModule = variantOp.getInnerModule();
  llvm::StringMap<IREE::HAL::ExecutableExportOp> entryPoints =
      getAllEntryPoints(innerModule);

  for (func::FuncOp funcOp : innerModule.getOps<func::FuncOp>()) {
    auto exportOp = entryPoints.lookup(funcOp.getName());
    if (!exportOp) continue;

    SmallVector<Operation *> computeOps;
    SmallVector<LoopTilingAndDistributionInfo> tiledLoops;
    if (failed(getComputeOps(funcOp, computeOps, tiledLoops))) {
      funcOp.emitOpError("failed to get compute ops in dispatch");
      return signalPassFailure();
    }
    if (!tiledLoops.empty()) {
      // The entry point already has distribution to workgroups. Do nothing.
      continue;
    }
    SmallVector<int64_t> tileSizes, staticLoopRanges, interchange;
    SmallVector<unsigned> partitionableLoops;
    Operation *dispatchRootOp = nullptr;
    for (auto op : computeOps) {
      if (isa<IREE::LinalgExt::PackOp>(op)) {
        if (dispatchRootOp) return signalPassFailure();
        dispatchRootOp = op;
      }
    }

    // Lower the workgroup count ops.
    RewritePatternSet patterns(context);
    populateLLVMCPUDispatchWorkgroupCountPatterns(patterns, dispatchRootOp);
    if (failed(applyPatternsAndFoldGreedily(exportOp, std::move(patterns)))) {
      exportOp.emitOpError("failed to lower number of workgroups");
      return signalPassFailure();
    }
  }
}

}  // namespace

void populateLLVMCPUDispatchWorkgroupCountPatterns(RewritePatternSet &patterns,
                                                   Operation *dispatchRootOp) {
  MLIRContext *context = patterns.getContext();
  auto packRootOp = dyn_cast_or_null<IREE::LinalgExt::PackOp>(dispatchRootOp);
  if (!packRootOp) return;

  FailureOr<IREE::LinalgExt::MaterializeEncodingInfo> encodingInfo =
      getMaterializationInfo(packRootOp);
  if (failed(encodingInfo)) return;

  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(packRootOp);
  auto materializeEncodingValueFn = getMaterializeEncodingValueFn(targetAttr);
  auto tensorType = packRootOp.getInputType().cast<RankedTensorType>();
  // The LowerDispatchWorkgroupCountFromSetEncodingOp pattern is going to
  // call materializeEncodingValueFn, passing it a tensor type, expecting
  // that tensor type to have a TensorEncodingAttr. The problem is that
  // MaterializeEncoding has already run, rewriting the SetEncoding op
  // and its result tensor, which used to hold the TensorEncodingAttr,
  // into a pack op, whose new result tensor does not anymore have a
  // TensorEncodingAttr. As a work-around for that, we made
  // MaterializeEncoding preserve the TensorEncodingAttr as an attr on the
  // pack op itself, so the present code can read it and reconstruct a
  // tensorTypeWithEncoding, so
  // LowerDispatchWorkgroupCountFromSetEncodingOp can call
  // materializeEncodingValueFn.
  Attribute encodingAttr =
      packRootOp->getAttr(StringAttr::get(context, "encoding"));
  auto tensorTypeWithEncoding = RankedTensorType::Builder(
      tensorType.getShape(), tensorType.getElementType(), encodingAttr);
  patterns.insert<LowerDispatchWorkgroupCountFromSetEncodingOp>(
      context, encodingInfo.value(), materializeEncodingValueFn,
      tensorTypeWithEncoding);
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createTestLLVMCPUMaterializeDispatchWorkgroupCountPass() {
  return std::make_unique<TestLLVMCPUMaterializeDispatchWorkgroupCountPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
