// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmgpu-promote-conv-img"

namespace mlir::iree_compiler {

// Returns the vector size to use for the given genericOp considering its
// operand/result element types.
static int getBaseVectorSize(linalg::GenericOp genericOp) {
  static constexpr int copyVectorNumBits = 128;
  assert(genericOp.getNumDpsInits() == 1);
  unsigned resultBW =
      llvm::cast<MemRefType>(genericOp.getDpsInitOperand(0)->get().getType())
          .getElementTypeBitWidth();
  // Check the operand element types. If we have some sub-byte types there, make
  // sure we at least read a full byte for the sub-byte-element operands.
  unsigned operandBW = std::numeric_limits<unsigned>::max();
  for (OpOperand *operand : genericOp.getDpsInputOperands()) {
    unsigned b =
        IREE::Util::getTypeBitWidth(getElementTypeOrSelf(operand->get()));
    operandBW = std::min(operandBW, b);
  }
  int vectorSize = copyVectorNumBits / resultBW;
  if (operandBW < resultBW && operandBW < 8) {
    // Scale up to make sure we read at least a full byte for the
    // sub-byte-element operand.
    vectorSize *= 8 / operandBW;
  }
  return vectorSize;
}

/// Return the shape of copy op that can be vectorized to a
/// transfer_read/transfer_write of size `targetVectorSize`.
static SmallVector<int64_t> getNativeDstShape(linalg::GenericOp copyOp) {
  int targetVectorSize = getBaseVectorSize(copyOp);
  SmallVector<int64_t> dstShape;
  for (int64_t dim : copyOp.getStaticLoopRanges()) {
    // Skip tiling of dimension of size 1 to simplify distribution.
    dstShape.push_back(dim == 1 ? 0 : 1);
  }
  dstShape.back() = targetVectorSize;
  return dstShape;
}

/// Break up the flat id onto the static loop ranges.
static SmallVector<linalg::ProcInfo> getIds(OpBuilder &b, Location loc,
                                            ArrayRef<Range> parallelLoopRanges,
                                            Value flatThreadId) {
  SmallVector<linalg::ProcInfo> infos;
  Value id = flatThreadId;
  AffineExpr d0 = b.getAffineDimExpr(0);
  for (Range r : llvm::reverse(parallelLoopRanges)) {
    linalg::ProcInfo info;
    auto offset = r.offset.dyn_cast<Attribute>();
    auto stride = r.stride.dyn_cast<Attribute>();
    auto size = r.size.dyn_cast<Attribute>();
    assert(offset && stride && size);
    int64_t numThreadsDim = (llvm::cast<IntegerAttr>(size).getInt() -
                             llvm::cast<IntegerAttr>(offset).getInt()) /
                            llvm::cast<IntegerAttr>(stride).getInt();
    Value dimId = id;
    if (infos.size() != parallelLoopRanges.size() - 1)
      dimId =
          affine::makeComposedAffineApply(b, loc, d0 % numThreadsDim, {dimId});
    info.procId = dimId;
    info.nprocs = b.create<arith::ConstantIndexOp>(loc, numThreadsDim);
    info.distributionMethod = linalg::DistributionMethod::Cyclic;
    infos.push_back(info);
    id = affine::makeComposedAffineApply(b, loc, d0.floorDiv(numThreadsDim),
                                         {id});
  }
  std::reverse(infos.begin(), infos.end());
  return infos;
}

/// Return a flattened Id Value by combining the 3D gpu thread IDs.
static Value createFlatId(mlir::FunctionOpInterface funcOp,
                          ArrayRef<int64_t> workgroupSize) {
  OpBuilder b(funcOp.getFunctionBody());
  Type indexType = b.getIndexType();
  AffineExpr d0 = getAffineDimExpr(0, b.getContext());
  AffineExpr d1 = getAffineDimExpr(1, b.getContext());
  AffineExpr d2 = getAffineDimExpr(2, b.getContext());
  Value threadX =
      b.create<gpu::ThreadIdOp>(funcOp.getLoc(), indexType, gpu::Dimension::x);
  Value threadY =
      b.create<gpu::ThreadIdOp>(funcOp.getLoc(), indexType, gpu::Dimension::y);
  Value threadZ =
      b.create<gpu::ThreadIdOp>(funcOp.getLoc(), indexType, gpu::Dimension::z);
  Value flatThreadId = affine::makeComposedAffineApply(
      b, funcOp.getLoc(),
      d0 + workgroupSize[0] * d1 + (workgroupSize[0] * workgroupSize[1]) * d2,
      {threadX, threadY, threadZ});
  return flatThreadId;
}

namespace {
struct LLVMGPUDistributeSharedMemcpyV2Pass
    : public LLVMGPUDistributeSharedMemcpyV2Base<
          LLVMGPUDistributeSharedMemcpyV2Pass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<amdgpu::AMDGPUDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    FailureOr<IREE::HAL::ExecutableExportOp> exportOp = getEntryPoint(funcOp);
    if (failed(exportOp)) {
      // We cannot do anything because we do not have the workgroup size
      // information, but the pass did not fail.
      return;
    }

    MLIRContext *ctx = &getContext();
    auto workgroupSize = getWorkgroupSize(exportOp.value());
    workgroupSize.resize(3, 1);
    SmallVector<linalg::GenericOp> candidates;
    funcOp.walk([&](linalg::GenericOp copyOp) {
      if (hasMarker(copyOp, getCopyToWorkgroupMemoryMarker()))
        candidates.push_back(copyOp);
    });
    if (candidates.empty()) {
      return;
    }

    IRRewriter rewriter(ctx);
    for (auto op : candidates) {
      rewriter.setInsertionPoint(op);

      linalg::TileSizeComputationFunction wgCopyTileSizeFn =
          [](OpBuilder &builder, Operation *operation) {
            SmallVector<Value> tileSizesVal;
            auto copyOp = dyn_cast<linalg::GenericOp>(operation);
            if (!copyOp)
              return tileSizesVal;
            SmallVector<int64_t> staticSize = getNativeDstShape(copyOp);
            for (int64_t dim : staticSize) {
              tileSizesVal.push_back(builder.create<arith::ConstantIndexOp>(
                  operation->getLoc(), dim));
            }
            return tileSizesVal;
          };

      Value flatId = createFlatId(funcOp, workgroupSize);
      auto getCopyThreadProcInfoFn =
          [flatId](OpBuilder &builder, Location loc,
                   ArrayRef<Range> parallelLoopRanges) {
            return getIds(builder, loc, parallelLoopRanges, flatId);
          };
      linalg::LinalgLoopDistributionOptions copyInvocationDistributionOptions;
      copyInvocationDistributionOptions.procInfo = getCopyThreadProcInfoFn;

      auto tilingOptions =
          linalg::LinalgTilingOptions()
              .setLoopType(linalg::LinalgTilingLoopType::Loops)
              .setTileSizeComputationFunction(wgCopyTileSizeFn)
              .setDistributionOptions(copyInvocationDistributionOptions);
      FailureOr<linalg::TiledLinalgOp> res =
          linalg::tileLinalgOp(rewriter, op, tilingOptions);
      if (failed(res)) {
        continue;
      }
      // auto forLoops = llvm::map_to_vector(
      //     res->loops, [](Operation *loop) { return cast<scf::ForOp>(loop); });
      // if (failed(coalesceLoops(rewriter, forLoops))) {
      //   op->emitOpError("failed to coalesce loops");
      //   return signalPassFailure();
      // }
      if (res->tensorResults.empty()) {
        rewriter.eraseOp(op);
      } else {
        rewriter.replaceOp(op, res->tensorResults);
      }
    }
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUDistributeSharedMemcpyV2Pass() {
  return std::make_unique<LLVMGPUDistributeSharedMemcpyV2Pass>();
}

} // namespace mlir::iree_compiler
