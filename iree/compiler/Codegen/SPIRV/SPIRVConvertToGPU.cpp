// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVConvertToGPUPass.cpp ------------------------------------------===//
//
// Partition computation within dispatch function to workgroups/workitems.
//
//===----------------------------------------------------------------------===//

#include <array>
#include <numeric>

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/KernelDispatchUtils.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Loop utilities
//===----------------------------------------------------------------------===//

/// Serializes the dimensions of the scf.parallel specified in
/// `serializedDimensions`, by creating an nested scf.for operation for each
/// dimension.
// TODO(ravishankarm): Move this into LoopUtils.h in MLIR.
static Operation *serializeDimensions(ConversionPatternRewriter &rewriter,
                                      scf::ParallelOp pLoopOp,
                                      ArrayRef<unsigned> serializedDimensions) {
  assert(!serializedDimensions.empty() &&
         "unhandled corner case of no serializing dims");
  OpBuilder::InsertionGuard guard(rewriter);
  DenseSet<unsigned> serializedDimSet;
  serializedDimSet.insert(serializedDimensions.begin(),
                          serializedDimensions.end());
  assert(serializedDimSet.size() == serializedDimensions.size() &&
         "cannot repeat dimensions during serialization of scf.parallel");
  SmallVector<LoopBounds, 2> newPLoopBounds, forBounds;
  SmallVector<unsigned, 2> permutation;
  auto lbs = pLoopOp.lowerBound();
  auto ubs = pLoopOp.upperBound();
  auto steps = pLoopOp.step();
  for (unsigned i : llvm::seq<unsigned>(0, pLoopOp.getNumLoops())) {
    if (serializedDimSet.count(i)) {
      forBounds.push_back({lbs[i], ubs[i], steps[i]});
    } else {
      newPLoopBounds.push_back({lbs[i], ubs[i], steps[i]});
      permutation.push_back(i);
    }
  }
  permutation.append(serializedDimensions.begin(), serializedDimensions.end());
  return replacePLoopOp(rewriter, pLoopOp, newPLoopBounds, forBounds,
                        permutation);
}

/// Serialize all inner dimensions of a `pLoopOp` starting from `serializeFrom`.
static Operation *serializeDimensionsFrom(ConversionPatternRewriter &rewriter,
                                          scf::ParallelOp pLoopOp,
                                          unsigned serializeFrom) {
  unsigned numLoops = pLoopOp.getNumLoops();
  assert(serializeFrom < numLoops &&
         "unhandled corner case of no serialization");
  SmallVector<unsigned, 2> serializedDimensions;
  for (unsigned dim : llvm::seq(serializeFrom, numLoops))
    serializedDimensions.push_back(dim);
  return serializeDimensions(rewriter, pLoopOp, serializedDimensions);
}

//===----------------------------------------------------------------------===//
// GPU processor ID mapping utilities
//===----------------------------------------------------------------------===//

/// Distributes scf.parallel to processors where `IdOp` is used to get the
/// processor ID and `DimOp` is used to get the number of processors along a
/// dimension. Assumes that the number of processors will be less than equal to
/// the number of iterations of the pLoopOp along all dimensions.
template <typename GPUIdOp, typename GPUCountOp>
static LogicalResult distributeSingleIterationPerProcessor(
    ConversionPatternRewriter &rewriter, scf::ParallelOp pLoopOp,
    bool generateGuard = true) {
  unsigned numLoops = pLoopOp.getNumLoops();
  if (numLoops > 3) {
    pLoopOp =
        cast<scf::ParallelOp>(serializeDimensionsFrom(rewriter, pLoopOp, 3));
    numLoops = 3;
  }
  auto procInfo = getGPUProcessorIdsAndCounts<GPUIdOp, GPUCountOp>(
      rewriter, pLoopOp.getLoc(), numLoops);
  return distributeSingleIterationPerProcessor(rewriter, pLoopOp, procInfo,
                                               generateGuard);
}


//===----------------------------------------------------------------------===//
// Pass and patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Pass to convert from tiled and fused linalg ops into gpu.func.
struct SPIRVConvertToGPUPass
    : public SPIRVConvertToGPUBase<SPIRVConvertToGPUPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect, memref::MemRefDialect,
                    scf::SCFDialect, ShapeDialect>();
  }
  void runOnOperation() override;
};

/// Given the workload return the workgroup count along X obtained by
/// linearizing the workload and dividing by the workgroup size.
static Value getWorkgroupCountX(OpBuilder &builder, Location loc,
                                ArrayRef<Value> values,
                                int64_t workgroupSizeX) {
  AffineExpr expr = builder.getAffineConstantExpr(1);
  for (auto val : enumerate(values)) {
    expr = expr * builder.getAffineSymbolExpr(val.index());
  }
  expr = expr.ceilDiv(workgroupSizeX);
  return linalg::applyMapToValues(
      builder, loc, AffineMap::get(0, values.size(), expr), values)[0];
}

/// Map linalg operation to execute on GPU in parallel by mapping the parallel
/// loops to "GlobalInvocationId".
template <typename LinalgOpTy>
struct MapLinalgOpToGlobalInvocationId
    : public OpConversionPattern<LinalgOpTy> {
  MapLinalgOpToGlobalInvocationId(MLIRContext *context,
                                  PatternBenefit benefit = 1)
      : OpConversionPattern<LinalgOpTy>(context, benefit) {}

  LogicalResult matchAndRewrite(
      LinalgOpTy linalgOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // If marker exists do nothing.
    if (hasMarker(linalgOp)) return failure();
    FuncOp funcOp = linalgOp->template getParentOfType<FuncOp>();
    if (!funcOp) return failure();
    Optional<linalg::LinalgLoops> loops =
        linalg::linalgOpToParallelLoops(rewriter, linalgOp);
    if (!loops) return failure();

    SmallVector<int64_t, 3> workgroupSize(3, 1);
    if (!loops.getValue().empty()) {
      scf::ParallelOp pLoopOp = dyn_cast<scf::ParallelOp>(loops.getValue()[0]);
      // If there are parallel loops partition them to threads using global
      // invocation ID.
      if (pLoopOp) {
        pLoopOp = collapseParallelLoops(rewriter, pLoopOp);
        if (!pLoopOp) return failure();
        if (failed(distributeSingleIterationPerProcessor<GPUGlobalId,
                                                         GPUGlobalCount>(
                rewriter, pLoopOp))) {
          return rewriter.notifyMatchFailure(
              linalgOp, "mapping to GlobalInvocationID failed");
        }
        workgroupSize = {32, 1, 1};
      }
    }
    WorkgroupCountRegionBuilder regionBuilder =
        [&workgroupSize](OpBuilder &b, Location loc,
                         std::array<Value, 3> workload) {
          Value one = b.create<ConstantIndexOp>(loc, 1);
          return std::array<Value, 3>{
              getWorkgroupCountX(b, loc, workload, workgroupSize[0]), one, one};
        };
    if (failed(defineWorkgroupCountRegion(rewriter, funcOp, regionBuilder))) {
      return failure();
    }
    if (failed(updateWorkGroupSize(funcOp, workgroupSize))) {
      return failure();
    }
    rewriter.eraseOp(linalgOp);
    return success();
  }
};

}  // namespace


void SPIRVConvertToGPUPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ConversionTarget target(*context);
  // After this pass Linalg and scf.parallel ops should be gone.
  target.addIllegalOp<scf::ParallelOp>();
  target.addIllegalDialect<linalg::LinalgDialect>();
  // Reshape ops are treated legal since they just change the way the underlying
  // buffer is viewed. These are legalized downstream. They become no ops when
  // lowering to SPIR-V since the SPIR-V code uses linearized arrays.
  target.addLegalOp<memref::CollapseShapeOp, memref::ExpandShapeOp>();
  // Let the rest fall through.
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  OwningRewritePatternList patterns(&getContext());

  patterns.insert<MapLinalgOpToGlobalInvocationId<linalg::CopyOp>,
                  MapLinalgOpToGlobalInvocationId<linalg::FillOp>,
                  MapLinalgOpToGlobalInvocationId<linalg::GenericOp>>(context);
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));

  for (FuncOp funcOp : getOperation().getInnerModule().getOps<FuncOp>()) {
    if (!isEntryPoint(funcOp)) continue;
    Region &body = funcOp.getBody();
    if (!llvm::hasSingleElement(body)) {
      funcOp.emitError("unhandled dispatch function with multiple blocks");
      return signalPassFailure();
    }
    if (failed(applyFullConversion(funcOp, target, frozenPatterns)))
      return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createSPIRVConvertToGPUPass() {
  return std::make_unique<SPIRVConvertToGPUPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
