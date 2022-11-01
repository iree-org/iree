// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Common/LinalgOpInfo.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"

#include "iree/compiler/Codegen/Utils/GPUUtils.h"

using mlir::iree_compiler::IREE::LinalgExt::TilingPatterns;


#define DEBUG_TYPE "iree-spirv-alloc"

namespace mlir {
namespace iree_compiler {

static void populateTilingReductionPatterns(
    RewritePatternSet &patterns,
    IREE::LinalgExt::LinalgTransformationFilter filter) {
  auto getTileSizeFn = [&](OpBuilder &builder, Operation *op) {
    return getTileSizes(builder, op, 2);
  };

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(getTileSizeFn);
  IREE::LinalgExt::TilingPatterns<linalg::BatchMatmulOp, linalg::MatmulOp, linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNchwFchwOp, linalg::DepthwiseConv2DNhwcHwcOp>::insert(
      patterns, tilingOptions, filter);
}

/// Returns true if the index map represents a transpose that benefits from
/// shared mem.
static bool sharedMemTransposeFilter(AffineMap indexMap) {
  if (!indexMap.isEmpty() && indexMap.isPermutation()) {
    // Ensure that the fasted moving dimension (the last one) is permuted,
    // Otherwise shared memory promotion will not benefit the operation.
    if (indexMap.getDimPosition(indexMap.getNumDims() - 1) !=
        indexMap.getNumDims() - 1) {
      return true;
    }
  }
  return false;
}

/// Apply tiling to reduction dimensions based on op attributes.
LogicalResult spirvtileToSerialLoops(func::FuncOp funcOp, bool onlyReduction = true) {
  {
    // Tile again at the workgroup level since redution dimension were
    // ignored. Dimensions already tiled will be ignore since we tile to the
    // same size.
    RewritePatternSet wgTilingPatterns(funcOp.getContext());
    IREE::LinalgExt::LinalgTransformationFilter filter(
        {StringAttr::get(funcOp.getContext(), getWorkgroupMemoryMarker())}, llvm::None);

    populateTilingReductionPatterns(wgTilingPatterns, filter); 
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(wgTilingPatterns)))) {
      return failure();
    }
  }

  {
    RewritePatternSet wgTilingCanonicalizationPatterns =
        linalg::getLinalgTilingCanonicalizationPatterns(funcOp.getContext());
    populateAffineMinSCFCanonicalizationPattern(
        wgTilingCanonicalizationPatterns);
    scf::populateSCFForLoopCanonicalizationPatterns(
        wgTilingCanonicalizationPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(wgTilingCanonicalizationPatterns)))) {
      return failure();
    }
    return success();
  }
}

// Filter to decide which conv op filters can be allocated
static bool convOpFilter(Operation *op) {
  auto linalgOp = cast<linalg::LinalgOp>(op);
  auto operand = linalgOp.getDpsInputOperands()[1];
  spirv::TargetEnvAttr targetEnvAttr = getSPIRVTargetEnvAttr(op);
  spirv::TargetEnv targetEnv(targetEnvAttr);
  spirv::ResourceLimitsAttr limits = targetEnv.getResourceLimits();
  unsigned memorySpace = 1;
  Type filterType = linalgOp.getDpsInputOperand(1)->get().getType();
  ArrayRef<int64_t> filterInputShape = filterType.cast<ShapedType>().getShape();
  for(int i = 0; i < filterInputShape.size(); i++) {
    memorySpace *= filterInputShape[i];
  }
  auto elementBits = operand->get().getType().cast<ShapedType>().getElementType().getIntOrFloatBitWidth();
  memorySpace *= elementBits / 8;
  return limits.getMaxComputeSharedMemorySize() > memorySpace;
}

/// Filter to decide which contract ops need allocations.
static bool contractOpFilter(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) return false;

  // The workgroup specialization already makes static shapes available for the
  // main tile part and makes the partial tile computation small, so promoting
  // to shared memory for the partial tile actually hurts the performance.
  if (linalgOp.hasDynamicShape()) return false;

  SmallVector<unsigned> dims;
  linalgOp.getParallelDims(dims);
  SmallVector<int64_t, 4> shapes = linalgOp.getStaticLoopRanges();
  // Don't promote vector*matrix kind of case.
  int numNonUnitParallelLoop = 0;
  for (unsigned parallelDim : dims) {
    if (shapes[parallelDim] != 1) {
      numNonUnitParallelLoop++;
    }
  }
  return numNonUnitParallelLoop > 1 && linalg::isaContractionOpInterface(op) &&
         linalgOp.getNumParallelLoops() >= 2 &&
         linalgOp.getNumParallelLoops() <= 3;
}

/// Filter to decide which transpose ops need allocations.
static bool transposeOpFilter(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) return false;
  LinalgOpInfo opInfo(linalgOp, sharedMemTransposeFilter);
  return opInfo.isTranspose();
}

/// Returns true if the index map represents a transpose that benefits from
/// shared mem.
static bool isSharedMemTranspose(AffineMap indexMap) {
  if (!indexMap.isEmpty() && indexMap.isPermutation()) {
    // Ensure that the fasted moving dimension (the last one) is permuted,
    // Otherwise shared memory promotion will not benefit the operation.
    if (indexMap.getDimPosition(indexMap.getNumDims() - 1) !=
        indexMap.getNumDims() - 1) {
      return true;
    }
  }
  return false;
}

LogicalResult copyToWorkgroupMem(OpBuilder &builder, Value src, Value dst) {
  Operation *copyOp = builder.create<memref::CopyOp>(src.getLoc(), src, dst);
  setMarker(copyOp, getCopyToWorkgroupMemoryMarker());
  return success();
}

static const char promoteConvMarker[] = "promote_conv";
// static const char promoteTransposeMarker[] = "promote_transpose";

template <typename T>
using LinalgPromotionPattern =
    mlir::iree_compiler::IREE::LinalgExt::LinalgPromotionPattern<T>;

static void populatePromotionPatterns(RewritePatternSet &patterns,
                                      StringAttr replaceMarker) {
  MLIRContext *context = patterns.getContext();
  auto baseOptions =
      linalg::LinalgPromotionOptions()
          .setAllocationDeallocationFns(allocateWorkgroupMemory,
                                        deallocateWorkgroupMemory)
          .setCopyInOutFns(copyToWorkgroupMem, copyToWorkgroupMem)
          .setUseFullTileBuffers({false, false});
  auto promoteConvOptions = baseOptions.setOperandsToPromote({1});
//  auto promoteTransposeOptions = baseOptions.setOperandsToPromote({0});

  IREE::LinalgExt::LinalgTransformationFilter promoteConvFilter(
      {StringAttr::get(context, promoteConvMarker)}, replaceMarker);
/*
  IREE::LinalgExt::LinalgTransformationFilter promoteTransposeFilter(
      {StringAttr::get(context, promoteTransposeMarker)}, replaceMarker);
*/
  patterns.insert<LinalgPromotionPattern<linalg::Conv2DNhwcHwcfOp>,
	          LinalgPromotionPattern<linalg::Conv2DNchwFchwOp>,
		  LinalgPromotionPattern<linalg::DepthwiseConv2DNhwcHwcOp>,
		  LinalgPromotionPattern<linalg::Conv1DNwcWcfOp>>(
      context, promoteConvOptions, promoteConvFilter);
/*
  patterns.insert<LinalgPromotionPattern<linalg::GenericOp>>(
      context, promoteTransposeOptions, promoteTransposeFilter);
*/
}

namespace {
struct SPIRVTensorAllocPass
    : public SPIRVTensorAllocBase<SPIRVTensorAllocPass> {
 public:
  SPIRVTensorAllocPass() {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    
    // Tile the reduction first to reduce the alloc size.
    if (failed(spirvtileToSerialLoops(funcOp))) {
      return signalPassFailure();
    }
   
    SmallVector<Operation *> opsToPromote;
    funcOp.walk([&](Operation *op) {
    
      // TODO: Add Contraction and Tensor Op Patterns
      if(isa<linalg::ConvolutionOpInterface>(*op)) {
        opsToPromote.push_back(op);
       }
      if (transposeOpFilter(op)) {
        opsToPromote.push_back(op);
      }
     });
    for (Operation *op : opsToPromote) {
      OpBuilder builder(op);
      auto linalgOp = cast<linalg::LinalgOp>(op);
      bufferization::BufferizationOptions options;
      // Promote only filter for convolutions
      if (isa<linalg::ConvolutionOpInterface>(*op)) {
        auto operand = linalgOp.getDpsInputOperands()[1]; 
	if (convOpFilter(op)) {
          FailureOr<Value> ret = bufferization::allocateTensorForShapedValue(
            builder, op->getLoc(), operand->get(), false, options, true);
          if (failed(ret)) {
            return signalPassFailure();
          }
          Value v = ret.value();
          operand->set(v);
        }
      }
      if (transposeOpFilter(op)) {
        LinalgOpInfo opInfo(linalgOp, sharedMemTransposeFilter);

        for (auto operand : opInfo.getTransposeOperands()) {
     	  FailureOr<Value> ret = bufferization::allocateTensorForShapedValue(
            builder, op->getLoc(), operand->get(), false, options, true);
          if (failed(ret)) {
            return signalPassFailure();
          }
          Value v = ret.value();
          operand->set(v);
        }
      }
    }
    auto workgroupMarker = StringAttr::get(funcOp.getContext(), getWorkgroupMemoryMarker());
    StringLiteral markerAttrName = IREE::LinalgExt::LinalgTransforms::kLinalgTransformMarker;
    funcOp.walk([&](Operation *op) {
      if(isa<linalg::ConvolutionOpInterface>(*op)) {
        StringAttr promoteMarker = StringAttr::get(funcOp.getContext(), promoteConvMarker);
        op->setAttr(markerAttrName, promoteMarker);
      }
      /*
      if (transposeOpFilter(op)) {
        StringAttr promoteMarker = StringAttr::get(funcOp.getContext(), promoteTransposeMarker);
	op->setAttr(markerAttrName, promoteMarker);
      }
      */
    });

    RewritePatternSet promotionPatterns(&getContext());
    populatePromotionPatterns(promotionPatterns, workgroupMarker);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(promotionPatterns)))) {
      return signalPassFailure();
    }

    funcOp.walk([&](linalg::LinalgOp linalgOp) {
      auto marker = linalgOp->getAttrOfType<StringAttr>(markerAttrName);
      if (!marker) return WalkResult::advance();
      if (marker.getValue() == promoteConvMarker/* || marker.getValue() == promoteTransposeMarker */)
        linalgOp->removeAttr(markerAttrName);
      return WalkResult::advance();
    });
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVTensorAlloc() {
  return std::make_unique<SPIRVTensorAllocPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
