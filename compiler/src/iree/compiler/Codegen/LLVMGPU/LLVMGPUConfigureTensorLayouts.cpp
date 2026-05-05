// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/src/iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/IndexingMapOpInterface.h"

#include "iree/compiler/Codegen/Utils/Utils.h"

#define DEBUG_TYPE "iree-llvmgpu-configure-vector-layouts"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUCONFIGURETENSORLAYOUTSPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

using IREE::GPU::MMASingleSubgroupLayout;
using IREE::VectorExt::NestedLayoutAttr;
using IREE::VectorExt::ToLayoutOp;
using IREE::VectorExt::VectorLayoutInterface;

namespace {

static SmallVector<bool> getPromotedOperands(Operation *op) {
  SmallVector<bool> promotedOperands(op->getNumOperands(), false);

  auto config = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
  if (!config) {
    return promotedOperands;
  }

  std::optional<SmallVector<int64_t>> promoteConfig =
      getPromotedOperandList(config);
  if (!promoteConfig) {
    return promotedOperands;
  }

  for (int64_t operand : promoteConfig.value()) {
    promotedOperands[operand] = true;
  }

  return promotedOperands;
}

static IREE::Codegen::InnerTileDescAttrInterface getIntrinsic(Operation *op) {
  auto config = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
  assert(config && "Cannot find intrinsic from unconfigured op.");

  IREE::Codegen::InnerTileDescAttrInterface mmaIntrinsic = getMmaKind(config);
  assert(mmaIntrinsic && "Cannot find intrinsic in lowering config.");
  return mmaIntrinsic;
}

/// Given two arrays bounds and tile, compute bounds = ceil(bounds / tile).
///
/// If "tile" contains 0, or is smaller than bounds, divide bounds by 1
/// for those values.
///
/// Returns the actual divisor (without zeros or out of bounds) used to compute
/// bounds /= divisor.
FailureOr<SmallVector<int64_t>> divideTile(SmallVector<int64_t> &bounds,
                                           ArrayRef<int64_t> tile) {
  assert(bounds.size() >= tile.size() &&
         "cannot divide bounds with a different rank");

  SmallVector<int64_t> divisor(bounds.size(), 1);
  for (auto [div, size] : llvm::zip(divisor, tile)) {
    if (size == 0) {
      continue;
    }
    div = size;
  }

  for (auto [bound, div] : llvm::zip_equal(bounds, divisor)) {
    bound = llvm::divideCeil(bound, div);
  }

  return divisor;
}

SmallVector<int64_t> applyProjectedPermutation(ArrayRef<int64_t> input,
                                               ArrayRef<int64_t> perm) {
  SmallVector<int64_t> result;
  result.reserve(perm.size());
  for (int64_t dim : perm) {
    result.push_back(input[dim]);
  }
  return result;
}

SmallVector<int64_t> getStridesFromBasis(ArrayRef<int64_t> basis) {
  SmallVector<int64_t> strides(basis.size());
  int64_t currStride = 1;
  for (auto [stride, size] : llvm::reverse(llvm::zip_equal(strides, basis))) {
    stride = currStride;
    currStride *= size;
  }
  return strides;
}

static LogicalResult distributeTilingSizes(Operation *candidate,
                                           IREE::GPU::LoweringConfigAttr config,
                                           IREE::GPU::TilingLevel level,
                                           SmallVector<int64_t> &bounds,
                                           SmallVector<int64_t> &sizes,
                                           SmallVector<int64_t> &strides) {
  if (ShapedType::isDynamicShape(bounds)) {
    candidate->emitError()
        << "Cannot set layouts on a dynamically shaped iteration space";
    return failure();
  }

  FailureOr<IREE::GPU::Basis> basis = IREE::GPU::getBasis(config, level);
  if (failed(basis)) {
    candidate->emitError()
        << "Could not find a subgroup basis from lowering config";
    return failure();
  }

  sizes = applyProjectedPermutation(basis->counts, basis->mapping);
  strides = applyProjectedPermutation(getStridesFromBasis(basis->counts),
                                      basis->mapping);

  if (failed(divideTile(bounds, sizes))) {
    candidate->emitError()
        << "Could not divide bounds over given basis for level: "
        << IREE::GPU::stringifyTilingLevel(level);
    return failure();
  }

  return success();
}

struct ContractionLayout {
  VectorLayoutInterface lhs;
  VectorLayoutInterface rhs;
  VectorLayoutInterface acc;
};

// Get the layouts to use for the contraction given the intrinsic to use and
// number of subgroups on the M and N dimension.
//
// The contraction is expected to have 3 operands: lhs, rhs and acc of the
// contraction and a single accumulator.
static FailureOr<ContractionLayout>
getContractionLayout(Operation *candidate, ArrayRef<int64_t> bounds,
                     ArrayRef<AffineMap> contractIndexingMaps) {
  auto config = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(candidate);
  if (!config) {
    return failure();
  }

  auto intrinsic =
      dyn_cast_if_present<IREE::GPU::MmaInterfaceAttr>(getIntrinsic(candidate));
  if (!intrinsic) {
    return failure();
  }

  int64_t rank = bounds.size();
  FailureOr<VectorContractOpInfo> maybeOpInfo =
      VectorContractOpInfo::inferFromIndexingMaps(contractIndexingMaps);
  if (failed(maybeOpInfo)) {
    return failure();
  }
  VectorContractOpInfo opInfo = maybeOpInfo.value();
  // Get the inner dimensions.
  int64_t innerMDim = opInfo.getMDims().back();
  int64_t innerNDim = opInfo.getNDims().back();
  int64_t innerKDim = opInfo.getKDims().back();

  bool isBlock = intrinsic.isBlockIntrinsic();
  // For block intrinsics the innermost batch dim is consumed by the intrinsic.
  int64_t innerBDim = -1;
  if (isBlock) {
    if (opInfo.getBatchDims().empty()) {
      return candidate->emitError(
          "block intrinsic requires at least one batch dimension");
    }
    innerBDim = opInfo.getBatchDims().back();
  }

  SmallVector<int64_t> batchCounts(bounds);

  // Subgroup distribution layouts.
  SmallVector<int64_t> subgroupCounts, subgroupStrides;
  if (failed(distributeTilingSizes(
          candidate, config, IREE::GPU::TilingLevel::Subgroup, batchCounts,
          subgroupCounts, subgroupStrides))) {
    return failure();
  }

  // Since these MMA intrinsics have a given tile size for each subgroup, we can
  // calculate the batch dimensions without looking at the subgroup layout.
  SmallVector<int64_t> subgroupSize(rank, 1);
  if (isBlock) {
    auto [bSize, mSize, nSize, kSize] = intrinsic.getBMNKShape();
    subgroupSize[innerBDim] = bSize;
    subgroupSize[innerMDim] = mSize;
    subgroupSize[innerNDim] = nSize;
    subgroupSize[innerKDim] = kSize;
  } else {
    auto [mSize, nSize, kSize] = intrinsic.getMNKShape();
    subgroupSize[innerMDim] = mSize;
    subgroupSize[innerNDim] = nSize;
    subgroupSize[innerKDim] = kSize;
  }

  for (auto i : llvm::seq<int64_t>(rank)) {
    batchCounts[i] = llvm::divideCeil(batchCounts[i], subgroupSize[i]);
  }

  // MMA intrinsics can be weird and usually don't have a single subgroup
  // iteration space, so we need to find their value subgroup iteration space
  // individually.
  auto getFragmentLayout = [&](int operandIndex, int64_t blockDim,
                               int64_t outerDim, int64_t innerDim,
                               AffineMap map) -> VectorLayoutInterface {
    // Note that the struct MMASingleSubgroupLayout contains the partial layout
    // for the canonical (M, K) x (K, N) -> (M, N) matmul form. We treat the
    // concrete nested layout as the layout for the innermost M, N, K
    // dimensions.
    SmallVector<int64_t> outerCounts(rank, 1);
    SmallVector<int64_t> elementCounts(rank, 1);
    SmallVector<int64_t> threadCounts(rank, 1);
    SmallVector<int64_t> threadStrides(rank, 0);

    MMASingleSubgroupLayout subgroupLayout =
        IREE::GPU::getSingleSubgroupLayout(intrinsic, operandIndex);

    if (isBlock) {
      // 3D layout: indices 0=Block, 1=outer(M/K), 2=inner(K/N).
      outerCounts[blockDim] = subgroupLayout.outer[0];
      outerCounts[outerDim] = subgroupLayout.outer[1];
      outerCounts[innerDim] = subgroupLayout.outer[2];
      threadCounts[blockDim] = subgroupLayout.thread[0];
      threadCounts[outerDim] = subgroupLayout.thread[1];
      threadCounts[innerDim] = subgroupLayout.thread[2];
      threadStrides[blockDim] = subgroupLayout.tstrides[0];
      threadStrides[outerDim] = subgroupLayout.tstrides[1];
      threadStrides[innerDim] = subgroupLayout.tstrides[2];
      elementCounts[blockDim] = subgroupLayout.element[0];
      elementCounts[outerDim] = subgroupLayout.element[1];
      elementCounts[innerDim] = subgroupLayout.element[2];
    } else {
      outerCounts[outerDim] = subgroupLayout.outer[0];
      outerCounts[innerDim] = subgroupLayout.outer[1];
      threadCounts[outerDim] = subgroupLayout.thread[0];
      threadCounts[innerDim] = subgroupLayout.thread[1];
      threadStrides[outerDim] = subgroupLayout.tstrides[0];
      threadStrides[innerDim] = subgroupLayout.tstrides[1];
      elementCounts[outerDim] = subgroupLayout.element[0];
      elementCounts[innerDim] = subgroupLayout.element[1];
    }

    // Get the fragment layout for the entire iteration space and then project
    // it. This is significantly easier than trying to create a layout for each
    // fragment itself.
    auto fragmentSpaceLayout = NestedLayoutAttr::get(
        map.getContext(), subgroupCounts, batchCounts, outerCounts,
        threadCounts, elementCounts, subgroupStrides, threadStrides);
    return fragmentSpaceLayout.apply(map);
  };

  VectorLayoutInterface lhs =
      getFragmentLayout(IREE::GPU::kMMAOperandLhs, innerBDim, innerMDim,
                        innerKDim, contractIndexingMaps[0]);
  VectorLayoutInterface rhs =
      getFragmentLayout(IREE::GPU::kMMAOperandRhs, innerBDim, innerKDim,
                        innerNDim, contractIndexingMaps[1]);
  VectorLayoutInterface acc =
      getFragmentLayout(IREE::GPU::kMMAOperandAcc, innerBDim, innerMDim,
                        innerNDim, contractIndexingMaps[2]);

  return ContractionLayout{lhs, rhs, acc};
}

SmallVector<int64_t> getIterationSpaceBounds(IndexingMapOpInterface candidate) {
  SmallVector<int64_t> bounds = candidate.getStaticLoopRanges();
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(candidate.getOperation())) {
    std::optional<VectorizationTileSizes> sizes =
        inferSizesFromIR(linalgOp, std::nullopt);
    if (sizes.has_value()) {
      bounds = sizes.value().vectorSizes;
    }
  }
  return bounds;
}

static LogicalResult
setContractionAnchor(IREE::Codegen::InnerTileDescAttrInterface intrinsic,
                     SmallVector<bool> promotedOperands, RewriterBase &rewriter,
                     linalg::LinalgOp contract) {
  // This function should have only be called on a contraction op.
  assert(linalg::isaContractionOpInterface(contract) &&
         "cannot set contraction anchor on non contraction op");

  SmallVector<int64_t> bounds = getIterationSpaceBounds(contract);
  auto layouts =
      getContractionLayout(contract, bounds, contract.getIndexingMapsArray());
  if (failed(layouts)) {
    return contract->emitError("cannot get concrete layout for contraction");
  }

  auto [aLayout, bLayout, cLayout] = *layouts;
  Location loc = contract.getLoc();

  Value lhs = contract->getOperand(0);
  Value rhs = contract->getOperand(1);
  Value acc = contract->getOperand(2);

  // Set layouts for lhs, rhs and acc.
  rewriter.setInsertionPoint(contract);
  auto layoutedLhs = ToLayoutOp::create(rewriter, loc, lhs, aLayout);
  auto layoutedRhs = ToLayoutOp::create(rewriter, loc, rhs, bLayout);
  auto layoutedAcc = ToLayoutOp::create(rewriter, loc, acc, cLayout);

  // Promote matmul lhs and rhs.
  // TODO: This is a hack until layout analysis is improved. The layout analysis
  // should decide where to put these shared memory conversions.
  if (promotedOperands[0]) {
    layoutedLhs.setSharedMemoryConversion(true);
  }

  if (promotedOperands[1]) {
    layoutedRhs.setSharedMemoryConversion(true);
  }

  if (promotedOperands[2]) {
    layoutedAcc.setSharedMemoryConversion(true);
  }

  contract->setOperand(0, layoutedLhs.getResult());
  contract->setOperand(1, layoutedRhs.getResult());
  contract->setOperand(2, layoutedAcc.getResult());

  // Set layout for result.
  rewriter.setInsertionPointAfter(contract);
  auto toLayout =
      ToLayoutOp::create(rewriter, loc, contract->getResult(0), cLayout);
  rewriter.replaceAllUsesExcept(contract->getResult(0), toLayout.getResult(),
                                toLayout);

  return success();
}

static LogicalResult setDerivedThreadConfigLayout(
    IREE::GPU::DerivedThreadConfigAttr config, IndexingMapOpInterface candidate,
    ArrayRef<int64_t> workgroupSize, RewriterBase &rewriter) {

  SmallVector<int64_t> opShape = getIterationSpaceBounds(candidate);
  int64_t opRank = opShape.size();

  SmallVector<int64_t> elementTile = config.getStaticTilingLevelSizes(
      static_cast<unsigned>(IREE::GPU::TilingLevel::Thread),
      candidate.getOperation());

  for (auto [index, size, element] : llvm::enumerate(opShape, elementTile)) {
    if (ShapedType::isDynamic(size)) {
      candidate->emitError() << "opShape could not be inferred";
      return failure();
    }

    if (size % element != 0) {
      candidate->emitError()
          << "Operation with unsupported number of elements. "
             "Chosen vector tile sizes for operation are not "
             "divisible by operation loop ranges at dim: "
          << index << ", size=" << size << ", vector size = " << element;
      return failure();
    }

    size /= element;
  }

  SmallVector<int64_t> threadTile(opRank, 1);
  SmallVector<int64_t> threadStrides(opRank, 0);

  int64_t residualThreads = ShapedType::getNumElements(workgroupSize);
  int64_t currStride = 1;

  for (auto [tile, stride, size] :
       llvm::reverse(llvm::zip(threadTile, threadStrides, opShape))) {
    int64_t threadBlock;
    if (residualThreads % size == 0) {
      threadBlock = size;
    } else if (size % residualThreads == 0) {
      threadBlock = residualThreads;
    } else {
      candidate->emitError()
          << "Operation with unsupported number of elements.";
      return failure();
    }

    tile = threadBlock;
    stride = currStride;
    size /= threadBlock;

    currStride *= threadBlock;
    residualThreads /= threadBlock;
  }

  SmallVector<int64_t> subgroupTile(opRank, 1);
  SmallVector<int64_t> subgroupStrides(opRank, 0);
  SmallVector<int64_t> outerTile(opRank, 1);

  MLIRContext *context = rewriter.getContext();
  auto layout = IREE::VectorExt::NestedLayoutAttr::get(
      context, subgroupTile, opShape, outerTile, threadTile, elementTile,
      subgroupStrides, threadStrides);

  Location loc = candidate->getLoc();
  auto dpsOp = cast<DestinationStyleOpInterface>(candidate.getOperation());

  rewriter.setInsertionPointAfter(candidate.getOperation());
  for (OpResult result : candidate->getResults()) {
    AffineMap resultMap = candidate.getMatchingIndexingMap(
        dpsOp.getDpsInitOperand(result.getResultNumber()));
    auto toLayout =
        ToLayoutOp::create(rewriter, loc, result, layout.apply(resultMap));
    rewriter.replaceAllUsesExcept(result, toLayout, toLayout);
  }

  return success();
}

static LogicalResult setIntrinsicLoweringConfigLayout(
    IREE::GPU::LoweringConfigAttr config, linalg::LinalgOp candidate,
    ArrayRef<int64_t> workgroupSize, RewriterBase &rewriter) {

  SmallVector<bool> promotedOperands = getPromotedOperands(candidate);
  IREE::Codegen::InnerTileDescAttrInterface intrinsic = getIntrinsic(candidate);

  if (linalg::isaContractionOpInterface(candidate)) {
    if (succeeded(setContractionAnchor(intrinsic, promotedOperands, rewriter,
                                       candidate))) {
      return success();
    }
  }

  candidate->emitError() << "Unable to set intrinsic layouts on operation "
                            "based on given lowering config: "
                         << config;
  return failure();
}

static LogicalResult setGPULoweringConfigLayout(
    IREE::GPU::LoweringConfigAttr config, IndexingMapOpInterface candidate,
    ArrayRef<int64_t> workgroupSize, RewriterBase &rewriter) {
  MLIRContext *context = config.getContext();
  Location loc = candidate->getLoc();

  SmallVector<int64_t> bounds = getIterationSpaceBounds(candidate);

  // Subgroup distribution layouts.
  SmallVector<int64_t> subgroupSizes, subgroupStrides;
  if (failed(distributeTilingSizes(candidate.getOperation(), config,
                                   IREE::GPU::TilingLevel::Subgroup, bounds,
                                   subgroupSizes, subgroupStrides))) {
    return failure();
  }

  // Thread distribution layouts.
  SmallVector<int64_t> threadSizes, threadStrides;
  if (failed(distributeTilingSizes(candidate.getOperation(), config,
                                   IREE::GPU::TilingLevel::Thread, bounds,
                                   threadSizes, threadStrides))) {
    return failure();
  }

  // Use thread tile sizes as the vector width for each thread.
  SmallVector<int64_t> threadTileSizes = config.getStaticTilingLevelSizes(
      llvm::to_underlying(IREE::GPU::TilingLevel::Thread),
      candidate.getOperation());
  FailureOr<SmallVector<int64_t>> elementTile =
      divideTile(bounds, threadTileSizes);
  if (failed(elementTile)) {
    candidate->emitError() << "Could not divide bounds over given thread tile";
  }
  // The remaining bounds become batch sizes. We could also use subgroup tile
  // sizes, as a way of specifying batch size, but since it is a derived
  // property, we choose to compute it.
  ArrayRef<int64_t> batchTile = bounds;
  SmallVector<int64_t> outerTile(bounds.size(), 1);

  auto layout = IREE::VectorExt::NestedLayoutAttr::get(
      context, subgroupSizes, batchTile, outerTile, threadSizes,
      elementTile.value(), subgroupStrides, threadStrides);

  SmallVector<bool> promotedOperands =
      getPromotedOperands(candidate.getOperation());

  rewriter.setInsertionPoint(candidate.getOperation());
  for (OpOperand &operand : candidate->getOpOperands()) {
    VectorLayoutInterface operandLayout =
        layout.apply(candidate.getMatchingIndexingMap(&operand));
    auto toLayout =
        ToLayoutOp::create(rewriter, loc, operand.get(), operandLayout);
    // Set shared memory promotion if requested.
    toLayout.setSharedMemoryConversion(
        promotedOperands[operand.getOperandNumber()]);
    operand.set(toLayout);
  }

  auto dpsOp = cast<DestinationStyleOpInterface>(candidate.getOperation());
  rewriter.setInsertionPointAfter(candidate.getOperation());
  for (OpResult result : candidate->getResults()) {
    AffineMap resultMap = candidate.getMatchingIndexingMap(
        dpsOp.getDpsInitOperand(result.getResultNumber()));
    auto toLayout =
        ToLayoutOp::create(rewriter, loc, result, layout.apply(resultMap));
    rewriter.replaceAllUsesExcept(result, toLayout, toLayout);
  }

  return success();
}

struct LLVMGPUConfigureTensorLayoutsPass final
    : impl::LLVMGPUConfigureTensorLayoutsPassBase<
          LLVMGPUConfigureTensorLayoutsPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::VectorExt::IREEVectorExtDialect>();
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();
    IRRewriter rewriter(funcOp);

    std::optional<SmallVector<int64_t>> maybeWorkgroupSize =
        getWorkgroupSize(funcOp);
    if (!maybeWorkgroupSize) {
      funcOp->emitOpError()
          << "unable to query workgroup_size information from entry point";
      return signalPassFailure();
    }

    if (failed(setLayoutsFromLoweringConfig(funcOp, maybeWorkgroupSize.value(),
                                            rewriter))) {
      return signalPassFailure();
    }
  }

  LogicalResult setLayoutsFromLoweringConfig(FunctionOpInterface funcOp,
                                             ArrayRef<int64_t> workgroupSize,
                                             RewriterBase &rewriter) {
    SmallVector<Operation *> candidates;
    funcOp.walk([&](Operation *op) {
      if (!isa<linalg::LinalgOp, IREE::LinalgExt::ScanOp>(op) ||
          !getLoweringConfig(op)) {
        return;
      }
      candidates.push_back(op);
    });

    for (Operation *candidate : candidates) {
      auto indexingMapOp = cast<IndexingMapOpInterface>(candidate);
      LogicalResult result =
          TypeSwitch<IREE::Codegen::LoweringConfigAttrInterface, LogicalResult>(
              getLoweringConfig(candidate))
              .Case([&](IREE::GPU::DerivedThreadConfigAttr config) {
                return setDerivedThreadConfigLayout(config, indexingMapOp,
                                                    workgroupSize, rewriter);
              })
              .Case([&](IREE::GPU::LoweringConfigAttr config) {
                if (getMmaKind(config)) {
                  return setIntrinsicLoweringConfigLayout(
                      config, cast<linalg::LinalgOp>(candidate), workgroupSize,
                      rewriter);
                }
                return setGPULoweringConfigLayout(config, indexingMapOp,
                                                  workgroupSize, rewriter);
              })
              .Default(failure());

      if (failed(result)) {
        return failure();
      }
    }

    return success();
  }
};
} // namespace

} // namespace mlir::iree_compiler
