// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"
#include <numeric>

#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-gpu-config-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler::IREE::GPU {

LogicalResult setMatmulLoweringConfig(IREE::GPU::TargetAttr target,
                                      mlir::FunctionOpInterface entryPoint,
                                      Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) {
    return failure();
  }

  if (target.getWgp().getMma().empty())
    return failure();

  const int64_t targetSubgroupSize = target.getPreferredSubgroupSize();

  SmallVector<int64_t, 4> bounds = linalgOp.getStaticLoopRanges();
  FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
      mlir::linalg::inferContractionDims(linalgOp);
  if (failed(contractionDims)) {
    return failure();
  }

  if (contractionDims->k.size() < 1 || contractionDims->m.size() < 1 ||
      contractionDims->n.size() < 1) {
    return failure();
  }

  // For now we are not being smart and trying to reshape dimensions to allow
  // for better usage of intrinsics, and instead are tiling all dimensions
  // except the inner most m, n, and k dimensions to 1.
  int64_t mDim = contractionDims->m.back();
  int64_t nDim = contractionDims->n.back();
  int64_t kDim = contractionDims->k.back();

  // Dynamic dims are expected to be taken care of earlier in the pipeline.
  if (ShapedType::isDynamic(bounds[mDim]) ||
      ShapedType::isDynamic(bounds[nDim]) ||
      ShapedType::isDynamic(bounds[kDim])) {
    return failure();
  }

  // Bail out on matvec-like cases.
  if (bounds[mDim] == 1 || bounds[nDim] == 1) {
    return failure();
  }

  Value lhs = linalgOp.getDpsInputOperand(0)->get();
  Value rhs = linalgOp.getDpsInputOperand(1)->get();
  Value init = linalgOp.getDpsInitOperand(0)->get();

  Type lhsElemType = getElementTypeOrSelf(lhs);
  Type rhsElemType = getElementTypeOrSelf(rhs);
  Type initElemType = getElementTypeOrSelf(init);

  GPUMatmulShapeType problem{bounds[mDim], bounds[nDim], bounds[kDim],
                             lhsElemType,  rhsElemType,  initElemType};

  SmallVector<GPUMatmulShapeType> intrinsics;
  intrinsics.reserve(target.getWgp().getMma().size());
  for (IREE::GPU::MMAAttr mma : target.getWgp().getMma()) {
    auto [mSize, nSize, kSize] = mma.getMNKShape();
    auto [aType, bType, cType] = mma.getABCElementTypes();
    if (mma.getSubgroupSize() != targetSubgroupSize)
      continue;
    intrinsics.emplace_back(mSize, nSize, kSize, aType, bType, cType);
  }
  if (intrinsics.empty())
    return failure();

  GPUMMAHeuristicSeeds seeds;

  // Note that the following heuristic seeds are just placeholder values.
  // We need to clean it up and make it adjusting to different targets.
  // See https://github.com/iree-org/iree/issues/16341 for details.
  if (problem.mSize * problem.nSize <= 512 * 512) {
    // For matmuls with small M*N size, we want to distribute M*N onto more
    // workgroups to fill the GPU. Use a smaller bestMNTileCountPerSubgroup
    // and a larger bestKTileCountPerSubgroup.
    seeds = {/*bestSubgroupCountPerWorkgroup=*/4,
             /*bestMNTileCountPerSubgroup=*/4,
             /*bestKTileCountPerSubgroup=*/8};
  } else {
    seeds = {/*bestSubgroupCountPerWorkgroup=*/4,
             /*bestMNTileCountPerSubgroup=*/8,
             /*bestKTileCountPerSubgroup=*/4};
  }

  int64_t maxSharedMemoryBytes = target.getWgp().getMaxWorkgroupMemoryBytes();

  LDBG("Matmul TileAndFuse Config");

  // Infer if lhs or rhs is transposed to help generate better schedule.
  // TODO: Drop this. This is only a consideration for other pipelines.
  SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();
  bool transposedLhs =
      kDim !=
      llvm::cast<AffineDimExpr>(maps[0].getResults().back()).getPosition();
  bool transposedRhs =
      nDim !=
      llvm::cast<AffineDimExpr>(maps[1].getResults().back()).getPosition();

  // First try to find a schedule with an exactly matching intrinsic.
  std::optional<GPUMMASchedule> schedule =
      deduceMMASchedule(problem, intrinsics, seeds, maxSharedMemoryBytes,
                        targetSubgroupSize, transposedLhs, transposedRhs);
  if (!schedule) {
    // Then try again by allowing upcasting accumulator.
    schedule = deduceMMASchedule(
        problem, intrinsics, seeds, maxSharedMemoryBytes, targetSubgroupSize,
        transposedLhs, transposedRhs, /*canUpcastAcc=*/true);
  }

  if (!schedule) {
    LDBG("Failed to deduce TileAndFuse MMA schedule");
    return failure();
  }

  LDBG("Target Subgroup size: " << targetSubgroupSize);
  LDBG("Schedule: sizes [" << schedule->mSize << ", " << schedule->nSize << ", "
                           << schedule->kSize << "]");
  LDBG("Schedule: tile counts [" << schedule->mTileCount << ", "
                                 << schedule->nTileCount << ", "
                                 << schedule->kTileCount << "]");
  LDBG("Schedule: warp counts [" << schedule->mWarpCount << ", "
                                 << schedule->nWarpCount << "]");

  std::array<int64_t, 3> workgroupSize{
      schedule->nWarpCount * targetSubgroupSize, schedule->mWarpCount, 1};

  SmallVector<int64_t> workgroupTileSizes(linalgOp.getNumLoops(), 0);
  SmallVector<int64_t> reductionTileSizes(linalgOp.getNumLoops(), 0);
  SmallVector<int64_t> subgroupTileSizes(linalgOp.getNumLoops(), 0);
  // Tile all batch dimensions with unit size.
  for (int64_t batch : contractionDims->batch) {
    workgroupTileSizes[batch] = 1;
  }

  // Tile all m, n, and k dimensions to 1 except the innermost. Unit dims
  // from this tiling are folded before vectorization.
  for (int64_t m : llvm::drop_end(contractionDims->m)) {
    workgroupTileSizes[m] = 1;
  }
  for (int64_t n : llvm::drop_end(contractionDims->n)) {
    workgroupTileSizes[n] = 1;
  }
  for (int64_t k : llvm::drop_end(contractionDims->k)) {
    reductionTileSizes[k] = 1;
  }

  // Compute the M/N dimension tile size by multiplying subgroup information.
  workgroupTileSizes[mDim] =
      schedule->mWarpCount * schedule->mTileCount * schedule->mSize;
  workgroupTileSizes[nDim] =
      schedule->nWarpCount * schedule->nTileCount * schedule->nSize;

  // Specify the subgroup tile sizes from the mma schedule. This is applied
  subgroupTileSizes[mDim] = schedule->mTileCount;
  subgroupTileSizes[nDim] = schedule->nTileCount;

  // Similarly the reduction tile size is just the post-packing tile count.
  reductionTileSizes[kDim] = schedule->kTileCount;

  IREE::GPU::MmaInterfaceAttr mmaKind =
      target.getWgp().getMma()[schedule->index];

  // Attach the MMA schedule as an attribute to the entry point export function
  // for later access in the pipeline.
  MLIRContext *context = linalgOp.getContext();
  SmallVector<NamedAttribute, 1> attrs;
  Builder b(context);
  attrs.emplace_back(StringAttr::get(context, "workgroup"),
                     b.getIndexArrayAttr(workgroupTileSizes));
  attrs.emplace_back(StringAttr::get(context, "reduction"),
                     b.getIndexArrayAttr(reductionTileSizes));
  attrs.emplace_back(StringAttr::get(context, "subgroup"),
                     b.getIndexArrayAttr(subgroupTileSizes));
  attrs.emplace_back(StringAttr::get(context, "mma_kind"), mmaKind);
  auto configDict = DictionaryAttr::get(context, attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  // TODO(qedawkins): Use a shared pipeline identifier here.
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, loweringConfig,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTileAndFuse,
      workgroupSize, targetSubgroupSize);
}

} // namespace mlir::iree_compiler::IREE::GPU
