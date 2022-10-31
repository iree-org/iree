// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- NVIDIAConfig.h - NVIDIA CodeGen Configurations ---------------------===//
//
// This file contains CodeGen configurations for NVIDIA GPUs.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#define DEBUG_TYPE "iree-spirv-nvidia-config"

using llvm::APIntOps::GreatestCommonDivisor;

// The default number of subgroups to use per workgroup.
constexpr unsigned numSubgroupsPerWorkgroup = 4;
// The default number of tiles along K dimension to use per workgroup.
constexpr unsigned numKTilesPerWorkgroup = 2;

namespace mlir {
namespace iree_compiler {
namespace detail {

struct CooperativeMatrixSize {
  int64_t m;       // Native cooperative matrix size along M dimension
  int64_t n;       // Native cooperative matrix size along N dimension
  int64_t k;       // Native cooperative matrix size along K dimension
  int64_t mCount;  // # subgroups along M dimension
  int64_t nCount;  // # subgroups along N dimension
  int64_t kCount;  // # tiles along K dimension
};

/// Returns the cooperative matrix (M, N, K) sizes that are supported by the
/// target environment and match the given parameters.
static Optional<CooperativeMatrixSize> getCooperativeMatrixSize(
    spirv::ResourceLimitsAttr resourceLimits, Type lhsType, Type rhsType,
    Type resultType, int64_t m, int64_t n, int64_t k) {
  auto properties = resourceLimits.getCooperativeMatrixPropertiesNv()
                        .getAsRange<spirv::CooperativeMatrixPropertiesNVAttr>();
  for (auto property : properties) {
    if (property.getAType() == lhsType && property.getBType() == rhsType &&
        property.getCType() == resultType &&
        property.getResultType() == resultType &&
        property.getScope().getValue() == spirv::Scope::Subgroup) {
      const unsigned matmulM = property.getMSize();
      const unsigned matmulN = property.getNSize();
      const unsigned matmulK = property.getKSize();
      if (m % matmulM == 0 && n % matmulN == 0 && k % matmulK == 0) {
        const uint64_t nTileCount = n / matmulN;
        const uint64_t mTileCount = m / matmulM;
        const APInt nTileCountAPInt(/*numBits=*/64, nTileCount);
        const APInt mTileCountAPInt(/*numBits=*/64, mTileCount);

        int64_t nCount = 0, mCount = 0;
        uint64_t subgroupCount = numSubgroupsPerWorkgroup;
        uint64_t squareRoot = 1u << (llvm::Log2_64(subgroupCount) / 2);

        // See if the square root of subgroupCount can divide mTileCount. If so
        // it means we can distribute to both dimensions evenly. Otherwise, try
        // to distribute to N and then M.
        if (mTileCount > squareRoot && mTileCount % squareRoot == 0) {
          mCount = squareRoot;
          APInt nGCD = GreatestCommonDivisor(
              nTileCountAPInt, APInt(64, subgroupCount / squareRoot));
          nCount = nGCD.getSExtValue();
        } else {
          APInt nGCD =
              GreatestCommonDivisor(nTileCountAPInt, APInt(64, subgroupCount));
          nCount = nGCD.getSExtValue();

          subgroupCount /= nCount;
          APInt mGCD =
              GreatestCommonDivisor(mTileCountAPInt, APInt(64, subgroupCount));
          mCount = mGCD.getSExtValue();
        }

        int64_t kCount = std::min<int64_t>(k / matmulK, numKTilesPerWorkgroup);

        return CooperativeMatrixSize{matmulM, matmulN, matmulK,
                                     mCount,  nCount,  kCount};
      }
    }
  }
  return llvm::None;
}

static LogicalResult setCooperativeMatrixConfig(
    const spirv::TargetEnv &targetEnv, linalg::LinalgOp op) {
  LLVM_DEBUG(llvm::dbgs() << "trying to matmul tensorcore config...\n");
  // This configuration is only for cooperative matrix.
  if (!targetEnv.allows(spirv::Capability::CooperativeMatrixNV) ||
      !targetEnv.allows(spirv::Extension::SPV_NV_cooperative_matrix)) {
    return success();
  }

  if (op.hasDynamicShape()) return success();

  Value lhs = op.getDpsInputOperand(0)->get();
  Value rhs = op.getDpsInputOperand(1)->get();
  Value init = op.getDpsInitOperand(0)->get();

  int lastParallelDim = -1;
  const auto [bIndex, mIndex, nIndex, kIndex] =
      getMatmulBMNKIndex(op, &lastParallelDim);
  if (mIndex < 0 || nIndex < 0 || kIndex < 0) return success();
  const bool isBM = bIndex >= 0;

  SmallVector<int64_t, 4> loopRanges = op.getStaticLoopRanges();

  const int64_t dimM = loopRanges[mIndex];
  const int64_t dimK = loopRanges[kIndex];
  const int64_t dimN = loopRanges[nIndex];
  LLVM_DEBUG({
    llvm::dbgs() << "input matmul shape (B, M, N, K) = ("
                 << (bIndex >= 0 ? loopRanges[bIndex] : -1) << ", " << dimM
                 << ", " << dimN << ", " << dimK << ")\n";
  });

  // TODO: Cooperative matrix support is fairly restricted. We can only have
  // a curated list of fused element wise ops as defined in the extension
  // SPV_NV_cooperative_matrix. Check that once we move bufferization after
  // vectorization.

  auto getElementType = [](Value v) {
    return v.getType().cast<ShapedType>().getElementType();
  };

  spirv::ResourceLimitsAttr resourceLimits = targetEnv.getResourceLimits();
  Optional<CooperativeMatrixSize> coopMatSize = getCooperativeMatrixSize(
      resourceLimits, getElementType(lhs), getElementType(rhs),
      getElementType(init), dimM, dimN, dimK);
  if (!coopMatSize) return success();

  auto pipeline = IREE::Codegen::DispatchLoweringPassPipeline::
      SPIRVCooperativeMatrixVectorize;

  std::array<int64_t, 3> workgroupSize{
      coopMatSize->nCount * resourceLimits.getSubgroupSize(),
      coopMatSize->mCount, 1};

  SmallVector<int64_t> workgroupTileSizes(kIndex + 1, 0);
  if (isBM) workgroupTileSizes[bIndex] = 1;
  workgroupTileSizes[mIndex] = coopMatSize->mCount * coopMatSize->m;
  workgroupTileSizes[nIndex] = coopMatSize->nCount * coopMatSize->n;
  workgroupTileSizes[kIndex] = coopMatSize->kCount * coopMatSize->k;

  SmallVector<int64_t> subgroupTileSizes(kIndex + 1, 0);
  if (isBM) subgroupTileSizes[bIndex] = 1;
  subgroupTileSizes[mIndex] = coopMatSize->m;
  subgroupTileSizes[nIndex] = coopMatSize->n;
  subgroupTileSizes[kIndex] = coopMatSize->k;

  // Also create one level for reduction. This is needed because of
  // SPIRVTileAndPromotePass requires it.
  // TODO(#10499): Consolidate tiling configuration across different pipelines.
  SmallVector<int64_t> reductionTileSizes;
  reductionTileSizes.append(lastParallelDim + 1, 0);
  reductionTileSizes.push_back(coopMatSize->kCount * coopMatSize->k);

  TileSizesListType tileSizes;
  tileSizes.reserve(3);
  tileSizes.push_back(workgroupTileSizes);
  tileSizes.push_back(subgroupTileSizes);
  tileSizes.push_back(reductionTileSizes);

  return setOpConfigAndEntryPointFnTranslation(
      op->getParentOfType<func::FuncOp>(), op, tileSizes, pipeline,
      workgroupSize);
}

static LogicalResult setNVIDIAMatmulConfig(linalg::LinalgOp op,
                                           const spirv::TargetEnv &targetEnv) {
  // First try to see if we can use tensor cores.
  spirv::ResourceLimitsAttr limits = targetEnv.getResourceLimits();
  if (failed(setCooperativeMatrixConfig(targetEnv, op))) return failure();
  if (getLoweringConfig(op)) return success();

  const int subgroupSize = limits.getSubgroupSize();
  const std::array<int64_t, 2> workgroupXY = {subgroupSize, 8};
  std::array<int64_t, 3> threadMNK;
  auto inputType = op.getDpsInputOperand(0)->get().getType().cast<ShapedType>();
  if (inputType.getElementType().getIntOrFloatBitWidth() == 16) {
    threadMNK = {8, 8, 32};
  } else {
    threadMNK = {4, 4, 32};
  }
  return setMatmulOpConfig(limits, op, workgroupXY, threadMNK,
                           /*enablePromotion=*/true);
}

// Volta architecture:
// https://docs.nvidia.com/cuda/volta-tuning-guide/index.html#sm-occupancy
//
// * 64K 32-bit registers per SM
// * 96KB shared memory per SM
// * Max 32 thread blocks per SM
// * Max 64 concurrent warps per SM
// * Max 255 registers per thread

// Turing architecture:
// https://docs.nvidia.com/cuda/turing-tuning-guide/index.html#sm-occupancy
//
// * 64K 32-bit registers per SM
// * 64KB shared memory per SM
// * Max 16 thread blocks per SM
// * Max 32 concurrent warps per SM
// * Max 255 registers per thread

// Ampere architecture:
// https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#sm-occupancy
//
// * 64K 32-bit registers per SM
// * 164KB/96KB shared memory for compute capability 8.0/8.6
// * Max 32/16 thread blocks per SM for compute capability 8.0/8.6
// * Max 64 concurrent warps per SM
// * Max 255 registers per thread

// Note that the above numbers are from CUDA docs; for Vulkan the drivder can
// expose slightly different numbers, e.g., max shared memory size is smaller.

LogicalResult setNVIDIACodeGenConfig(const spirv::TargetEnv &targetEnv,
                                     Operation *rootOp) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp)) {
    if (isMatmulOrBatchMatmul(linalgOp))
      return setNVIDIAMatmulConfig(linalgOp, targetEnv);
  }

  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::BatchMatmulOp, linalg::MatmulOp>(
          [&](auto op) { return setNVIDIAMatmulConfig(op, targetEnv); })
      .Default([](Operation *) { return success(); });
}

}  // namespace detail
}  // namespace iree_compiler
}  // namespace mlir
