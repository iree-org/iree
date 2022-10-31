// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- AMDConfig.h - AMD CodeGen Configurations ---------------------------===//
//
// This file contains CodeGen configurations for AMD GPUs.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"

#define DEBUG_TYPE "iree-spirv-amd-config"

using llvm::APIntOps::GreatestCommonDivisor;

// The default number of subgroups to use per workgroup.
constexpr unsigned numSubgroupsPerWorkgroup = 4;
// The default number of tiles along K dimension to use per workgroup.
constexpr unsigned numKTilesPerWorkgroup = 2;

namespace mlir {
namespace iree_compiler {
namespace detail {

constexpr unsigned AMDSoftwarePipelineDepth = 2;

/// Return the unique instance of OpType in `block` if it is indeed unique.
/// Return null if none or more than 1 instances exist.
static bool containsOnlySupportedOps(Block &block) {
  bool hasOnlySupportedOps = true;
  block.walk([&](Operation *op) {
    if (!isa<arith::AddFOp, arith::AddIOp, arith::SubIOp, arith::SubFOp,
             arith::DivFOp, arith::DivUIOp, arith::DivSIOp, arith::NegFOp>(
            op)) {
      hasOnlySupportedOps = false;
      return WalkResult::interrupt();
    } else {
    }
    return WalkResult::advance();
  });
  return hasOnlySupportedOps;
}

// Check if the given function contains an op that may require a broadcast of
// the reduced result.
static bool isFusedWithLegalCoopMatOps(linalg::LinalgOp reduce) {
  func::FuncOp entryPoint = reduce->getParentOfType<func::FuncOp>();
  bool fusedWithLegalOps = true;
  entryPoint.walk([&](linalg::LinalgOp linalgOp) {
    if (linalgOp == reduce || isMatmulOrBatchMatmul(linalgOp)) {
      return WalkResult::advance();
    }
    if (!containsOnlySupportedOps(linalgOp->getRegion(0).front())) {
      fusedWithLegalOps = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return fusedWithLegalOps;
}

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
  // This configuration is only for cooperative matrix.
  if (!targetEnv.allows(spirv::Capability::CooperativeMatrixNV) ||
      !targetEnv.allows(spirv::Extension::SPV_NV_cooperative_matrix)) {
    return success();
  }

  // Uncomment to skip ops.
  // if (!isa<linalg::BatchMatmulOp, linalg::MatmulOp>(*op)) return success();
  // if (!isFusedWithLegalCoopMatOps(op)) return success();

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

static LogicalResult setAMDMatmulConfig(linalg::LinalgOp op,
                                        const spirv::TargetEnv &targetEnv) {
  if (failed(setCooperativeMatrixConfig(targetEnv, op))) return failure();
  if (getLoweringConfig(op)) return success();

  spirv::ResourceLimitsAttr limits = targetEnv.getResourceLimits();
  const int subgroupSize = limits.getSubgroupSize();
  const std::array<int64_t, 2> workgroupXY = {subgroupSize / 2, 8};
  std::array<int64_t, 3> threadMNK;
  auto inputType = op.getDpsInputOperand(0)->get().getType().cast<ShapedType>();
  if (inputType.getElementType().getIntOrFloatBitWidth() == 16) {
    threadMNK = {8, 8, 32};
  } else {
    threadMNK = {8, 4, 16};
  }
  return setMatmulOpConfig(limits, op, workgroupXY, threadMNK,
                           /*enablePromotion=*/true, AMDSoftwarePipelineDepth);
}

// RDNA architecture:
// https://gpuopen.com/wp-content/uploads/2019/08/RDNA_Architecture_public.pdf
//
// Workgroup Processor (WGP) is the block for workgroups in RDNA; it has its own
// instruction/constant cache, L0 cache x2, Local Data Share (LDS, a.k.a. shared
// memory), SALU x4, SIMD32 x4.
//
// * 1024 registers per SIMD32
// * 128KB LDS per WGP
// * Max 20 waves per SIMD32
// * Max 64KB LDS per workgroup

LogicalResult setAMDCodeGenConfig(const spirv::TargetEnv &targetEnv,
                                  Operation *rootOp) {
  spirv::ResourceLimitsAttr limits = targetEnv.getResourceLimits();
  int subgroupSize = limits.getSubgroupSize();

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp)) {
    if (isMatmulOrBatchMatmul(linalgOp))
      return setAMDMatmulConfig(linalgOp, targetEnv);
  }

  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::BatchMatmulOp, linalg::MatmulOp>(
          [targetEnv](auto op) { return setAMDMatmulConfig(op, targetEnv); })
      .Case<linalg::Conv2DNchwFchwOp, linalg::Conv2DNhwcHwcfOp>(
          [subgroupSize](auto op) {
            bool hasPaddedInput =
                op.image().template getDefiningOp<tensor::PadOp>();
            int bestTilingFactor = hasPaddedInput ? 16 : 32;
            return setConvOpConfig(op, subgroupSize, bestTilingFactor);
          })
      .Case<linalg::DepthwiseConv2DNhwcHwcOp>([subgroupSize](auto op) {
        bool hasPaddedInput =
            op.image().template getDefiningOp<tensor::PadOp>();
        int bestTilingFactor = hasPaddedInput ? 16 : 32;
        return setConvOpConfig(op, subgroupSize, bestTilingFactor);
      })
      .Default([](Operation *) { return success(); });
}

}  // namespace detail
}  // namespace iree_compiler
}  // namespace mlir
