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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"

#define DEBUG_TYPE "iree-spirv-nvidia-config"

namespace mlir {
namespace iree_compiler {
namespace detail {

struct CooperativeMatrixSize {
  int64_t m;
  int64_t n;
  int64_t k;
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
      int matmulM = property.getMSize();
      int matmulN = property.getNSize();
      int matmulK = property.getKSize();
      if (m % matmulM == 0 && n % matmulN == 0 && k % matmulK == 0) {
        return CooperativeMatrixSize{matmulM, matmulN, matmulK};
      }
    }
  }
  return llvm::None;
}

static LogicalResult setOpConfig(const spirv::TargetEnv &targetEnv,
                                 linalg::MatmulOp op) {
  // This configuration is only for cooperative matrix.
  if (!targetEnv.allows(spirv::Capability::CooperativeMatrixNV) ||
      !targetEnv.allows(spirv::Extension::SPV_NV_cooperative_matrix)) {
    return success();
  }

  Value lhs = op.inputs()[0], rhs = op.inputs()[1], init = op.outputs()[0];

  ArrayRef<int64_t> lhsShape = lhs.getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> rhsShape = rhs.getType().cast<ShapedType>().getShape();
  if (llvm::any_of(lhsShape, ShapedType::isDynamic)) return success();
  if (llvm::any_of(rhsShape, ShapedType::isDynamic)) return success();

  // TODO: Cooperative matrix support is fairly restricted. We can only have
  // a curated list of fused element wise ops as defined in the extension
  // SPV_NV_cooperative_matrix. Check that once we move bufferization after
  // vectorization.

  auto getElementType = [](Value v) {
    return v.getType().cast<ShapedType>().getElementType();
  };

  auto resourceLimits = targetEnv.getResourceLimits();
  auto coopMatSize = getCooperativeMatrixSize(
      resourceLimits, getElementType(lhs), getElementType(rhs),
      getElementType(init), lhsShape[0], rhsShape[1], lhsShape[1]);
  if (!coopMatSize) return success();

  auto pipeline = IREE::Codegen::DispatchLoweringPassPipeline::
      SPIRVVectorizeToCooperativeOps;

  // For now only support one subgroup per workgroup because in the above
  // configuration deduction step we only consider whether the input workload is
  // perfectly divisible by some native cooperative matrix size.
  //
  // TODO: Use some heuristics to deduce how many subgroups should be used and
  // the tile sizes for each subgroup, considering the input workload size and
  // native cooperative matrix size choices.
  int subgroupSize = resourceLimits.getSubgroupSize();
  std::array<int64_t, 3> workgroupSize = {subgroupSize, 1, 1};

  TileSizesListType tileSizes;
  // Again because we only consider whether the input workload is perfectly
  // divisible by some native cooperative matrix size, not some multiples of it,
  // need to make sure the subgroup tile sizes are the same as the workgroup
  // one.
  tileSizes.push_back({coopMatSize->m, coopMatSize->n, coopMatSize->k});
  tileSizes.push_back({coopMatSize->m, coopMatSize->n, coopMatSize->k});

  return setOpConfigAndEntryPointFnTranslation(
      op->getParentOfType<func::FuncOp>(), op, tileSizes, pipeline,
      workgroupSize);
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
  int subgroupSize = targetEnv.getResourceLimits().getSubgroupSize();

  // First try to see if we can use tensor cores.
  if (auto matmulOp = dyn_cast<linalg::MatmulOp>(rootOp)) {
    if (failed(setOpConfig(targetEnv, matmulOp))) return failure();
    if (getLoweringConfig(rootOp)) return success();
  }

  if (isa<linalg::BatchMatmulOp, linalg::MatmulOp>(rootOp)) {
    std::array<int64_t, 2> workgroupXY = {subgroupSize, 8};
    std::array<int64_t, 3> threadMNK = {4, 4, 32};
    return setMatmulOpConfig(rootOp, subgroupSize, workgroupXY, threadMNK,
                             /*useWorkgroupMemory=*/true);
  }

  return success();
}

}  // namespace detail
}  // namespace iree_compiler
}  // namespace mlir
