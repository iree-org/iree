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
  auto properties = resourceLimits.cooperative_matrix_properties_nv()
                        .getAsRange<spirv::CooperativeMatrixPropertiesNVAttr>();
  for (auto property : properties) {
    if (property.a_type().getValue() == lhsType &&
        property.b_type().getValue() == rhsType &&
        property.c_type().getValue() == resultType &&
        property.result_type().getValue() == resultType &&
        property.scope().getValue() == spirv::Scope::Subgroup) {
      int64_t matmulM = property.m_size().getValue().getZExtValue();
      int64_t matmulN = property.n_size().getValue().getZExtValue();
      int64_t matmulK = property.k_size().getValue().getZExtValue();
      if (m % matmulM == 0 && n % matmulN == 0 && k % matmulK == 0)
        return CooperativeMatrixSize{matmulM, matmulN, matmulK};
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

  ArrayRef<int64_t> lhsShape = getUntiledShape(lhs);
  ArrayRef<int64_t> rhsShape = getUntiledShape(rhs);
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
  int64_t subgroupSize = resourceLimits.subgroup_size().getInt();
  std::array<int64_t, 3> workgroupSize = {subgroupSize, 1, 1};

  TileSizesListType tileSizes;
  // Again because we only consider whether the input workload is perfectly
  // divisible by some native cooperative matrix size, not some multiples of it,
  // need to make sure the subgroup tile sizes are the same as the workgroup
  // one.
  tileSizes.push_back({coopMatSize->m, coopMatSize->n, coopMatSize->k});
  tileSizes.push_back({coopMatSize->m, coopMatSize->n, coopMatSize->k});

  return setOpConfigAndEntryPointFnTranslation(op->getParentOfType<FuncOp>(),
                                               op, tileSizes, {}, pipeline,
                                               workgroupSize);
}

LogicalResult setNVIDIACodeGenConfig(const spirv::TargetEnv &targetEnv,
                                     Operation *rootOp) {
  if (auto matmulOp = dyn_cast<linalg::MatmulOp>(rootOp)) {
    return setOpConfig(targetEnv, matmulOp);
  }
  return success();
}

}  // namespace detail
}  // namespace iree_compiler
}  // namespace mlir
