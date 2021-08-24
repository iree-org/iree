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

#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

namespace mlir {
namespace iree_compiler {
namespace detail {

namespace {

/// Returns the cooperative matrix (M, N, K) sizes that are supported by the
/// target environment and match the given parameters.
Optional<SmallVector<int64_t, 4>> getCooperativeMatrixSize(
    spirv::ResourceLimitsAttr resourceLimits, Type lhsType, Type rhsType,
    Type initType, Type resultType) {
  for (auto coopMatmulProperties :
       resourceLimits.cooperative_matrix_properties_nv()
           .getAsRange<spirv::CooperativeMatrixPropertiesNVAttr>()) {
    if (coopMatmulProperties.a_type().getValue() == lhsType &&
        coopMatmulProperties.b_type().getValue() == rhsType &&
        coopMatmulProperties.c_type().getValue() == initType &&
        coopMatmulProperties.result_type().getValue() == resultType &&
        coopMatmulProperties.scope().getValue() == spirv::Scope::Subgroup) {
      return SmallVector<int64_t, 4>{
          coopMatmulProperties.m_size().getValue().getSExtValue(),
          coopMatmulProperties.n_size().getValue().getSExtValue(),
          coopMatmulProperties.k_size().getValue().getSExtValue()};
    }
  }
  return llvm::None;
}

Optional<SPIRVCodeGenConfig> getOpConfig(const spirv::TargetEnv &targetEnv,
                                         linalg::MatmulOp op) {
  if (!targetEnv.allows(spirv::Capability::CooperativeMatrixNV) ||
      !targetEnv.allows(spirv::Extension::SPV_NV_cooperative_matrix)) {
    return llvm::None;
  }

  ArrayRef<int64_t> lhsShape = getUntiledShape(op.inputs()[0]);
  ArrayRef<int64_t> rhsShape = getUntiledShape(op.inputs()[1]);

  if (llvm::any_of(lhsShape, ShapedType::isDynamic) ||
      llvm::any_of(rhsShape, ShapedType::isDynamic)) {
    return llvm::None;
  }

  auto resourceLimits = targetEnv.getResourceLimits();
  auto getElementType = [](Value v) {
    return v.getType().cast<ShapedType>().getElementType();
  };

  auto outputElementType = getElementType(op.outputs()[0]);

  Optional<SmallVector<int64_t, 4>> coopMatSize = getCooperativeMatrixSize(
      resourceLimits, getElementType(op.inputs()[0]),
      getElementType(op.inputs()[1]), outputElementType, outputElementType);
  if (!coopMatSize) return llvm::None;

  // Check that the matmul sizes are a multiple of the tilesize.
  auto isMultipleOf = [](int64_t s, int64_t ts) {
    return !ShapedType::isDynamic(s) && (s % ts) == 0;
  };

  if (!isMultipleOf(lhsShape[0], (*coopMatSize)[0]) ||
      !isMultipleOf(rhsShape[1], (*coopMatSize)[1]) ||
      !isMultipleOf(lhsShape[1], (*coopMatSize)[2]) ||
      !isMultipleOf(rhsShape[0], (*coopMatSize)[2])) {
    return llvm::None;
  }

  // For now this is being hard-wired to be {4, 4, 2}. This can actually be set
  // to whatever, but ultimately depends on register pressure.
  const int64_t numVecMatmulPerSubgroupX = 4;
  const int64_t numVecMatmulPerSubgroupY = 4;
  const int64_t numVecMatmulPerSubgroupK = 2;

  SPIRVCodeGenConfig config = {};
  config.pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize;

  config.workgroupTileSizes = {numVecMatmulPerSubgroupY * (*coopMatSize)[0],
                               numVecMatmulPerSubgroupX * (*coopMatSize)[1],
                               numVecMatmulPerSubgroupK * (*coopMatSize)[2]};

  config.subgroupTileSizes = {numVecMatmulPerSubgroupY * (*coopMatSize)[0],
                              numVecMatmulPerSubgroupX * (*coopMatSize)[1]};

  int64_t subgroupSize =
      resourceLimits.subgroup_size().getValue().getSExtValue();
  config.workgroupSize = {subgroupSize, 1, 1};

  return config;
}

}  // namespace

Optional<SPIRVCodeGenConfig> getNVIDIACodeGenConfig(
    const spirv::TargetEnv &targetEnv, Operation *op) {
  if (auto matmulOp = dyn_cast<linalg::MatmulOp>(op)) {
    return getOpConfig(targetEnv, matmulOp);
  }
  return llvm::None;
}

}  // namespace detail
}  // namespace iree_compiler
}  // namespace mlir
