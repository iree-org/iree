// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/LLVMGPUUtils.h"

#include "mlir/Dialect/GPU/Passes.h"

namespace mlir {
namespace iree_compiler {

llvm::SmallVector<mlir::linalg::ProcInfo, 2> getGPUThreadIdsAndCounts(
    mlir::OpBuilder &builder, mlir::Location loc, unsigned numDims,
    llvm::ArrayRef<int64_t> workgroupSize) {
  assert(numDims <= kNumGPUDims);
  llvm::SmallVector<mlir::linalg::ProcInfo, 2> procInfo(numDims);
  std::array<gpu::Dimension, kNumGPUDims> dimAttr{
      gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
  mlir::Type indexType = builder.getIndexType();
  for (unsigned i = 0; i < numDims; ++i) {
    procInfo[numDims - 1 - i] = {
        builder.create<mlir::gpu::ThreadIdOp>(loc, indexType, dimAttr[i]),
        builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(workgroupSize[i]))};
  }
  return procInfo;
}

llvm::SmallVector<mlir::linalg::ProcInfo, 2> getSubgroupIdsAndCounts(
    mlir::OpBuilder &builder, mlir::Location loc, unsigned numDims,
    llvm::ArrayRef<int64_t> numSubgroups) {
  assert(numDims <= kNumGPUDims);
  llvm::SmallVector<mlir::linalg::ProcInfo, 2> procInfo(numDims);
  std::array<gpu::Dimension, kNumGPUDims> dimAttr{
      gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
  mlir::Type indexType = builder.getIndexType();
  for (unsigned i = 0; i < numDims; ++i) {
    mlir::Value subgroupId =
        builder.create<mlir::gpu::ThreadIdOp>(loc, indexType, dimAttr[i]);
    if (i == 0) {
      mlir::AffineExpr d0 = builder.getAffineDimExpr(0);
      subgroupId = mlir::makeComposedAffineApply(
          builder, loc, d0.floorDiv(builder.getAffineConstantExpr(kWarpSize)),
          {subgroupId});
    }
    procInfo[numDims - 1 - i] = {
        subgroupId, builder.create<mlir::arith::ConstantOp>(
                        loc, builder.getIndexAttr(numSubgroups[i]))};
  }
  return procInfo;
}

std::array<int64_t, 3> getWorkgroupSize(mlir::func::FuncOp funcOp) {
  std::array<int64_t, 3> workgroupSize;
  auto entryPointOp = mlir::iree_compiler::getEntryPoint(funcOp);
  llvm::Optional<mlir::ArrayAttr> workgroupSizeAttr =
      entryPointOp.workgroup_size();
  assert(workgroupSizeAttr.hasValue());
  for (auto it : llvm::enumerate(workgroupSizeAttr.getValue())) {
    workgroupSize[it.index()] =
        it.value().cast<mlir::IntegerAttr>().getValue().getZExtValue();
  }
  return workgroupSize;
}

}  // namespace iree_compiler
}  // namespace mlir
