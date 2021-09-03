// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_LLVMGPUUTILS_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_LLVMGPUUTILS_H_

#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/GPU/Passes.h"

static constexpr int32_t kNumGPUDims = 3;

static llvm::SmallVector<mlir::linalg::ProcInfo, 2> getGPUThreadIdsAndCounts(
    mlir::OpBuilder &builder, mlir::Location loc, unsigned numDims,
    llvm::ArrayRef<int64_t> workgroupSize) {
  assert(numDims <= kNumGPUDims);
  llvm::SmallVector<mlir::linalg::ProcInfo, 2> procInfo(numDims);
  std::array<llvm::StringRef, kNumGPUDims> dimAttr{"x", "y", "z"};
  mlir::Type indexType = builder.getIndexType();
  for (unsigned i = 0; i < numDims; ++i) {
    mlir::StringAttr attr = builder.getStringAttr(dimAttr[i]);
    procInfo[numDims - 1 - i] = {
        builder.create<mlir::gpu::ThreadIdOp>(loc, indexType, attr),
        builder.create<mlir::ConstantOp>(
            loc, builder.getIndexAttr(workgroupSize[i]))};
  }
  return procInfo;
}

static std::array<int64_t, 3> getWorkgroupSize(mlir::FuncOp funcOp) {
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

#endif  // IREE_COMPILER_CODEGEN_LLVMGPU_LLVMGPUUTILS_H_
