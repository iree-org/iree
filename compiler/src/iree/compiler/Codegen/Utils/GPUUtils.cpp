// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/GPUUtils.h"

#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"

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
            loc, builder.getIndexAttr(workgroupSize[i])),
        linalg::DistributionMethod::Cyclic};
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
        subgroupId,
        builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(numSubgroups[i])),
        linalg::DistributionMethod::Cyclic};
  }
  return procInfo;
}

std::array<int64_t, 3> getWorkgroupSize(mlir::func::FuncOp funcOp) {
  std::array<int64_t, 3> workgroupSize;
  FailureOr<IREE::HAL::ExecutableExportOp> exportOp =
      mlir::iree_compiler::getEntryPoint(funcOp);
  llvm::Optional<mlir::ArrayAttr> workgroupSizeAttr =
      exportOp->getWorkgroupSize();
  assert(workgroupSizeAttr.has_value());
  for (auto it : llvm::enumerate(workgroupSizeAttr.value())) {
    workgroupSize[it.index()] =
        it.value().cast<mlir::IntegerAttr>().getValue().getZExtValue();
  }
  return workgroupSize;
}

bool canPerformVectorAccessUsingAllThreads(ArrayRef<int64_t> shape,
                                           int64_t threadCount,
                                           int64_t vectorSize) {
  // Verify that each dimension of the shape can be distributed on the
  // threads
  int64_t threadsAvailable = threadCount;
  for (auto &dim : llvm::enumerate(llvm::reverse(shape))) {
    int64_t numElementPerThread = dim.index() == 0 ? vectorSize : 1;
    int64_t numThreads = dim.value() / numElementPerThread;
    if (numThreads == 0) return false;
    if (numThreads > threadsAvailable) {
      // If there are no enough remaining threads to distribute the current
      // dimension, try to use all remaining threads. But we still need to make
      // sure all work can be distributed to these threads evenly.
      if (numThreads % threadsAvailable != 0) return false;
      numThreads = threadsAvailable;
    }
    if (threadsAvailable % numThreads != 0) return false;
    threadsAvailable = threadsAvailable / numThreads;
    if (threadsAvailable == 1) break;
  }
  return threadsAvailable == 1;
}

Optional<Value> allocateWorkgroupMemory(OpBuilder &builder,
                                        memref::SubViewOp subview,
                                        ArrayRef<Value> sizeBounds,
                                        DataLayout &) {
  OpBuilder::InsertionGuard guard(builder);

  func::FuncOp funcOp = subview->getParentOfType<func::FuncOp>();
  if (!funcOp) return llvm::None;

  // The subview size bounds are expected to be constant; they specify the shape
  // of the allocation.
  SmallVector<int64_t, 2> shape;
  for (Value bound : sizeBounds) {
    APInt value;
    if (!matchPattern(bound, m_ConstantInt(&value))) return llvm::None;
    shape.push_back(value.getSExtValue());
  }

  builder.setInsertionPoint(&funcOp.front(), funcOp.front().begin());
  auto type = MemRefType::get(shape, subview.getType().getElementType(), {},
                              gpu::GPUDialect::getWorkgroupAddressSpace());
  Value buffer = builder.create<memref::AllocOp>(funcOp.getLoc(), type);
  return buffer;
}

LogicalResult deallocateWorkgroupMemory(OpBuilder &, Value /*buffer*/) {
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
