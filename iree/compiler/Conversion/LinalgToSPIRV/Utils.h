// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Utils.h - Utility functions used in Linalg to SPIR-V lowering ------===//
//
// Utility functions used while lowering from Linalg to SPIRV.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_CONVERSION_LINALGTOSPIRV_UTILS_H_
#define IREE_COMPILER_CONVERSION_LINALGTOSPIRV_UTILS_H_

#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/FoldUtils.h"

namespace mlir {
namespace iree_compiler {

static constexpr int kNumGPUDims = 3;

/// Allocation callback for allocation workgroup local memory.
Optional<Value> allocateWorkgroupMemory(OpBuilder &b, memref::SubViewOp subview,
                                        ArrayRef<Value> boundingSubViewSize,
                                        DataLayout &layout);

/// Function used as callback for copyin/copyout in promotion pattern used
/// to promote subviews to workgroup memory when the number of threads is
/// known to be greater than equal to the number of iteration of loops the
/// copy is lowered to.
LogicalResult copyToWorkgroupMemory(OpBuilder &b, Value src, Value dst);

/// Deallocation callback for allocation workgroup local memory.
LogicalResult deallocateWorkgroupMemory(OpBuilder &b, Value buffer);

/// Generate the operations that compute the processor ID and number of
/// processors. Used as the callback needed for LinalgDistributionOptions.
class GPUGlobalId;
class GPUGlobalCount;
template <typename GPUIdOp, typename GPUCountOp>
SmallVector<linalg::ProcInfo, 2> getGPUProcessorIdsAndCounts(OpBuilder &builder,
                                                             Location loc,
                                                             unsigned numDims);
/// Updates the workgroup size used for the dispatch region.
LogicalResult updateWorkGroupSize(FuncOp funcOp,
                                  ArrayRef<int64_t> workGroupSize);

}  // namespace iree_compiler
}  // namespace mlir

#endif  //  IREE_COMPILER_CONVERSION_LINALGTOSPIRV_UTILS_H_
