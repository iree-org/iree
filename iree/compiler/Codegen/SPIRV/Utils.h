// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Utils.h - Utility functions for lowering Linalg to SPIR-V ----------===//
//
// Utility functions used while lowering from Linalg to SPIR-V.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_SPIRV_UTILS_H_
#define IREE_COMPILER_CODEGEN_SPIRV_UTILS_H_

#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/FoldUtils.h"

namespace mlir {
namespace iree_compiler {

static constexpr int kNumGPUDims = 3;

//===----------------------------------------------------------------------===//
// Attribute utils
//===----------------------------------------------------------------------===//

/// Given an operation, return the `spv.target_env` attribute.
spirv::TargetEnvAttr getSPIRVTargetEnvAttr(Operation *op);

/// Returns the attribute name carrying information about distribution.
const char *getSPIRVDistributeAttrName();

//===----------------------------------------------------------------------===//
// Workgroup memory utils
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Processor ID/size utils
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Loop utils
//===----------------------------------------------------------------------===//

/// Collapses all loops in a scf.parallel into one scf.parallel operation. This
/// is done by
/// 1) Normalize the loop bounds to be [0, (ub - lb) / step)
/// 2) Compute the total number of iterations.
/// 3) From the induction variable of the modified loop, compute the values of
///    the original induction variables by de-linearization.
scf::ParallelOp collapseParallelLoops(PatternRewriter &rewriter,
                                      scf::ParallelOp pLoopOp);

struct LoopBounds {
  Value lb;
  Value ub;
  Value step;
};

/// Replaces a scf.parallelOp with an optional scf.parallel op and nested
/// scf.for operations. To create the scf.parallel op as the outermost loop,
/// pass the lower bound, upper bound and steps in `newPLoopLbs`, `newPLoopUbs`,
/// and `newPLoopStep` respectively. The bounds of the inner scf.for operations
/// to be created are passed in `forLbs`, `forUbs`, and `forStep`. The
/// `permutation` vector contains a mapping from the original loop order, to the
/// loop order to be generated.
Operation *replacePLoopOp(ConversionPatternRewriter &rewriter,
                          scf::ParallelOp pLoopOp,
                          ArrayRef<LoopBounds> newPLoopBounds,
                          ArrayRef<LoopBounds> forBounds,
                          ArrayRef<unsigned> permutation);

/// Distributes scf.parallel to processors with the processors logically
/// arranged with same dimensionality as the number of loops, i.e. a
/// scf.parallel with 2 loops to a 2D grid of processors. `processorIDs` must be
/// of same size as the number of loops and are the values to use for process ID
/// and number of processors along each dimension in the distributed code.  This
/// method assumes that the number of processors is greater than or equal to the
/// number of iterations. So just generates an if statement to mask of
/// processors with no work. When the number of processors is known to be
/// exactly equal to the number of iterations, the if statement is not needed as
/// well. In such cases, `generateGuard` can be set to `false` to avoid
/// generating the if statement.
LogicalResult distributeSingleIterationPerProcessor(
    ConversionPatternRewriter &rewriter, scf::ParallelOp pLoopOp,
    ArrayRef<linalg::ProcInfo> procInfo, bool generateGuard = false);

}  // namespace iree_compiler
}  // namespace mlir

#endif  //  IREE_COMPILER_CODEGEN_SPIRV_UTILS_H_
