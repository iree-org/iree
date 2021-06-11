// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {

/// Alias for callback function that allocates workgroup level memory. The
/// callback expects the static shape and element-type of the result memref
/// type. Also expects values for the dynamic dimension of the allocated memref,
/// where each dynamic dimension corresponds to a `ShapedType::kDynamicSize`
/// value in `staticShape`.
using WorkgroupMemoryAllocationFn = std::function<Value(
    OpBuilder &builder, Location loc, ArrayRef<int64_t> staticShape,
    Type elementType, ArrayRef<Value> dynamicSizes)>;

/// Adds passes to convert tiled+distributed linalg on tensors code to linalg on
/// buffers.
void addLinalgBufferizePasses(
    OpPassManager &passManager,
    WorkgroupMemoryAllocationFn allocationFn = nullptr);

/// Pass to perform linalg on tensor bufferization. The function passed into the
/// pass through the `allocationFn` argument is invoked whenever a new buffer is
/// to be created. The callback will be passed the Values for the dynamic
/// dimensions in the memref type that is to be allocated.  The callback is
/// expected to return a MemRefType Value.  When no `allocationFn` is specified,
/// the default allocator generates an `std.alloc` instruction with the
/// allocated MemRefType having no stride map (i.e. default row-major striding)
/// and default memory space.
std::unique_ptr<OperationPass<FuncOp>> createLinalgBufferizePass(
    WorkgroupMemoryAllocationFn allocationFn = nullptr);

/// Pass to optimize vector transfer_read and transfer_write.
std::unique_ptr<FunctionPass> createVectorTransferOptimizationPass();

/// An ad-hoc pass to canonicalize selected loop carried dependencies on
/// scf.for.
std::unique_ptr<FunctionPass> createForOpCanonicalizationPass();

/// Pass to perform canonicalizations/cleanups related to HAL interface/buffer
/// allocations and view operations.
std::unique_ptr<FunctionPass> createBufferAllocViewCleanUpPass();

/// Flattens n-D MemRef subspan ops to 1-D MemRef and folds the byte offsets on
/// subspan ops to the consumer load/store ops, in preparation for lowering to
/// backends that require linearized access.
std::unique_ptr<OperationPass<ModuleOp>> createFlattenMemRefSubspanPass();

/// Sets the number of workgroups to use for each entry point in the dispatch
/// region.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createSetNumWorkgroupsPass(ArrayRef<int64_t> workgroupSize = {});

/// After running the upstream TensorConstantBufferize pass, remove tensor_loads
/// introduced for use only in tensor_extract. These can be folded to use a load
/// of the created memref object that holds the constant values.
std::unique_ptr<OperationPass<>> createFoldTensorExtractOpPass();

}  // namespace iree_compiler
}  // namespace mlir
