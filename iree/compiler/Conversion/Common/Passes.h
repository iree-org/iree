// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

/// Pass to rewrite Linalg on tensors destructive updates into updates through
/// memory.
std::unique_ptr<OperationPass<FuncOp>>
createLinalgRewriteDestructiveUpdatesPass();

/// Pass to optimize vector transfer_read and transfer_write.
std::unique_ptr<FunctionPass> createVectorTransferOptimizationPass();

// Pass to perform canonicalizations/cleanups related to HAL interface/buffer
// allocations and view operations.
std::unique_ptr<FunctionPass> createBufferAllocViewCleanUpPass();

}  // namespace iree_compiler
}  // namespace mlir
