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

/// Alias for callback function that allocates workgroup level memory.
using WorkgroupMemoryAllocationFn = std::function<Value(
    OpBuilder &builder, Location loc, ArrayRef<Value> dynamicSizes,
    MemRefType allocationType)>;

/// Adds passes to convert tiled+distributed linalg on tensors code to linalg on
/// buffers.
void addLinalgBufferizePasses(
    OpPassManager &passManager,
    WorkgroupMemoryAllocationFn allocationFn = nullptr);

/// Pass to initialize the function that computes the number of workgroups for
/// each entry point function. The function is defined, but is populated later.
std::unique_ptr<OperationPass<ModuleOp>> createDeclareNumWorkgroupsFnPass();

/// Pass to legalize function that returns number of workgroups to use for
/// launch to be runnable on the host.
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeNumWorkgroupsFnPass();

/// Pass to perform linalg on tensor bufferization.
std::unique_ptr<OperationPass<FuncOp>> createLinalgBufferizePass(
    WorkgroupMemoryAllocationFn allocationFn = nullptr);

/// Pass to rewrite Linalg on tensors destructive updates into updates through
/// memory.
std::unique_ptr<OperationPass<FuncOp>>
createLinalgRewriteDestructiveUpdatesPass();

/// Pass to remove operations that have allocate semantics but have no
/// uses. These arent removed by CSE.
std::unique_ptr<OperationPass<>> createRemoveDeadMemAllocsPass();

/// Pass to optimize vector transfer_read and transfer_write.
std::unique_ptr<FunctionPass> createVectorTransferOptimizationPass();

}  // namespace iree_compiler
}  // namespace mlir
