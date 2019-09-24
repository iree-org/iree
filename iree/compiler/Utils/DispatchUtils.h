// Copyright 2019 Google LLC
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

// Utilities for dispatch region and function manipulation.
// These are shared between all dispatchable types such as the standard
// iree.dispatch_region as well as dispatch-related types like
// iree.reduction_region.

#ifndef IREE_COMPILER_UTILS_DISPATCHUTILS_H_
#define IREE_COMPILER_UTILS_DISPATCHUTILS_H_

#include <utility>

#include "iree/compiler/IR/Ops.h"
#include "iree/compiler/IR/StructureOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace iree_compiler {

// Calculates the workload for |op| based on the op type.
Value *calculateWorkload(Operation *op, Value *baseOperand);

// Returns true if the func is trivially dispatchable, meaning that:
// - it contains a single block
// - it contains a single dispatch region
// - it contains a return op directly returning the dispatch region results
bool isTriviallyDispatchable(FuncOp func);

// Builds a new iree.dispatch_region with the given |ops|.
// The region will capture all required values and return all values used
// outside of the |ops| provided. The region will be inserted at the location of
// the last operation in the set.
//
// All |ops| must be compatible with the |workload| specified as they will all
// be dispatched with the same workgroup structure.
// TODO(benvanik): ensure we want to insert at end. Maybe front?
LogicalResult buildDispatchRegion(FuncOp func, Block *parentBlock,
                                  Value *workload, ArrayRef<Operation *> ops);

// Merges multiple dispatch regions within a block into the same region,
// if possible. Operations may be reordered if it's possible to merge more while
// still obeying data dependencies.
LogicalResult mergeBlockDispatchRegions(FuncOp func, Block *parentBlock);

// Inlines use of the given |value| from outside of a dispatch region to inside
// of it and removes the argument. Supports multiple arguments that reference
// |value| and will clone the entire value tree.
LogicalResult inlineDispatchRegionOperandsUsingValue(
    IREE::DispatchRegionOp dispatchRegionOp, Value *value);

// Creates an iree.multi_arch_executable containing an iree.executable with an
// exported function containing the body region of |op|. Created executables
// will be named for their original function concatenated with |symbolSuffix|.
std::pair<IREE::MultiArchExecutableOp, FuncOp> createRegionExecutable(
    Operation *op, FunctionType functionType, StringRef symbolSuffix);

// Inserts a conversion of an arbitrary |value| to a memref, possibly by way of
// wrapping in an allocation.
// Returns a new memref containing the value or an alias to |value|.
Value *insertDispatcherStore(Operation *op, Value *value, OpBuilder &builder);

// Inserts a load from a wrapped memref.
// Returns the value in the original type or an alias to the |value| memref.
Value *insertDispatcherLoad(Operation *op, Value *originalValue,
                            Value *allocatedValue, OpBuilder &builder);

// TODO(benvanik): enough information to walk into dispatch region and compute
// shape when not static.
Value *allocateDispatchOutputBuffer(Location loc, MemRefType type,
                                    OpBuilder &builder);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_DISPATCHUTILS_H_
