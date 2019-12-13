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

#ifndef IREE_COMPILER_DIALECT_FLOW_UTILS_DISPATCHUTILS_H_
#define IREE_COMPILER_DIALECT_FLOW_UTILS_DISPATCHUTILS_H_

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

// Returns true if we know about this dialect and have special dispatchability
// information about it.
bool isOpOfKnownDialect(Operation *op);

// Builds a new dispatch region with the given |ops|.
// The region will capture all required values and return all values used
// outside of the |ops| provided. The region will be inserted at the location of
// the last operation in the set.
//
// All |ops| must be compatible with the |workload| specified as they will all
// be dispatched with the same workgroup structure.
// TODO(benvanik): ensure we want to insert at end. Maybe front?
LogicalResult buildDispatchRegion(FuncOp func, Block *parentBlock,
                                  Value *workload, ArrayRef<Operation *> ops);

// Creates an executable containing exported function containing the body region
// of |op|. Created executables will be named for their original function
// concatenated with |symbolSuffix|. All functions reachable by the region will
// be added to the executable by looking them up in |dispatchableFuncOps|.
std::pair<IREE::Flow::ExecutableOp, FuncOp> createRegionExecutable(
    Operation *op, FunctionType functionType, StringRef symbolSuffix,
    llvm::StringMap<FuncOp> &dispatchableFuncOps);

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_FLOW_UTILS_DISPATCHUTILS_H_
