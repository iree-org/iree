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
// dispatch region as well as dispatch-related types like reduction region.

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
LogicalResult buildDispatchRegion(Block *parentBlock, Value workload,
                                  ArrayRef<Operation *> ops);

// Creates a flow.executable out of a set of functions, pulling in all other
// functions reachable by the provided functions.
ExecutableOp createExecutable(Location loc, StringRef executableName,
                              ArrayRef<FuncOp> funcOps, ModuleOp parentModuleOp,
                              llvm::StringMap<FuncOp> &dispatchableFuncOps);

// Converts a region body to a function.
// The region entry block args and return terminators are used to derive the
// function type.
FuncOp createRegionFunction(Location loc, StringRef functionName,
                            Region &region);

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_FLOW_UTILS_DISPATCHUTILS_H_
