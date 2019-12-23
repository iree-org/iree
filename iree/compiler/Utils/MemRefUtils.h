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

#ifndef IREE_COMPILER_UTILS_MEMREFUTILS_H_
#define IREE_COMPILER_UTILS_MEMREFUTILS_H_

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
// Attempts to resolve the use of a value back to the MemRef it was loaded from.
// Returns either a MemRef view containing the value or nullptr if the value was
// not loaded from a MemRef (or is possibly unknown).
ValuePtr resolveValueToSourceMemRef(ValuePtr value, Operation *useOp);

// Wraps a memref with a MemRefToTensorOp, returning the resulting Tensor value.
ValuePtr wrapAsTensor(ValuePtr value, Operation *srcOp, OpBuilder &builder);

// Wraps a tensor with a TensorToMemrefOp, returning the resulting MemRef value.
ValuePtr wrapAsMemRef(ValuePtr value, Operation *srcOp, OpBuilder &builder);

// For non-{Tensor,MemRef} fetches either the associated MemRef if a LoadOp,
// otherwise append a Alloc and StoreOp.
ValuePtr loadAccessValue(Location location, ValuePtr operand,
                         OpBuilder &builder);

// Adds a LoadOp on a non-{Tensor,MemRef} type that returns the stored value.
ValuePtr loadResultValue(Location location, const Type &originalType,
                         ValuePtr result, OpBuilder &builder);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_MEMREFUTILS_H_
