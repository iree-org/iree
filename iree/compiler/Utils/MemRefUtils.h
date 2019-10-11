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

static const int kBoolBitWidth = 8;
static const int kIndexBitWidth = 32;

// Converts types to MemRefs using convertTypeToMemRef.
class MemRefTypeConverter : public TypeConverter {
 public:
  explicit MemRefTypeConverter(MLIRContext *context) {}
  Type convertType(Type type) override;
};

Type legalizeType(Type type);

// Converts a type (scalar, tensor, etc) to a MemRef-based type.
MemRefType convertTypeToMemRef(Type type);

// Attempts to resolve the use of a MemRef back to the value stored into it.
// Returns either the value stored into the given index or nullptr if the value
// is unavailable (or possibly unknown).
Value *resolveMemRefSourceValue(Value *memRef, Operation *useOp,
                                llvm::ArrayRef<Value *> indices = {});

// Attempts to resolve the use of a value back to the MemRef it was loaded from.
// Returns either a MemRef view containing the value or nullptr if the value was
// not loaded from a MemRef (or is possibly unknown).
Value *resolveValueToSourceMemRef(Value *value, Operation *useOp);

// Returns an equivalent TensorType for a MemRef value or returns the values
// current type.
Type getTensorType(Value *value, OpBuilder &builder);

// Returns an equivalent MemRefType for a Tensor value or returns the values
// current type.
Type getMemRefType(Value *value, OpBuilder &builder);

// Wraps a memref with a MemRefToTensorOp, returning the resulting Tensor value.
Value *wrapAsTensor(Value *value, Operation *srcOp, OpBuilder &builder);

// Wraps a tensor with a TensorToMemrefOp, returning the resulting MemRef value.
Value *wrapAsMemRef(Value *value, Operation *srcOp, OpBuilder &builder);

// For non-{Tensor,MemRef} fetches either the associated MemRef if a LoadOp,
// otherwise append a Alloc and StoreOp.
Value *loadAccessValue(Location location, Value *operand, OpBuilder &builder);

// Adds a LoadOp on a non-{Tensor,MemRef} type that returns the stored value.
Value *loadResultValue(Location location, const Type &originalType,
                       Value *result, OpBuilder &builder);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_MEMREFUTILS_H_
