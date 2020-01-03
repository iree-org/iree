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

#ifndef IREE_COMPILER_DIALECT_HAL_UTILS_TYPEUTILS_H_
#define IREE_COMPILER_DIALECT_HAL_UTILS_TYPEUTILS_H_

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Returns the number of bytes an element of the given type occupies
// post-conversion. For example, the size of i1 would be '1 byte'.
int32_t getRoundedElementByteWidth(Type type);

// Returns an array of i32 values representing the shape of the |shapedType|.
SmallVector<Value, 4> getStaticShapeDims(Location loc, ShapedType shapedType,
                                         ConversionPatternRewriter &rewriter);

// Returns an array of i32 values representing the shape of the |shapedValue|.
SmallVector<Value, 4> getShapeDims(Value shapedValue,
                                   ConversionPatternRewriter &rewriter);

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_UTILS_TYPEUTILS_H_
