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

#ifndef IREE_COMPILER_DIALECT_SHAPE_IR_IREETYPES_H_
#define IREE_COMPILER_DIALECT_SHAPE_IR_IREETYPES_H_

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

namespace detail {
struct RankedShapeTypeStorage;
}  // namespace detail

// A shape with a fixed ranked and a mixture of static and dynamic dimensions
// which can express partially shaped values in the tensor domain and be
// easily lowered to the memref domain (only retaining the dynamic dims upon
// conversion).
class RankedShapeType : public Type::TypeBase<RankedShapeType, Type,
                                              detail::RankedShapeTypeStorage> {
 public:
  using Base::Base;

  // Gets an instance of a RankedShapeType given an array of dimensions.
  // Any dynamic dim should be -1.
  static RankedShapeType get(ArrayRef<int64_t> dims, MLIRContext *context);
  static RankedShapeType getChecked(ArrayRef<int64_t> dims, Location loc);

  // Verifies construction invariants and issues errors/warnings.
  static LogicalResult verifyConstructionInvariants(Location loc,
                                                    ArrayRef<int64_t> dims);

  // Gets the rank (counting all dims, static and dynamic).
  int64_t getRank() const;

  // Whether the shape is fully static.
  bool isFullyStatic() const;

  // Gets all dims of this shape, where dynamic dims are represented by -1.
  // The size of the dims vector will be the same as reported by getRank().
  ArrayRef<int64_t> getAllDims() const;

  // Gets the number of dynamic dims.
  unsigned getNumDynamicDims() const;

  // Returns whether the indexed dimension is dynamic.
  bool isDimDynamic(int allDimsIndex) const;

  // Returns the static dimension at the overall shape index.
  // It is an error to request a static index for which isDimDynamic() is
  // true.
  int64_t getStaticDim(int allDimsIndex) const;
};

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_SHAPE_IR_IREETYPES_H_
