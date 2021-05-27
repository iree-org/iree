// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_SHAPE_IR_IREETYPES_H_
#define IREE_COMPILER_DIALECT_SHAPE_IR_IREETYPES_H_

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
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
  static RankedShapeType getChecked(
      function_ref<InFlightDiagnostic()> emitError, MLIRContext *context,
      ArrayRef<int64_t> dims);

  // Derives a RankedShapeType from a ShapedType.
  static RankedShapeType get(ShapedType shapedType);

  // Verifies construction invariants and issues errors/warnings.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
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
