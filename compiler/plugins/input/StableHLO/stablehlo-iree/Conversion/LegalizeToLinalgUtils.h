// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Utils for lowering of the StableHLO dialect to the Linalg dialect.

#ifndef STABLEHLO_IREE_CONVERSION_LEGALIZE_TO_LINALG_UTILS_H_
#define STABLEHLO_IREE_CONVERSION_LEGALIZE_TO_LINALG_UTILS_H_

#include <algorithm>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo-iree/Conversion/MapStableHLOToScalarOp.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

/// Returns an ArrayAttr that contains `nLoops` attributes. All the attributes
/// are "parallel" except the last `nReduction` elements, where are "reduction"
/// attributes.
SmallVector<utils::IteratorType, 3>
getParallelAndReductionIterators(unsigned nLoops, unsigned nReduction);

/// Returns an ArrayAttr that contains `nParallelLoops` "parallel" attributes.
SmallVector<utils::IteratorType, 3>
getNParallelLoopsAttrs(unsigned nParallelLoops);

/// Generates an init sparse tensor.
Value getEmptySparseTensor(OpBuilder &b, Location loc, ShapedType type,
                           ArrayRef<Value> dynSizes);

/// Generates a tensor.empty op.
Value getEmptyTensor(OpBuilder &b, Location loc, ShapedType type,
                     ArrayRef<Value> dynSizes);

/// Generates an empty tensor for the result of the operation, which could be a
/// dense tensor or a sparse tensor.
Value getEmptyTensorFor(OpBuilder &b, Location loc, ShapedType resultType,
                        Operation *op, ValueRange operands);

/// Ensures a tensor has the same shape (not including the element type) as
/// another.
Value coerceTensorShape(OpBuilder &builder, Location loc,
                        TypedValue<ShapedType> value, ShapedType targetType);

/// Verifies |op|'s semantics by checking if all operands and results have
/// ranged tensor types.
LogicalResult verifyHloOpBufferOrTensorSemantics(Operation *op);

/// Fills |tensor| with a zero constant of the matching type. Returns the new
/// value.
Value fillTensorWithZeros(OpBuilder &builder, Location loc, Value tensor);

/// Sparsifies a (block of) operation(s) that cannot be handled directly
/// by the sparse compiler but has well-known semi-ring semantics.
///
/// This yields something of the following form:
///
///   %result = sparse_tensor.unary %values[0]
///     present={
///       ^bb1(%val):
///         ... codegen proceeds here using %val ....
///         sparse_tensor.yield
///     }
///     absent={}
///   linalg.yield %result
Value preSparsify(Operation *op, llvm::SmallVector<Value, 2> &values, Type rtp,
                  OpBuilder *b);

/// Finalizes sparse semi-ring construction.
Value postSparsify(Operation *op, Value semiring, Value result, OpBuilder *b);

/// Returns true if all operands are tensors with rank 0.
bool allOperandsAreScalarTensors(Operation *op);

/// Returns true if parent op is linalg.
bool isInBodyOfLinalgOps(Operation *op);

/// Extracts integer values from the attribute |elements|.
SmallVector<int64_t> extract1DVector(DenseIntElementsAttr elements);

/// Returns true if the given |values| is a splat of the given |queryValue|.
inline bool isSplatValue(const ArrayRef<int64_t> &values, int64_t queryValue) {
  for (auto value : values) {
    if (value != queryValue) {
      return false;
    }
  }
  return true;
}

/// Returns true if the given |attr| is a splat of the given |value|.
inline bool isSplatValue(DenseIntElementsAttr attr, uint64_t value) {
  return attr.isSplat() && attr.getSplatValue<uint64_t>() == value;
}

} // namespace mlir::iree_compiler::stablehlo

#endif // STABLEHLO_IREE_CONVERSION_LEGALIZE_TO_LINALG_UTILS_H_
