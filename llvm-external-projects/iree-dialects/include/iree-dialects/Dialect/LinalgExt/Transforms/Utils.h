// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_UTILS_H_
#define IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_UTILS_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class Location;
class OpBuilder;
class Operation;
class Value;

namespace tensor {
class ExtractSliceOp;
}

namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

/// Helper function which auto-completes the missing trailing dimensions to
/// always be offset = 0, size = dim, stride = 1.
void completeOffsetsSizesAndStrides(OpBuilder &b, Location loc, Value tensor,
                                    ArrayRef<Value> leadingOffsets,
                                    ArrayRef<Value> leadingSizes,
                                    ArrayRef<Value> leadingStrides,
                                    SmallVectorImpl<Value> &offsets,
                                    SmallVectorImpl<Value> &sizes,
                                    SmallVectorImpl<Value> &strides);

/// Create a tensor::ExtractSliceOp by auto-completing the missing trailing
/// dimensions to always be offset = 0, size = dim, stride = 1.
Value createSubsetExtractOpFromLeadingOffsetsSizesAndStrides(
    OpBuilder &b, Location loc, Value tensor,
    llvm::ArrayRef<Value> leadingOffsets, ArrayRef<Value> leadingSizes,
    ArrayRef<Value> leadingStrides);

/// Create a tensor::InsertSliceOp by auto-completing the missing trailing
/// dimensions to always be offset = 0, size = dim, stride = 1.
Value createSubsetInsertOpFromLeadingOffsetsSizesAndStrides(
    OpBuilder &b, Location loc, Value tensor, Value dest,
    ArrayRef<Value> leadingOffsets, ArrayRef<Value> leadingSizes,
    ArrayRef<Value> leadingStrides);

/// Insert the `source` tensor into the `dest` tensor by creating the relevant
/// `subset_insert` op. The details of the `subset_insert` op are retrieved
/// from the `subset_extract` op so that they form a matching extract/insert
/// pair.
Value createMatchingSubsetInsertOp(OpBuilder &b, Location loc,
                                   tensor::ExtractSliceOp subsetExtractOp,
                                   Value source, Value dest);

/// Create the parallel insertion terminator version of
/// `createMatchingSubsetInsertOp`.
void createMatchingParallelSubsetInsertOp(
    OpBuilder &b, Location loc, tensor::ExtractSliceOp subsetExtractOp,
    Value source, Value dest);

struct AffineValueExpr {
  explicit AffineValueExpr(AffineExpr e) : e(e) {}
  AffineValueExpr bind(Value v) {
    this->v = v;
    return *this;
  }
  operator AffineExpr() const { return e; }
  operator Value() const { return v; }
  AffineExpr e;
  Value v;
};

/// Helper struct to build simple arithmetic quantiAffineValueExprs with minimal
/// type inference support.
// TODO: move into ArithBuilder once ops have been moved into arith.
struct AffineBuilder {
  AffineBuilder(OpBuilder &b, Location loc) : b(b), loc(loc) {}

  Value add(AffineValueExpr lhs, AffineValueExpr rhs) {
    return b.createOrFold<AffineApplyOp>(
        loc, ArrayRef<AffineExpr>{lhs.e + rhs.e}, ValueRange{lhs, rhs});
  }
  Value sub(AffineValueExpr lhs, AffineValueExpr rhs) {
    return b.createOrFold<AffineApplyOp>(
        loc, ArrayRef<AffineExpr>{lhs.e - rhs.e}, ValueRange{lhs, rhs});
  }
  Value mul(AffineValueExpr lhs, AffineValueExpr rhs) {
    return b.createOrFold<AffineApplyOp>(
        loc, ArrayRef<AffineExpr>{lhs.e * rhs.e}, ValueRange{lhs, rhs});
  }
  Value ceil(AffineValueExpr lhs, AffineValueExpr rhs) {
    return b.createOrFold<AffineApplyOp>(
        loc, ArrayRef<AffineExpr>{lhs.e.ceilDiv(rhs.e)}, ValueRange{lhs, rhs});
  }
  Value min(ValueRange vals) {
    return b.createOrFold<AffineMinOp>(
        loc, AffineMap::getMultiDimIdentityMap(vals.size(), b.getContext()),
        vals);
  }
  Value max(ValueRange vals) {
    return b.createOrFold<AffineMaxOp>(
        loc, AffineMap::getMultiDimIdentityMap(vals.size(), b.getContext()),
        vals);
  }

private:
  OpBuilder &b;
  Location loc;
};

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_UTILS_H_
