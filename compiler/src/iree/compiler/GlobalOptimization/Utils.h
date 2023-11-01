// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef IREE_COMPILER_GLOBALOPTIMIZATION_UTILS_H_
#define IREE_COMPILER_GLOBALOPTIMIZATION_UTILS_H_

namespace mlir {
class MLIRContext;
class IntegerType;
class Type;
class Operation;
class Value;

namespace iree_compiler {
namespace GlobalOptimization {

// If the producer is a CastOpInterface, or a linalg::GenericOp that performs
// only a CastOpInterface on its input, return the CastOpInterface op.
//
// **Note: If the CastOpInterface has been generalized, the return Operation
//         is the body CastOpInterface op, not the linalg::GenericOp.
Operation *getDefiningCastOp(Value input);

// Returns an IntegerType with the specified bitwidth and signedness.
IntegerType getIntegerTypeWithSignedness(MLIRContext *ctx, int bitWidth,
                                         bool isSigned);

// Returns the source element type of the defining CastOpInterface of `input`,
// if there is one.
Type getCastElemType(Value input);

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_GLOBALOPTIMIZATION_UTILS_H_
