//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_DIALECTS_DIALECT_TMTENSOR_IR_TMTENSOROPS_H_
#define TORCH_MLIR_DIALECTS_DIALECT_TMTENSOR_IR_TMTENSOROPS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/ScalarLoopOpInterface.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorInterfaces.h"

namespace mlir {
namespace torch {
namespace TMTensor {

/// Returns a `memref.dim` or `tensor.dim` operation to get the shape of `v` at
/// `dim`.
Value getDimValue(OpBuilder &builder, Location loc, Value v, int64_t dim);

/// Returns a `memref.dim` or `tensor.dim` operation to get the shape of `v` at
/// `dim`. If the shape is constant, returns the shape as an `IntegerAttr`.
OpFoldResult getDim(OpBuilder &builder, Location loc, Value v, int64_t dim);

} // namespace TMTensor
} // namespace torch
} // namespace mlir

#define GET_OP_CLASSES
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.h.inc" // IWYU pragma: export

#endif // TORCH_MLIR_DIALECTS_DIALECT_TMTENSOR_IR_TMTENSOROPS_H_
