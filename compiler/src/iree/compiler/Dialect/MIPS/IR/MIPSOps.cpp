// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/MIPS/IR/MIPSOps.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::MIPS;

//===----------------------------------------------------------------------===//
// MatmulOp — ReifyRankedShapedTypeOpInterface
//
// Returns the output shape [M, N] so IREE's dispatch formation can compute
// the workload when wrapping mips.matmul in a flow.dispatch.workgroups region.
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  // Result is always tensor<M x N>. M from lhs dim 0, N from rhs dim 1.
  reifiedReturnShapes.push_back({tensor::getMixedSize(b, getLoc(), getLhs(), 0),
                                 tensor::getMixedSize(b, getLoc(), getRhs(), 1)});
  return success();
}

//===----------------------------------------------------------------------===//
// MatmulOp — MemoryEffectsOpInterface
//
// In the tensor domain, ops are nominally pure (tensors are values, not memory).
// However mips.matmul uses DPS — the init operand logically "carries" the
// result.  We declare read on lhs/rhs and read+write on init so that alias
// analyses outside of bufferization correctly treat init as modified.
//===----------------------------------------------------------------------===//

void MatmulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInitMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getInitMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// MatmulOp — Verifier
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::verify() {
  auto shape = [](Value v) {
    return cast<RankedTensorType>(v.getType()).getShape();
  };
  auto elemTy = [](Value v) {
    return cast<RankedTensorType>(v.getType()).getElementType();
  };

  // All operands must be 2-D tensors.
  for (Value v : {getLhs(), getRhs(), getInit()}) {
    if (cast<RankedTensorType>(v.getType()).getRank() != 2)
      return emitOpError("all operands must be 2-D ranked tensors");
  }

  // Dimension compatibility: lhs[M x K], rhs[K x N], init[M x N].
  // Only validate static dimensions; dynamic dims are checked at runtime.
  auto compat = [](int64_t a, int64_t b) {
    return ShapedType::isDynamic(a) || ShapedType::isDynamic(b) || a == b;
  };
  if (!compat(shape(getLhs())[0], shape(getInit())[0]))
    return emitOpError("lhs dim 0 (M) must match init dim 0 (M)");
  if (!compat(shape(getLhs())[1], shape(getRhs())[0]))
    return emitOpError("lhs dim 1 (K) must match rhs dim 0 (K)");
  if (!compat(shape(getRhs())[1], shape(getInit())[1]))
    return emitOpError("rhs dim 1 (N) must match init dim 1 (N)");

  // All element types must match.
  if (elemTy(getLhs()) != elemTy(getRhs()) || elemTy(getLhs()) != elemTy(getInit()))
    return emitOpError("element types of all operands must match");

  // Result type must match init type (both tensor<MxNxf32>).
  if (getResult().getType() != getInit().getType())
    return emitOpError("result type must match init type");

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/MIPS/IR/MIPSOps.cpp.inc"
