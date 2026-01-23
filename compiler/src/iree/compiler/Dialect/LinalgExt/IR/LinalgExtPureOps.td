// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGEXT_PURE_OPS
#define IREE_DIALECT_LINALGEXT_PURE_OPS

include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtBase.td"
include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// Utility ops
//===----------------------------------------------------------------------===//

def OpGroupUtilityOps : OpDocGroup {
  let summary = "Utility ops";
  let description = "";
}

let opDocGroup = OpGroupUtilityOps in {

def IREELinalgExt_IndexOp : IREELinalgExt_PureOp<"index", [Pure]>,
    Arguments<(ins ConfinedAttr<I64Attr, [IntMinValue<0>]>:$dim)>,
    Results<(outs Index:$result)> {
  let summary = [{LinalgExt index operation.}];
  let description = [{
    This operation is a mirror of `linalg.index` operation and has the same
    semantics, except that `linalg.index` enforces that the parent op is a
    `LinalgOp`, and the `iree_linalg_ext.index` operation enforces that the
    parent op is one of `IREE::LinalgExt::CustomOp` or `IREE::LinalgExt::AttentionOp`.
  }];

  let assemblyFormat = [{ $dim attr-dict `:` type($result) }];
  let hasVerifier = 1;
}

def IREELinalgExt_YieldOp : IREELinalgExt_PureOp<"yield", [Pure, ReturnLike, Terminator]> {
  let summary = [{LinalgExt yield op.}];
  let description = [{
    `iree_linalg_ext.yield` is a special terminator operation for blocks inside
    regions in `iree_linalg_ext` ops.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

} // OpGroupUtilityOps

#endif  // IREE_DIALECT_LINALGEXT_PURE_OPS
