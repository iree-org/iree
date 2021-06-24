// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGPLUS_OPS
#define IREE_DIALECT_LINALGPLUS_OPS

include "iree/compiler/Dialect/LinalgPlus/IR/LinalgPlusBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

class LinalgPlus_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<LinalgPlus_Dialect, mnemonic, traits> {
  let verifier = [{ return verify$cppClass(*this); }];
}

//===----------------------------------------------------------------------===//
// Non-structured ops
//===----------------------------------------------------------------------===//

def LinalgPlus_SortOp : LinalgPlus_Op<"sort", [RecursiveSideEffects,
                                               SameOperandsAndResultShape]> {
  let summary = "Sort operator";
  let description = [{
    Based on XLA operation semantics, sorts the given `operands` at the given
    `dimension` with the given `comparator`.

    See https://www.tensorflow.org/xla/operation_semantics#sort.
  }];
  let arguments = (ins
    Variadic<AnyType>:$operands,
    DefaultValuedAttr<I64Attr, "-1">:$dimension
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$comparator);

  let builders = [
    OpBuilder<(ins "ValueRange":$operands,
      CArg<"int64_t", "-1">:$dimension)>
  ];
}

def LinalgPlus_YieldOp : LinalgPlus_Op<"yield", [NoSideEffect, ReturnLike, Terminator]> {
  let summary = "yield op";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif  // IREE_DIALECT_LINALGPLUS_OPS
