// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGPLUS_OPS
#define IREE_DIALECT_LINALGPLUS_OPS

include "iree/compiler/Dialect/LinalgPlus/IR/LinalgPlusBase.td"
include "iree/compiler/Dialect/LinalgPlus/IR/LinalgPlusInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

class LinalgPlus_PureOp<string mnemonic, list<OpTrait> traits = []> :
    Op<LinalgPlus_Dialect, mnemonic, traits> {
}

class LinalgPlus_Op<string mnemonic, list<OpTrait> traits = []> :
    LinalgPlus_PureOp<mnemonic, !listconcat(traits, [LinalgPlusInterface])> {
  let verifier = [{ return verify$cppClass(*this); }];
  let printer = [{ return print$cppClass(p, *this); }];
  let parser = [{ return parse$cppClass(parser, result); }];
}

//===----------------------------------------------------------------------===//
// Non-structured ops
//===----------------------------------------------------------------------===//

def LinalgPlus_SortOp : LinalgPlus_Op<"sort", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Sort operator";
  let description = [{
    Based on XLA operation semantics, sorts the given `operands` at the given
    `dimension` with the given `comparator`.

    See https://www.tensorflow.org/xla/operation_semantics#sort.
  }];

  // Define arguments and results like linalg.generic op. The attribute has the
  // same definision as mhlo.sort::dimension. If the rank is greater than 1,
  // the attribute must be set. If the rank is exacatly 1, the dimension is
  // optional.
  let arguments = (ins Variadic<AnyType>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       OptionalAttr<I64Attr>:$dimension
  );
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
  let regions = (region AnyRegion:$region);
}

//===----------------------------------------------------------------------===//
// Pure ops
//===----------------------------------------------------------------------===//

def LinalgPlus_YieldOp : LinalgPlus_PureOp<"yield", [NoSideEffect, ReturnLike, Terminator]> {
  let summary = "LinalgPlus yield op";
  let description = [{
    `linalg_plus.yield` is a special terminator operation for blocks inside
    regions in `linalg_plus` ops.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif  // IREE_DIALECT_LINALGPLUS_OPS
