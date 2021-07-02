// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGEXT_OPS
#define IREE_DIALECT_LINALGEXT_OPS

include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtBase.td"
include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

class LinalgExt_PureOp<string mnemonic, list<OpTrait> traits = []> :
    Op<LinalgExt_Dialect, mnemonic, traits> {
}

class LinalgExt_Op<string mnemonic, list<OpTrait> traits = []> :
    LinalgExt_PureOp<mnemonic, !listconcat(traits,
        [AttrSizedOperandSegments,
         DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
         LinalgExtInterface, SingleBlockImplicitTerminator<"YieldOp">])> {
  let verifier = [{ return verify$cppClass(*this); }];
  let printer = [{ return print$cppClass(p, *this); }];
  let parser = [{ return parse$cppClass(parser, result); }];
}

//===----------------------------------------------------------------------===//
// Non-structured ops
//===----------------------------------------------------------------------===//

def LinalgExt_ScatterOp : LinalgExt_Op<"scatter"> {
  let summary = "Scatter operator";
  let description = [{
    Based on XLA operation semantics, takes two `inputs` (`update` and
    `indices`) and `outputs` value (`original`). The operation updates
    the value at the slices specified by `indices` by combining the
    current value with the value in `updates` using the computation
    specified in `region`. The `region` specifies a binary operation
    of signature (T, T) -> T, where `T` is the element-type of
    `updates` (and `original`). The first argument correspond the
    value to be updated (i.e. from `updates`), and the second the
    current value (i.e. value from `original`).
  }];
  let arguments = (ins
      Variadic<AnyRankedTensorOrMemRefType>:$inputs,
      Variadic<AnyRankedTensorOrMemRefType>:$outputs
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = [{
    attr-dict custom<LinalgExtInsList>($inputs, type($inputs))
    custom<LinalgExtOutsList>($outputs, type($outputs))
    $region (`->` type($results)^)?
  }];
}

def LinalgExt_SortOp : LinalgExt_Op<"sort"> {
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
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = [{
    (`dimension` `(` $dimension^ `)`)?
    attr-dict custom<LinalgExtInsList>($inputs, type($inputs))
    custom<LinalgExtOutsList>($outputs, type($outputs))
    $region (`->` type($results)^)?
  }];
}

//===----------------------------------------------------------------------===//
// Pure ops
//===----------------------------------------------------------------------===//

def LinalgExt_YieldOp : LinalgExt_PureOp<"yield", [NoSideEffect, ReturnLike, Terminator]> {
  let summary = "LinalgExt yield op";
  let description = [{
    `linalg_ext.yield` is a special terminator operation for blocks inside
    regions in `linalg_ext` ops.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif  // IREE_DIALECT_LINALGEXT_OPS
