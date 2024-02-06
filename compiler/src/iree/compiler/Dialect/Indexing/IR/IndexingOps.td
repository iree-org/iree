// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_INDEXING_IR_INDEXING_OPS
#define IREE_DIALECT_INDEXING_IR_INDEXING_OPS

include "iree/compiler/Dialect/Indexing/IR/IndexingBase.td"
include "iree/compiler/Dialect/Indexing/IR/IndexingInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Op types
//===----------------------------------------------------------------------===//

class Indexing_Op<string mnemonic, list<Trait> traits = []> :
    Op<Indexing_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;
}

class Indexing_PureOp<string mnemonic, list<Trait> traits = []> :
    Indexing_Op<mnemonic, !listconcat(traits, [Pure])>;

//===----------------------------------------------------------------------===//
// Type manipulation
//===----------------------------------------------------------------------===//

def OpGroupAssertionOps : OpDocGroup {
  let summary = "Type manipulation ops";
  let description = "";
}

let opDocGroup = OpGroupAssertionOps in {

def Indexing_AssertAlignedRangeOp : Indexing_PureOp<"assert.aligned_range", [
  AllTypesMatch<["value", "result"]>,
  Indexing_StaticBoundsOpInterface,
  DeclareOpInterfaceMethods<Indexing_StaticBoundsOpInterface, [
    "getIndexRange",
    "initializeRange",
  ]>,
]> {
  let summary = [{asserts a static aligned range of an index like value}];
  let description = [{
    Allows inserting range and alignment requirements in SSA chains.
  }];

  let arguments = (ins
    Indexing_IndexLike:$value,
    OptionalAttr<SI64Attr>:$minValue,
    OptionalAttr<SI64Attr>:$maxValue,
    DefaultValuedAttr<I64Attr, "1">:$alignment
  );
  let results = (outs
    Indexing_IndexLike:$result
  );

  let assemblyFormat = [{
    $value `range` `(`
    ($minValue^):(`UNBOUNDED`)? `,`
    ($maxValue^):(`UNBOUNDED`)? `)`
    (`align` `(` $alignment^ `)`)? attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
}

def Indexing_AssertDimRangeOp : Indexing_PureOp<"assert.dim_range", [
  AllTypesMatch<["value", "result"]>,
  Indexing_StaticBoundsOpInterface,
  DeclareOpInterfaceMethods<Indexing_StaticBoundsOpInterface, [
    "getDynamicDimRanges",
    "initializeDimRanges",
  ]>,
]> {
  let summary = [{asserts a static aligned range of a dimension}];
  let description = [{
    Allows inserting range and alignment requirements in SSA chains for dynamic
    dimensions of shaped values.
  }];

  let arguments = (ins
    AnyShaped:$value,
    I64Attr:$dim,
    OptionalAttr<SI64Attr>:$minValue,
    OptionalAttr<SI64Attr>:$maxValue,
    DefaultValuedAttr<I64Attr, "1">:$alignment
  );
  let results = (outs
    AnyShaped:$result
  );

  let assemblyFormat = [{
    $value `[` $dim `]` `range` `(`
    ($minValue^):(`UNBOUNDED`)? `,`
    ($maxValue^):(`UNBOUNDED`)? `)`
    (`align` `(` $alignment^ `)`)? attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
}

} // OpGroupAssertionOps

#endif  // IREE_DIALECT_INDEXING_IR_INDEXING_OPS
