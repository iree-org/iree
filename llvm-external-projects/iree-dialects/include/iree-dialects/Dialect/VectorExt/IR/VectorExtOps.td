// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_VECTOREXT_OPS
#define IREE_DIALECT_VECTOREXT_OPS

include "iree-dialects/Dialect/VectorExt/IR/VectorExtBase.td"
include "iree-dialects/Dialect/VectorExt/IR/VectorExtInterfaces.td"

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

class IREEVectorExt_PureOp<string mnemonic, list<Trait> traits = []> :
    Op<IREEVectorExt_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Layout ops.
//===----------------------------------------------------------------------===//

def IREEVectorExt_LayoutConflictResolutionOp : IREEVectorExt_PureOp<"layout_conflict_resolution"> {
  let summary = "Layout Conflict Resolution operator";
  let description = [{
    The layout conflict resolution operator takes a vector and a
    desired layout and transforms the vector to one with the
    desired layout.
  }];
  let arguments = (ins
    AnyVector:$input,
    VectorLayoutInterface:$sourceLayout,
    VectorLayoutInterface:$desiredLayout
  );
  let results = (outs
    AnyVector:$output
  );
  let extraClassDeclaration = [{}];
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}

#endif  // IREE_DIALECT_VECTOREXT_OPS

