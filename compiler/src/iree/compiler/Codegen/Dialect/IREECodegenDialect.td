// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_IREECODEGEN_DIALECT
#define IREE_CODEGEN_DIALECT_IREECODEGEN_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IREE Codegen dialect
//===----------------------------------------------------------------------===//

def IREECodegen_Dialect : Dialect {
  let name = "iree_codegen";
  let cppNamespace = "::mlir::iree_compiler::IREE::Codegen";

  let summary = [{
    A dialect representing attributes used by the IREE Code generation.
  }];
  let description = [{
    This dialect is primarily meant to hold attributes that carry the
    state of the compilation when lowered to scalar code for an
    architecture. Typically, a backend starts by analysing the entry
    point functions within the `hal.executable.variant` and deciding
    which compilation pipeline to chose. During this, even the values
    for parameters such as tile sizes, etc. are also decided. The rest
    of the compilation flow does not make any heuristic decisions,
    rather just looks at the values of the decision specified in
    attributes that belong to this dialect. This allows an external
    search to easily override the heuristics that are hard-coded
    within a backend.
  }];
  let extraClassDeclaration = [{
    void initializeCodegenAttrs();
  }];
  let useDefaultAttributePrinterParser = 1;
}

def AnyRankedTensorOrMemRefType : AnyTypeOf<[AnyRankedTensor, AnyMemRef]>;

class RankedTensorOrMemRefType<list<Type> allowedTypes, list<int> ranks>
    : AnyTypeOf<[TensorRankOf<allowedTypes, ranks>,
        MemRefRankOf<allowedTypes, ranks>]>;

#endif // IREE_CODEGEN_DIALECT_IREECODEGEN_DIALECT
