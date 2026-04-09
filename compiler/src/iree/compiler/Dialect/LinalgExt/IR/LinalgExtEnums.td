// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Standalone enum definitions for the LinalgExt dialect. This file contains
// only I32EnumAttr definitions with no dialect dependency, so it can be
// processed by --gen-enum-decls without pulling in StructuredOpsUtils.td
// (which would cause IteratorType redefinition conflicts).

#ifndef IREE_DIALECT_LINALGEXT_ENUMS
#define IREE_DIALECT_LINALGEXT_ENUMS

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Im2col decomposition mode
//===----------------------------------------------------------------------===//

def IREELinalgExt_Im2colDecomposeMode_StreamCopy
    : I32EnumAttrCase<"StreamCopy", 0, "stream_copy">;
def IREELinalgExt_Im2colDecomposeMode_AsyncCopy
    : I32EnumAttrCase<"AsyncCopy",  1, "async_copy">;

def IREELinalgExt_Im2colDecomposeMode
    : I32EnumAttr<"Im2colDecomposeMode",
                  "Mode selecting how an im2col op is decomposed",
                  [IREELinalgExt_Im2colDecomposeMode_StreamCopy,
                   IREELinalgExt_Im2colDecomposeMode_AsyncCopy]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::LinalgExt";
  // Prevent generation of the old-style IntegerAttr wrapper class; the
  // AttrDef-based Im2colDecomposeModeAttr serves that role.
  let genSpecializedAttr = 0;
}

#endif // IREE_DIALECT_LINALGEXT_ENUMS
