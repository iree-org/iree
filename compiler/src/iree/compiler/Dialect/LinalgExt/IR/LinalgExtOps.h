// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_IR_LINALGEXTOPS_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_IR_LINALGEXTOPS_H_

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

// clang-format off


#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtAttrs.h.inc" // IWYU pragma: export

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtPureOps.h.inc" // IWYU pragma: export

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h.inc" // IWYU pragma: export

namespace mlir {
class RewriterBase;
} // namespace mlir

namespace mlir::iree_compiler::IREE::LinalgExt {

/// Replaces each `iree_linalg_ext.index` in the attention region `body`, a
/// score coordinate under `oldScoreMap`, with the value `getOldDim` builds for
/// its original iteration dim. Its last argument builds an index of a new
/// iteration dim, read through `newScoreMap`.
void remapAttentionScoreIndices(
    RewriterBase &rewriter, Block *body, AffineMap oldScoreMap,
    AffineMap newScoreMap,
    function_ref<Value(OpBuilder &, Location, unsigned,
                       function_ref<Value(unsigned)>)>
        getOldDim);

} // namespace mlir::iree_compiler::IREE::LinalgExt

// clang-format on

#endif // IREE_COMPILER_DIALECT_LINALGEXT_IR_LINALGEXTOPS_H_
