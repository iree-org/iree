// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_SMTCONSTRAINTUTILS_H_
#define IREE_COMPILER_CODEGEN_COMMON_SMTCONSTRAINTUTILS_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir::iree_compiler {

/// Result of creating a ConstraintsOp shell with common constraints.
struct ConstraintsOpShell {
  IREE::Codegen::ConstraintsOp op;
  /// Block args of type !smt.int, one per loop dim.
  SmallVector<Value> smtDimArgs;
};

/// Creates a ConstraintsOp with common constraints already emitted.
///
/// Extracts dim values from rootOp operands via indexingMaps, creates the
/// op with block args, emits static dim constraints. Returns with builder
/// positioned at the end of the block, ready for pipeline-specific
/// constraints.
ConstraintsOpShell createConstraintsOpShell(
    OpBuilder &builder, Operation *rootOp, IREE::Codegen::RootOpAttr rootOpAttr,
    IREE::Codegen::PipelineAttrInterface pipelineAttr, DictionaryAttr knobs,
    unsigned numLoops, ArrayRef<AffineMap> indexingMaps);

/// Helper to create an SMT integer constant.
Value mkIntConst(OpBuilder &builder, Location loc, int64_t v);

/// Helper to create an SMT knob op.
Value mkKnob(OpBuilder &builder, Location loc, StringRef name);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_SMTCONSTRAINTUTILS_H_
