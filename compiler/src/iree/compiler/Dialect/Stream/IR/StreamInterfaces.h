// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_STREAM_IR_STREAMINTERACES_H_
#define IREE_COMPILER_DIALECT_STREAM_IR_STREAMINTERACES_H_

#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::IREE::Stream {

using AffinityAndOpPair = std::pair<AffinityAttr, Operation *>;

// The function could be slow, if any data flow analysis is involved. Thus, the
// API provides the batch mode.
using ResolveLayoutAttrFn = std::function<LogicalResult(
    ArrayRef<AffinityAndOpPair> batchQueries,
    llvm::DenseMap<AffinityAndOpPair, SetVector<Attribute>> &layoutAttrs)>;

class AffinityAnalysisDialectInterface
    : public DialectInterface::Base<AffinityAnalysisDialectInterface> {
public:
  AffinityAnalysisDialectInterface(Dialect *dialect) : Base(dialect) {}

  /// The `moduleOp` must remain live and unmodified for as long as the returned
  /// capture is. Otherwise, it will likely be incorrect or crash if the module
  /// op is mutated, especially when module scope analysis is run.
  virtual ResolveLayoutAttrFn
  makeLayoutAttrResolver(ModuleOp moduleOp) const = 0;
};

} // namespace mlir::iree_compiler::IREE::Stream

#endif // IREE_COMPILER_DIALECT_STREAM_IR_STREAM_INTERFACES_H_
