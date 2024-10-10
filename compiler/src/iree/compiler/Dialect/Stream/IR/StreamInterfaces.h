// Copyright 2024 The IREE Authors
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

class AffinityAnalysisDialectInterface
    : public DialectInterface::Base<AffinityAnalysisDialectInterface> {
public:
  AffinityAnalysisDialectInterface(Dialect *dialect) : Base(dialect) {}

  virtual std::function<LogicalResult(AffinityAttr, Operation *,
                                      SetVector<Attribute> &)>
  makeTargetResolver(ModuleOp moduleOp) const = 0;
};

} // namespace mlir::iree_compiler::IREE::Stream

#endif // IREE_COMPILER_DIALECT_STREAM_IR_STREAM_INTERFACES_H_
