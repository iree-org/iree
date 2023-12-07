// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Stream {
namespace {

static std::optional<int64_t> matchConstant(Value value) {
  if (!value)
    return std::nullopt;
  APInt constant;
  if (!matchPattern(value, m_ConstantInt(&constant)))
    return std::nullopt;
  return constant.getSExtValue();
}

static LogicalResult
verifyAsyncAccessRange(IREE::Stream::AsyncAccessOpInterface accessOp,
                       IREE::Stream::AsyncAccessRange &range) {
  auto start = matchConstant(range.start);
  auto length = matchConstant(range.length);
  auto end = matchConstant(range.end);
  auto resourceSize =
      matchConstant(IREE::Util::SizeAwareTypeInterface::findSizeValue(
          range.resource, accessOp->getBlock(), Block::iterator(accessOp)));

  auto appendValue = [&](InFlightDiagnostic &diagnostic, Value value) {
    std::string str;
    llvm::raw_string_ostream os(str);
    value.printAsOperand(os, OpPrintingFlags());
    diagnostic << str;
  };
  auto emitRangeError = [&]() {
    auto diagnostic = accessOp.emitOpError();
    diagnostic << "has invalid "
               << IREE::Stream::stringifyResourceAccessBitfield(range.access)
               << " access range [";
    start ? (diagnostic << *start) : (diagnostic << "?");
    diagnostic << " to ";
    end ? (diagnostic << *end) : (diagnostic << "?");
    diagnostic << " for ";
    length ? (diagnostic << *length) : (diagnostic << "?");
    diagnostic << "] of resource ";
    appendValue(diagnostic, range.resource);
    diagnostic << " with size ";
    resourceSize ? (diagnostic << *resourceSize) : (diagnostic << "?");
    return diagnostic;
  };

  if (start && end) {
    if (start.value() > end.value()) {
      return emitRangeError() << "; start > end";
    }
  }
  if (length && end) {
    if (length.value() > end.value()) {
      return emitRangeError() << "; length > end";
    }
  }
  if (start && length && end) {
    if (start.value() + length.value() != end.value()) {
      return emitRangeError() << "; start + length != end";
    }
  }
  if (resourceSize) {
    if (start && *start > *resourceSize) {
      return emitRangeError() << "; start > resource size";
    }
    if (length && *length > *resourceSize) {
      return emitRangeError() << "; length > resource size";
    }
    if (end && *end > *resourceSize) {
      return emitRangeError() << "; end > resource size";
    }
  }
  return success();
}

// Statically verifies that the ranges used by |accessOp| are in bounds.
// Emits errors for all ranges declared on the op that are invalid.
static LogicalResult
verifyAsyncAccessOp(IREE::Stream::AsyncAccessOpInterface accessOp) {
  SmallVector<AsyncAccessRange> ranges;
  accessOp.getAsyncAccessRanges(ranges);
  bool allSucceeded = true;
  for (auto &range : ranges) {
    if (failed(verifyAsyncAccessRange(accessOp, range))) {
      allSucceeded = false;
    }
  }
  return success(allSucceeded);
}

class VerifyAsyncAccessRangesPass
    : public VerifyAsyncAccessRangesBase<VerifyAsyncAccessRangesPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Stream::StreamDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    // TODO(benvanik): do whole-program data flow analysis to get bounded sizes
    // for range checking. Today we just do static checks.
    if (moduleOp
            .walk([&](IREE::Stream::AsyncAccessOpInterface accessOp) {
              return succeeded(verifyAsyncAccessOp(accessOp))
                         ? WalkResult::advance()
                         : WalkResult::interrupt();
            })
            .wasInterrupted()) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createVerifyAsyncAccessRangesPass() {
  return std::make_unique<VerifyAsyncAccessRangesPass>();
}

} // namespace mlir::iree_compiler::IREE::Stream
