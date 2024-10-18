// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/SymbolRanges/Index.h"
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::Util {

namespace {

static StringAttr getMapString(MLIRContext *context, AffineMap lb,
                               AffineMap ub) {
  std::string s;
  llvm::raw_string_ostream stream(s);
  if (lb) {
    stream << "lower bounds: ";
    stream << lb;
    if (ub)
      stream << ", ";
  }
  if (ub) {
    stream << "upper bounds: ";
    stream << ub;
  }
  return StringAttr::get(context, s);
}

static StringAttr getBoundsString(MLIRContext *context,
                                  std::optional<int64_t> lb,
                                  std::optional<int64_t> ub) {
  std::string s;
  llvm::raw_string_ostream stream(s);
  stream << "lower bound: ";
  if (lb) {
    stream << *lb;
  } else {
    stream << "-INFINITY";
  }
  stream << ", upper bound: ";
  if (ub) {
    stream << ub;
  } else {
    stream << "INFINITY";
  }
  return StringAttr::get(context, s);
}

class TestIndexRangeAnalysisPass
    : public TestIndexRangeAnalysisBase<TestIndexRangeAnalysisPass> {
public:
  void runOnOperation() override {

    IndexRangeAnalysis analysis(getOperation());

    // Solve.
    if (failed(analysis.run())) {
      return signalPassFailure();
    }

    // Update.
    for (auto func :
         getOperation()->getRegion(0).getOps<FunctionOpInterface>()) {
      func.walk([&](Operation *op) {
        for (auto [iter, o] : llvm::enumerate(op->getOperands())) {
          int64_t idx = iter;
          Value operand = o;
          TypeSwitch<Type>(operand.getType())
              .Case([&](ShapedType shapedType) {
                // Track all dynamic dimensions for the given shaped type.
                for (int64_t i = 0, e = shapedType.getRank(); i < e; ++i) {
                  if (shapedType.isDynamicDim(i)) {
                    std::stringstream ss;
                    ss << "operand_";
                    ss << idx;
                    ss << "_dim_";
                    ss << i;

                    auto [lb, ub] = analysis.getConstantBounds(operand, i);
                    if (lb || ub) {
                      op->setAttr(ss.str(),
                                  getBoundsString(&getContext(), lb, ub));
                      continue;
                    }

                    auto [lbmap, ubmap] = analysis.getBounds(operand, i);
                    op->setAttr(ss.str(),
                                getMapString(&getContext(), lbmap, ubmap));
                  }
                }
              })
              .Case([&](IndexType) {
                std::stringstream ss;
                ss << "operand_";
                ss << idx;

                auto [lb, ub] =
                    analysis.getConstantBounds(operand, std::nullopt);
                if (lb || ub) {
                  op->setAttr(ss.str(), getBoundsString(&getContext(), lb, ub));
                  return;
                }

                auto [lbmap, ubmap] = analysis.getBounds(operand, std::nullopt);

                op->setAttr(ss.str(),
                            getMapString(&getContext(), lbmap, ubmap));
              });
        }
      });
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<void>> createTestIndexRangeAnalysisPass() {
  return std::make_unique<TestIndexRangeAnalysisPass>();
}

} // namespace mlir::iree_compiler::IREE::Util
