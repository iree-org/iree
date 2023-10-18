//===- PatternApplicatorPassBase.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base class with shared implementation for custom pattern applicator
// passes.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_UTILS_CUSTOMPATTERNAPPLICATORPASSBASE_H_
#define IREE_COMPILER_UTILS_CUSTOMPATTERNAPPLICATORPASSBASE_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

using transform::TransformOptions;

namespace iree_compiler {

namespace detail {
void populateCommonNativeRewriteHelpers(RewritePatternSet &patterns);

LogicalResult populatePDLModuleFromFileName(MLIRContext *context,
                                            RewritePatternSet &patterns,
                                            llvm::StringRef pdlModuleFileName);
} // namespace detail

template <typename Concrete, template <typename> typename GeneratedBase>
class PatternApplicatorPassBase : public GeneratedBase<Concrete> {
public:
  explicit PatternApplicatorPassBase(
      const TransformOptions &options = TransformOptions())
      : options(options) {}

  PatternApplicatorPassBase(const PatternApplicatorPassBase &pass) {
    options = pass.options;
  }

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet tmpPatterns(context);
    detail::populateCommonNativeRewriteHelpers(tmpPatterns);
    if (failed(static_cast<Concrete *>(this)->initializePatterns(
            context, tmpPatterns))) {
      return failure();
    }
    patterns = std::move(tmpPatterns);
    return success();
  }

  // Hook for populating necessary library constraints/rewrites for the pattern
  // applicator at initialization time, as well as setting up the type
  // converter.
  LogicalResult initializePatterns(MLIRContext *context,
                                   RewritePatternSet &tmpPatterns) {
    return success();
  }

  void runOnOperation() override {
    auto *pass = static_cast<Concrete *>(this);
    Operation *op = pass->getOperation();

    /// If there are no patterns nothing to do.
    if (!patterns.getPDLByteCode()) {
      return;
    }
    if (failed(applyPatternsAndFoldGreedily(op, patterns)))
      return pass->signalPassFailure();
  }

private:
  /// Pattern applicator options.
  TransformOptions options;

  FrozenRewritePatternSet patterns;
};

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_UTILS_CUSTOMPATTERNAPPLICATORPASSBASE_H_
