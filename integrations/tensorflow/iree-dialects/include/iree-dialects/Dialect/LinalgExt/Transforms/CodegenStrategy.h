// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_CODEGENSTRATEGY_H_
#define IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_CODEGENSTRATEGY_H_

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Pass/PassManager.h"

#include <utility>

//===----------------------------------------------------------------------===//
// Strategies moved from upstream MLIR as IREE still heavily relies on patterns
// that compose through filters.
// TODO: Deprecate everything below.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

/// Abstract Transformation class applied in a sequence that also handles state
/// through markers.
struct Transformation {
  explicit Transformation(
      LinalgExt::LinalgTransformationFilter::FilterFunction f)
      : filter(std::move(f)) {}
  virtual ~Transformation() = default;
  virtual void
  addToPassPipeline(OpPassManager &pm,
                    LinalgExt::LinalgTransformationFilter m) const = 0;
  LinalgExt::LinalgTransformationFilter::FilterFunction filter = nullptr;
};

/// Represent one application of createLinalgStrategyLowerVectorsPass.
struct VectorLowering : public Transformation {
  explicit VectorLowering(
      LinalgVectorLoweringOptions options,
      LinalgExt::LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(std::move(f)), options(options) {}

  void
  addToPassPipeline(OpPassManager &pm,
                    LinalgExt::LinalgTransformationFilter m) const override {
    pm.addPass(createLinalgStrategyLowerVectorsPass(options, m));
  }

private:
  LinalgVectorLoweringOptions options;
};

/// Codegen strategy controls how a Linalg op is progressively lowered.
struct CodegenStrategy {
  /// Append a pattern to lower all vector operations.
  CodegenStrategy &vectorLowering(LinalgVectorLoweringOptions options) {
    transformationSequence.emplace_back(
        std::make_unique<VectorLowering>(options));
    return *this;
  }
  /// Configure the post staged-patterns global enabling passes options.
  CodegenStrategy &
  setVectorTransferToSCFOptions(LinalgEnablingOptions options) {
    linalgEnablingOptions = options;
    return *this;
  }

  /// Apply the transformation patterns in sequence with cleanup
  /// transformations interleaved.
  void configurePassPipeline(OpPassManager &pm, MLIRContext *context,
                             bool addEnablePass = true) const;

private:
  LogicalResult postPatternTransforms(Operation *func) const;

  LinalgEnablingOptions linalgEnablingOptions;
  SmallVector<std::unique_ptr<Transformation>, 4> transformationSequence;
};

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_CODEGENSTRATEGY_H_
