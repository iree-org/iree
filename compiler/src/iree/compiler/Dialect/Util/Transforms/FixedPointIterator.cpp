// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::IREE::Util {
namespace {

// Dynamic pass which runs a sub-pipeline to a fixed point or a maximum
// iteration count.
//
// There is no direct coupling between this iterator and the contained passes.
// Indirectly, at the start of each iteration, this pass will set the
// "iree.fixedpoint.converged" unit attribute on the root operation. If it is
// still there when the sub-pipeline is complete, it will be removed and
// iteration terminates. If a sub-pass removes it, then iteration will
// continue.
class FixedPointIteratorPass
    : public FixedPointIteratorBase<FixedPointIteratorPass> {
public:
  FixedPointIteratorPass() = default;
  FixedPointIteratorPass(const FixedPointIteratorPass &other)
      : FixedPointIteratorBase<FixedPointIteratorPass>(other) {}
  FixedPointIteratorPass(OpPassManager pipeline);

private:
  LogicalResult initializeOptions(StringRef options) override;
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;

  std::optional<OpPassManager> pipeline;

  // Serialized form of the body pipeline.
  Option<std::string> pipelineStr{
      *this, "pipeline", llvm::cl::desc("Pipeline to run to a fixed point")};
  Option<int> maxIterations{*this, "max-iterations",
                            llvm::cl::desc("Maximum number of iterations"),
                            llvm::cl::init(10)};
};

FixedPointIteratorPass::FixedPointIteratorPass(OpPassManager pipeline)
    : pipeline(std::move(pipeline)) {
  llvm::raw_string_ostream ss(pipelineStr);
  this->pipeline->printAsTextualPipeline(ss);
  ss.flush();
}

LogicalResult FixedPointIteratorPass::initializeOptions(StringRef options) {
  if (failed(Pass::initializeOptions(options)))
    return failure();
  if (pipeline)
    return success();

  // Pipelines are expected to be of the form `<op-name>(<pipeline>)`.
  // TODO: This was lifted from the Inliner pass. We should provide a parse
  // entry point that is the direct inverse of printAsTextualPipeline() and
  // at least keep this internal to the upstream implementation.
  // See: https://github.com/llvm/llvm-project/issues/52813
  StringRef pipelineSr = pipelineStr;
  size_t pipelineStart = pipelineSr.find_first_of('(');
  if (pipelineStart == StringRef::npos || !pipelineSr.consume_back(")"))
    return failure();
  StringRef opName = pipelineSr.take_front(pipelineStart);
  OpPassManager pm(opName);
  if (failed(parsePassPipeline(pipelineSr.drop_front(1 + pipelineStart), pm)))
    return failure();
  pipeline = std::move(pm);
  return success();
}

void FixedPointIteratorPass::getDependentDialects(
    DialectRegistry &registry) const {
  pipeline->getDependentDialects(registry);
}

void FixedPointIteratorPass::runOnOperation() {
  MLIRContext *context = &getContext();
  StringAttr markerName = StringAttr::get(context, "iree.fixedpoint.iteration");
  StringAttr modifiedName =
      StringAttr::get(context, "iree.fixedpoint.modified");

  if (getOperation()->hasAttr(markerName)) {
    emitError(getOperation()->getLoc())
        << "nested fixed point pipelines not supported";
    return signalPassFailure();
  }

  for (int i = 0; i < maxIterations; ++i) {
    getOperation()->setAttr(markerName,
                            IntegerAttr::get(IndexType::get(context), i));
    getOperation()->removeAttr(modifiedName);
    if (failed(runPipeline(*pipeline, getOperation()))) {
      return signalPassFailure();
    }

    if (!getOperation()->hasAttr(modifiedName)) {
      // Normal exit.
      getOperation()->removeAttr(markerName);
      return;
    }
  }

  // Abnormal exit - iteration count exceeded.
  emitError(getOperation()->getLoc())
      << "maximum iteration count exceeded in fixed point pipeline";
  return signalPassFailure();
}

} // namespace

std::unique_ptr<OperationPass<void>>
createFixedPointIteratorPass(OpPassManager pipeline) {
  return std::make_unique<FixedPointIteratorPass>(std::move(pipeline));
}

} // namespace mlir::iree_compiler::IREE::Util
