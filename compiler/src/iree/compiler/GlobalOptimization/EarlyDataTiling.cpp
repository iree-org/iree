// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/DispatchCreation/MaterializedLayoutTarget.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

namespace mlir::iree_compiler::GlobalOptimization {
#define GEN_PASS_DEF_EARLYDATATILINGPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

static void buildEarlyDataTilingPipeline(
    OpPassManager &pipeline,
    const DispatchCreation::DataTilingEncodingOptions &encodingOptions) {
  MultiOpNest<func::FuncOp, IREE::Util::FuncOp, IREE::Util::InitializerOp>(
      pipeline)
      // Expose the same producer/consumer shapes used by dispatch formation.
      // In particular, unit-batch folding may have left a collapse between a
      // named convolution and its pointwise consumer.
      .addPass(
          []() { return DispatchCreation::createBubbleUpExpandShapesPass(); })
      .addPass(DispatchCreation::createCollapseContractionDimensionsPass)
      // Normalization itself can introduce expands after a contraction.
      .addPass(DispatchCreation::createSinkReshapesPass);
  DispatchCreation::buildDataTilingEncodingPassPipeline(pipeline,
                                                        encodingOptions);
  MultiOpNest<func::FuncOp, IREE::Util::FuncOp, IREE::Util::InitializerOp>(
      pipeline)
      .addPass(DispatchCreation::createPropagateDataTilingEncodingsPass);
  // Propagation can reify dynamic dimensions from encoded epilogue results.
  // Resolve them while their logical shapes are still available, before
  // materialization replaces them with padded physical shapes.
  pipeline.addPass(memref::createResolveRankedShapeTypeResultDimsPass());
  pipeline.addPass(createMaterializeHostEncodingPass());
  // Some physical encodings only reshape a tensor, such as a packed 1-D
  // broadcast bias. Resolve these while still in tensor IR so dispatch
  // creation never turns a metadata change into a separate copy dispatch.
  pipeline.addPass(createSimplifyPackUnpackPass());
}

namespace {
struct EarlyDataTilingPass final
    : impl::EarlyDataTilingPassBase<EarlyDataTilingPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    OpPassManager pipeline(ModuleOp::getOperationName());
    buildEarlyDataTilingPipeline(pipeline, getEncodingOptions());
    pipeline.getDependentDialects(registry);
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    // Layouts are materialized at most once, and only for the target they
    // were materialized for.
    if (DispatchCreation::getMaterializedLayoutTarget(moduleOp)) {
      if (failed(DispatchCreation::verifyMaterializedLayoutTarget(moduleOp))) {
        return signalPassFailure();
      }
      return;
    }

    // Only the default strategy materializes into packed layouts. Leave other
    // strategies, such as padding, to dispatch-time data tiling.
    if (encodingOption != DispatchCreation::EncodingOptions::Generic) {
      return;
    }

    // TODO: Materializing layouts here makes global optimization target aware
    // on purpose. The layout decision, however, is split between this pass,
    // isHostEncodingMaterializationSupported and the dispatch creation passes
    // that read the materialized layout target, it only supports modules with
    // a single CPU target, and it makes GlobalOptimization depend on HAL and
    // Codegen. Move the layout decision into its own component that decides
    // per affinity, so that heterogeneous modules and other backends can take
    // this route.
    IREE::HAL::DeviceAnalysis deviceAnalysis(moduleOp);
    if (failed(deviceAnalysis.run())) {
      return signalPassFailure();
    }
    SetVector<IREE::HAL::ExecutableTargetAttr> targets;
    deviceAnalysis.gatherAllExecutableTargets(targets);
    // Other modules keep the entire late layout route. In particular, do not
    // assign encodings and then erase them via identity materialization for a
    // heterogeneous module.
    if (targets.size() != 1 ||
        !isHostEncodingMaterializationSupported(targets.front())) {
      return;
    }

    OpPassManager pipeline(ModuleOp::getOperationName());
    buildEarlyDataTilingPipeline(pipeline, getEncodingOptions());
    if (failed(runPipeline(pipeline, moduleOp))) {
      return signalPassFailure();
    }
    // Record the target even if no op was data-tiled: the module now takes the
    // early route, so dispatch creation must not assign encodings later.
    DispatchCreation::setMaterializedLayoutTarget(moduleOp, targets.front());
  }

private:
  DispatchCreation::DataTilingEncodingOptions getEncodingOptions() const {
    return {llvm::to_vector(opTypes), encodingOption};
  }
};
} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
