// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Dialect/LinalgTransform/TransformInterpreterPassBase.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/TransformExtensions/FlowExtensions.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

/// Builds transform IR forming dispatch regions for reductions.
void buildReductionDispatch(ImplicitLocOpBuilder &builder, Value scopeH,
                            bool emitRemarkOnMatch = false) {
  auto pdlOperation = pdl::OperationType::get(builder.getContext());
  SmallVector<Type> matchedTypes(4, pdlOperation);
  auto matched = builder.create<transform_ext::MatchCallbackOp>(
      matchedTypes, "reduction_partial",
      transform::FailurePropagationMode::Suppress, scopeH);

  if (emitRemarkOnMatch) {
    builder.create<transform_ext::EmitRemarkOp>(
        matched->getResults().drop_back().back(), "dispatch matched reduction");
  }

  auto [firstH, restH] =
      buildSelectFirstNonEmpty(builder, matched->getResults().back(),
                               matched->getResults().drop_back().back());
  Value regionH = builder.create<transform_dialect::WrapInDispatchRegionOp>(
      pdlOperation, firstH);
  SmallVector<Value> handlesToMerge(matched->getResults().begin(),
                                    std::prev(matched->getResults().end(), 2));
  handlesToMerge.push_back(restH);
  Value mergedHandlesH = builder.create<transform::MergeHandlesOp>(
      handlesToMerge, /*deduplicate=*/false);
  regionH =
      builder.create<transform_dialect::MovePrecedingOpIntoDispatchRegionOp>(
          mergedHandlesH, regionH);
  builder.create<transform_dialect::RegionToWorkgroupsOp>(pdlOperation,
                                                          regionH);
}

/// Pass declaration.
/// Interpreter pass that applies transform dialect ops for dispatch region
/// formation. This needs to be its own pass because the registration mechanism
/// and ops available are different than for other interpreters.
struct DispatchWithTransformDialect
    : public transform::TransformInterpreterPassBase<
          DispatchWithTransformDialect, DispatchWithTransformDialectBase> {
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<mlir::iree_compiler::IREE::LinalgExt::IREELinalgExtDialect,
                    IREE::Flow::FlowDialect,
                    AffineDialect,
                    arith::ArithDialect,
                    linalg::LinalgDialect,
                    pdl::PDLDialect,
                    pdl_interp::PDLInterpDialect,
                    scf::SCFDialect,
                    tensor::TensorDialect,
                    transform::TransformDialect
    >();
    // clang-format on
  }

  DispatchWithTransformDialect(StringRef transformFileName = StringRef(),
                               StringRef debugPayloadRootTag = StringRef(),
                               StringRef debugTransformRootTag = StringRef(),
                               bool debugEmitRemarkOnMatch = false) {
    this->transformFileName = transformFileName.str();
    this->debugPayloadRootTag = debugPayloadRootTag.str();
    this->debugTransformRootTag = debugTransformRootTag.str();
    this->debugEmitRemarkOnMatch = debugEmitRemarkOnMatch;
  }
  DispatchWithTransformDialect(const DispatchWithTransformDialect &pass)
      : TransformInterpreterPassBase(pass) {
    this->transformFileName = pass.transformFileName;
    this->debugPayloadRootTag = pass.debugPayloadRootTag;
    this->debugTransformRootTag = pass.debugTransformRootTag;
    this->debugEmitRemarkOnMatch = pass.debugEmitRemarkOnMatch;
  }

  void getPayloadRoots(SmallVectorImpl<Operation *> &targets) {
    getOperation()->walk<WalkOrder::PreOrder>([&](linalg::LinalgOp linalgOp) {
      targets.push_back(linalgOp);
      return WalkResult::skip();
    });
  }

  OwningOpRef<ModuleOp> constructTransformModule(Location initialLoc) {
    auto m = ModuleOp::create(initialLoc);
    OpBuilder b(initialLoc->getContext());
    b.setInsertionPointToEnd(m.getBody());
    b.create<transform::SequenceOp>(
        initialLoc, TypeRange(), transform::FailurePropagationMode::Propagate,
        b.getType<transform::AnyOpType>(),
        [&](OpBuilder &b, Location loc, Value rootH) {
          ImplicitLocOpBuilder ib(loc, b);
          ib.create<transform_ext::RegisterMatchCallbacksOp>();

          // Matchers+dispatch builders for each case, ordered by priority.
          buildReductionDispatch(ib, rootH, debugEmitRemarkOnMatch);

          b.create<transform::YieldOp>(loc);
        });

    return m;
  }

 private:
  Statistic numDispatches{this, "number of dispatches",
                          "Number of Flow dispatches created"};
};
}  // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createDispatchWithTransformDialect(StringRef transformFileName,
                                   StringRef debugPayloadRootTag,
                                   StringRef debugTransformRootTag,
                                   bool debugEmitRemarkOnMatch) {
  return std::make_unique<DispatchWithTransformDialect>(
      transformFileName, debugPayloadRootTag, debugTransformRootTag,
      debugEmitRemarkOnMatch);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
