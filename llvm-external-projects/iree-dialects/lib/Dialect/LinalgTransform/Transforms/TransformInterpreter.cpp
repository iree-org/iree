//===-- TransformInterpreter.cpp - Interpreter of Linalg transforms -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files implements and interpreter pass for Linalg Transform dialect. This
// pass reads a module consisting of transformable IR and transformation control
// IR and applies the latter to the former.
//
//===----------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree-dialects/Dialect/LinalgTransform/TrackingCSE.h"
#include "iree-dialects/Dialect/LinalgTransform/TrackingRewriteDriver.h"
#include "iree-dialects/Dialect/LinalgTransform/TransformOpInterface.h"
#include "iree-dialects/Dialect/LinalgTransform/TransformOpMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arithmetic/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/AffineInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ModuleBufferization.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "transform-interpreter"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::linalg;

//===----------------------------------------------------------------------===//
// Linalg Interpreter Driver
//===----------------------------------------------------------------------===//

/// Run enabling transformations (LICM and its variants, single-iteration loop
/// removal, CSE) on the given function.
static LogicalResult performEnablerTransformations(
    FuncOp func, RewriteListener &listener,
    linalg::LinalgEnablingOptions options = linalg::LinalgEnablingOptions()) {
  MLIRContext *ctx = func->getContext();
  RewritePatternSet patterns(ctx);
  linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  if (failed(applyPatternsTrackAndFoldGreedily(func, listener,
                                               std::move(patterns))))
    return failure();

  // This assumes LICM never removes operations so we don't need tracking.
  if (options.licm) {
    WalkResult result =
        func->walk([](LoopLikeOpInterface loopLike) -> WalkResult {
          return moveLoopInvariantCode(loopLike);
        });
    if (result.wasInterrupted()) return failure();
  }

  func.walk([](Operation *op) {
    (void)llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<AffineForOp, scf::ForOp>(
            [](auto loop) { return promoteIfSingleIteration(loop); })
        .Default([](Operation *) { return success(); });
  });

  if (options.hoistRedundantVectorTransfers)
    hoistRedundantVectorTransfers(func);
  if (options.hoistRedundantVectorTransfersOnTensor)
    hoistRedundantVectorTransfersOnTensor(func);

  return eliminateCommonSubexpressionsWithTrackedOps(func, listener);
}

/// Run enabling transformations on the given model while preserving the
/// operation tracking information.
static LogicalResult performEnablerTransformations(
    ModuleOp module, RewriteListener &listener,
    linalg::LinalgEnablingOptions options = linalg::LinalgEnablingOptions()) {
  for (auto func : module.getOps<FuncOp>()) {
    if (failed(performEnablerTransformations(func, listener, options)))
      return failure();
  }
  return success();
}

static LogicalResult executeTransform(Operation *operation,
                                      transform::TransformState &state) {
  auto iface = dyn_cast<transform::TransformOpInterface>(operation);
  if (!iface)
    return operation->emitError() << "unknown transformation operation";

  return state.applyTransform(iface);
}

/// Perform the transformation specified by the callback and unconditionally
/// check the error state of the listener. Return failure if either failed.
static LogicalResult checkedListenerTransform(
    function_ref<LogicalResult(TrackingListener &)> transform,
    TrackingListener &listener) {
  // Make sure we check the listener error state regardless of the transform
  // result.
  LogicalResult transformResult = transform(listener);
  LogicalResult listenerResult = listener.checkErrorState();
  return failure(failed(transformResult) || failed(listenerResult));
}

/// Applies the transformations listed in the `sequence` to operations starting
/// from `target`. The following transformations may be applied to operations
/// produced by previous transformations as indicated by SSA value flow in the
/// Linalg Transform dialect.
static LogicalResult executeSequence(linalg::transform::SequenceOp sequence,
                                     ModuleOp module) {
  MLIRContext *ctx = module->getContext();
  RewritePatternSet patternList(ctx);
  for (Dialect *dialect : ctx->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patternList);
  for (RegisteredOperationName op : ctx->getRegisteredOperations())
    op.getCanonicalizationPatterns(patternList, ctx);
  FrozenRewritePatternSet patterns(std::move(patternList));

  transform::TransformState state(module);
  TrackingListener &listener = state.addExtension<TrackingListener>();

  // Run the canonicalizations upfront so we don't match and transform
  // operations only to drop them later.
  if (failed(checkedListenerTransform(
          [&](TrackingListener &listener) {
            return eliminateCommonSubexpressionsWithTrackedOps(module,
                                                               listener);
          },
          listener))) {
    LLVM_DEBUG(DBGS() << "failed to perform CSE\n");
    return failure();
  }
  if (failed(checkedListenerTransform(
          [&](TrackingListener &listener) {
            return applyPatternsTrackAndFoldGreedily(module, listener,
                                                     patterns);
          },
          listener))) {
    LLVM_DEBUG(DBGS() << "failed to apply canonicalization patterns\n");
    return failure();
  }

  for (Operation &transform : sequence.body().front()) {
    if (failed(executeTransform(&transform, state))) {
      std::string str;
      llvm::raw_string_ostream ss(str);
      ss << "failed to apply: " << transform << "\nto module\n" << module;
      ss.flush();
      return transform.emitError() << str;
    }

    LLVM_DEBUG(DBGS() << "successfully applied transform: " << transform
                      << "\n");

    // Run CSE, enabling transformations and canonicalization. This is similar
    // to running the respective pass, but (a) keeps tracking the value/op
    // mapping and (b) avoids constructing the pattern set + pass pipeline on
    // every step.
    // TODO: consider better targeting than module-level transformations here:
    // e.g., the enabler internals can apply to one function only. Furthermore,
    // we don't need all of enabler transformations after/before all passes.
    if (failed(checkedListenerTransform(
            [&](TrackingListener &listener) {
              return eliminateCommonSubexpressionsWithTrackedOps(module,
                                                                 listener);
            },
            listener))) {
      LLVM_DEBUG(DBGS() << "failed to perform CSE\n");
      return failure();
    }

    // TODO: this runs CSE internally, mostly redundant with the above.
    if (failed(checkedListenerTransform(
            [&](TrackingListener &listener) {
              return performEnablerTransformations(module, listener);
            },
            listener))) {
      LLVM_DEBUG(DBGS() << "enabler transformations failed\n");
      return failure();
    }

    if (failed(checkedListenerTransform(
            [&](TrackingListener &listener) {
              return applyPatternsTrackAndFoldGreedily(module, listener,
                                                       patterns);
            },
            listener))) {
      LLVM_DEBUG(DBGS() << "failed to apply canonicalization patterns\n");
      return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Linalg Interpreter Pass
//===----------------------------------------------------------------------===//

namespace {
/// Pass that executes transformations specified by a module-level
/// iree_linalg_transform.apply operation on the same module.
struct InterpreterPass : public PassWrapper<InterpreterPass, Pass> {
  StringRef getArgument() const final { return "linalg-interp-transforms"; }

  StringRef getDescription() const final {
    return "Executes transformations specified in Linalg Transform dialect";
  }

  bool canScheduleOn(RegisteredOperationName opName) const override {
    return true;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<arith::ArithmeticDialect, 
                    AffineDialect,
                    bufferization::BufferizationDialect,
                    func::FuncDialect,
                    linalg::LinalgDialect, 
                    LLVM::LLVMDialect,
                    pdl::PDLDialect,
                    pdl_interp::PDLInterpDialect,
                    scf::SCFDialect, 
                    tensor::TensorDialect,
                    vector::VectorDialect
                    // clang-format on
                    >();

    linalg::comprehensive_bufferize::affine_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::comprehensive_bufferize::std_ext::
        registerModuleBufferizationExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    auto module = dyn_cast<ModuleOp>(getOperation());
    if (!module) return signalPassFailure();

    auto result = module.walk([&](linalg::transform::SequenceOp sequenceOp) {
      if (failed(executeSequence(sequenceOp, module)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) signalPassFailure();
  }
};

struct DropScheduleFromModulePass
    : public PassWrapper<DropScheduleFromModulePass, Pass> {
  StringRef getArgument() const final {
    return "linalg-drop-schedule-from-module";
  }

  StringRef getDescription() const final {
    return "Drop the schedule from the module";
  }

  bool canScheduleOn(RegisteredOperationName opName) const override {
    return true;
  }

  void runOnOperation() override {
    auto module = dyn_cast<ModuleOp>(getOperation());
    if (!module) return signalPassFailure();

    module.walk([&](Operation *nestedOp) {
      if (isa<linalg::transform::SequenceOp>(nestedOp) ||
          isa<pdl::PatternOp>(nestedOp))
        nestedOp->erase();
    });
  }
};
}  // namespace

namespace mlir {
/// Create a Linalg Transform interpreter pass.
std::unique_ptr<Pass> createLinalgTransformInterpreterPass() {
  return std::make_unique<InterpreterPass>();
}
/// Create a Linalg pass to drop the schedule from the module.
std::unique_ptr<Pass> createDropScheduleFromModulePass() {
  return std::make_unique<DropScheduleFromModulePass>();
}
}  // namespace mlir

/// Registration hook for the Linalg Transform interpreter pass.
void mlir::linalg::transform::registerLinalgTransformInterpreterPass() {
  PassRegistration<InterpreterPass>();
}

/// Registration hook for the Linalg drop schedule from module pass.
void mlir::linalg::transform::registerDropScheduleFromModulePass() {
  PassRegistration<DropScheduleFromModulePass>();
}
