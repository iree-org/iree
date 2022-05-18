// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/ScopedTransform.h"
#include "iree-dialects/Dialect/LinalgTransform/TrackingRewriteDriver.h"
#include "iree-dialects/Transforms/Listener.h"
#include "iree-dialects/Transforms/ListenerCSE.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "transform-ops-ext"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;

//===----------------------------------------------------------------------===//
// Additional constraints for PDLMatchOp.
//===----------------------------------------------------------------------===//

/// Hook for PDL driver to check if an operation (`values[0]`) is directly
/// nested in a function with the name provided by an attribute (`values[1]`).
/// TODO: PDL needs user-defined "questions".
static LogicalResult nestedInFunc(PatternRewriter &rewriter,
                                  Operation *operation, Attribute attr) {
  auto func = operation->getParentOfType<func::FuncOp>();
  if (!func)
    return rewriter.notifyMatchFailure(operation, "not nested in a function");
  auto functionSymbol = attr.dyn_cast<SymbolRefAttr>();
  if (!functionSymbol)
    return rewriter.notifyMatchFailure(operation, "not a function identifier");
  return success(functionSymbol.getLeafReference() == func.getName());
}

/// Construct a BlockAndValueMapping from `linalgOp` to `genericLinalgModelOp`.
/// Walk both ops and check whether all subops are the same.
static LogicalResult
haveIdenticalBodiesImpl(linalg::LinalgOp linalgOp,
                        linalg::LinalgOp genericLinalgModelOp) {
  BlockAndValueMapping bvm;
  bvm.map(linalgOp.getBlock()->getArguments(),
          genericLinalgModelOp.getBlock()->getArguments());
  SmallVector<Operation *> linalgBodyOps;
  linalgOp.getBlock()->walk(
      [&](Operation *op) { linalgBodyOps.push_back(op); });

  unsigned idx = 0;
  WalkResult res = genericLinalgModelOp.getBlock()->walk([&](Operation *op) {
    Operation *linalgSubOp = linalgBodyOps[idx++];
    if (op->getName() != linalgSubOp->getName())
      return WalkResult::interrupt();
    if (op->getAttrs() != linalgSubOp->getAttrs())
      return WalkResult::interrupt();
    for (auto it : llvm::zip(op->getOperands(), linalgSubOp->getOperands()))
      if (std::get<0>(it) != bvm.lookupOrNull(std::get<1>(it)))
        return WalkResult::interrupt();
    bvm.map(linalgSubOp->getResults(), op->getResults());
    return WalkResult::advance();
  });

  return success(!res.wasInterrupted());
}

/// Dispatch body equivalence check depending on case.
static LogicalResult haveEquivalentBodies(linalg::LinalgOp linalgOp,
                                          linalg::LinalgOp genericLinalgModelOp,
                                          PatternRewriter &rewriter) {
  if (succeeded(haveIdenticalBodiesImpl(linalgOp, genericLinalgModelOp)))
    return success();
  // TODO: haveEquivalentBodiesImpl, see e.g.
  // https://gist.github.com/nicolasvasilache/39e89e18c46e02335c16db6ec20a07e3
  return failure();
}

/// Succeed when `linalgOp` and `linalgModelOp` are deemed equivalent.
static LogicalResult isEquivalentToOpImpl(PatternRewriter &rewriter,
                                          linalg::LinalgOp linalgOp,
                                          linalg::LinalgOp linalgModelOp) {
  // If basic properties do not match, return failure.
  if (linalgOp.inputs() != linalgModelOp.inputs() ||
      linalgOp.outputs() != linalgModelOp.outputs() ||
      linalgOp.indexing_maps() != linalgModelOp.indexing_maps() ||
      linalgOp.iterator_types() != linalgModelOp.iterator_types())
    return failure();

  // Build the block and go perform a body comparison.
  {
    // createBlock moves the insertion point, scope it in an RAII block.
    OpBuilder::InsertionGuard guard(rewriter);
    Region &r = linalgModelOp->getRegion(0);
    Block *bodyBlock = rewriter.createBlock(
        &r, r.end(), linalgOp.getBlock()->getArgumentTypes(),
        llvm::to_vector<4>(
            llvm::map_range(linalgOp.getBlock()->getArguments(),
                            [](Value v) { return v.getLoc(); })));
    ImplicitLocOpBuilder b(linalgModelOp.getLoc(), rewriter);
    auto regionBuilder = linalgModelOp.getRegionBuilder();
    llvm::ArrayRef<mlir::NamedAttribute> attrs = {};
    regionBuilder(b, *bodyBlock, attrs);
  }

  return haveEquivalentBodies(linalgOp, linalgModelOp, rewriter);
}

/// Check whether the unique Operation* stored in `values[0]` (assumed) is
/// equivalent to the unique StringRefAttr passed in `values[1]` (assumed).
/// Equivalence is achieved when either:
///   1. `values[0]` has the name stored in `values[1]`.
///   2. `values[0]` and `values[1]` are both linalg ops and their structured
///      interfaces as well as their bodies are equivalent.
///      Structured interfaces equivalence is a simple attribute level check.
///      Body equivalence is more involved and currently limited:
///        a. the current impl constructs an instance of the op whose name is
///           specified in `values[1]` and checks for exact body equality.
///        b. a more advanced version would "subtract" the bodies and fold, cse
///           and canonicalize to fixed point. If the result is "all zeros",
///           then the bodies would be equivalent (really isomorphic).
///   3. other cases TBD (e.g. vector.generic when available).
static LogicalResult isEquivalentToOp(PatternRewriter &rewriter,
                                      Operation *operation,
                                      Attribute attribute) {
  auto modelOpNameAttr = attribute.dyn_cast<StringAttr>();
  if (!modelOpNameAttr)
    return failure(); // TODO: notifyMatchFailure needs an Operation* handle.
  auto modelOpName = modelOpNameAttr.strref();

  // 1. If op has name `modelOpName`, the match is trivial.
  if (operation->getName().getStringRef() == modelOpName)
    return success();

  // 2. Linalg vs Linalg.
  // Create op from `modelOpName`.
  OperationState modelOpState(
      operation->getLoc(), modelOpName, operation->getOperands(),
      operation->getResultTypes(), operation->getAttrs());
  modelOpState.addRegion();
  Operation *modelOp = rewriter.create(modelOpState);
  auto g1 = llvm::make_scope_exit([&]() { rewriter.eraseOp(modelOp); });
  linalg::LinalgOp linalgOp = dyn_cast<linalg::LinalgOp>(operation);
  linalg::LinalgOp linalgModelOp = dyn_cast<linalg::LinalgOp>(modelOp);
  if (linalgOp && linalgModelOp)
    return isEquivalentToOpImpl(rewriter, linalgOp, linalgModelOp);

  // 3. TBD
  return failure();
}

/// Assume that:
///   1. `values[0]` is an operands range
///   2. `values[1]` contains a DictAttr with `operand_number`, `dim` and
///      `divisor` IntegerAttr entries.
/// Succeed if `operands`[`operand_number`] is a ranked type whose `dim` is a
/// multiple of `divisor`.
/// Note: 0 is the convention to express "do not tile", it is considered to
/// divide everything.
static LogicalResult isDimMultipleOf(PatternRewriter &rewriter,
                                     ValueRange operands, Attribute attribute) {
  auto dict = attribute.dyn_cast<DictionaryAttr>();
  if (!dict)
    return failure(); // TODO: notifyMatchFailure needs an Operation* handle.

  int64_t dim;
  auto dimAttr = dict.getAs<IntegerAttr>("dim");
  if (!dimAttr)
    return failure(); // TODO: notifyMatchFailure needs an Operation* handle.
  dim = dimAttr.getInt();

  int64_t divisor;
  auto divisorAttr = dict.getAs<IntegerAttr>("divisor");
  if (!divisorAttr)
    return failure(); // TODO: notifyMatchFailure needs an Operation* handle.
  divisor = divisorAttr.getInt();

  int64_t operandNumber;
  auto operandNumberAttr = dict.getAs<IntegerAttr>("operand_number");
  if (!operandNumberAttr)
    return failure(); // TODO: notifyMatchFailure needs an Operation* handle.
  operandNumber = operandNumberAttr.getInt();

  ShapedType shapedType;
  if (static_cast<int64_t>(operands.size()) > operandNumber)
    shapedType = operands[operandNumber].getType().dyn_cast<ShapedType>();
  if (!shapedType || shapedType.getRank() <= dim)
    return failure();
  return success(divisor == 0 || (shapedType.getShape()[dim] > 0 &&
                                  shapedType.getShape()[dim] % divisor == 0));
}

/// Assume that:
///   1. `values[0]` is an operands range
///   2. `values[1]` contains a DictAttr with `operand_number` and `dim`
///       IntegerAttr entries.
/// Succeed if `value`[`operand_number`] is a ranked type whose `dim` is
/// dynamic.
static LogicalResult isDimStatic(PatternRewriter &rewriter, ValueRange operands,
                                 Attribute attribute) {
  auto dict = attribute.dyn_cast<DictionaryAttr>();
  if (!dict)
    return failure(); // TODO: notifyMatchFailure needs an Operation* handle.

  int64_t dim;
  auto dimAttr = dict.getAs<IntegerAttr>("dim");
  if (!dimAttr)
    return failure(); // TODO: notifyMatchFailure needs an Operation* handle.
  dim = dimAttr.getInt();

  int64_t operandNumber;
  auto operandNumberAttr = dict.getAs<IntegerAttr>("operand_number");
  if (!operandNumberAttr)
    return failure(); // TODO: notifyMatchFailure needs an Operation* handle.
  operandNumber = operandNumberAttr.getInt();

  ShapedType shapedType;
  if (static_cast<int64_t>(operands.size()) > operandNumber)
    shapedType = operands[operandNumber].getType().dyn_cast<ShapedType>();
  return success(shapedType && !shapedType.isDynamicDim(dim));
}

/// Assume that:
///   1. `values[0]` is an operands range
///   2. `values[1]` contains a DictAttr with `operand_number` and `dim`
///       IntegerAttr entries.
/// Succeed if `value`[`operand_number`] is a ranked type whose `dim` is
/// dynamic.
static LogicalResult isDimDynamic(PatternRewriter &rewriter,
                                  ValueRange operands, Attribute attribute) {
  auto dict = attribute.dyn_cast<DictionaryAttr>();
  if (!dict)
    return failure(); // TODO: notifyMatchFailure needs an Operation* handle.

  int64_t dim;
  auto dimAttr = dict.getAs<IntegerAttr>("dim");
  if (!dimAttr)
    return failure(); // TODO: notifyMatchFailure needs an Operation* handle.
  dim = dimAttr.getInt();

  int64_t operandNumber;
  auto operandNumberAttr = dict.getAs<IntegerAttr>("operand_number");
  if (!operandNumberAttr)
    return failure(); // TODO: notifyMatchFailure needs an Operation* handle.
  operandNumber = operandNumberAttr.getInt();

  ShapedType shapedType;
  if (static_cast<int64_t>(operands.size()) > operandNumber)
    shapedType = operands[operandNumber].getType().dyn_cast<ShapedType>();
  return success(shapedType && shapedType.isDynamicDim(dim));
}

//===----------------------------------------------------------------------===//
// StructuredTransformOpsExtension
//===----------------------------------------------------------------------===//

transform_ext::StructuredTransformOpsExtension::
    StructuredTransformOpsExtension() {
  registerTransformOps<
#define GET_OP_LIST
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.cpp.inc"
      >();

  registerPDLMatchConstraintFn("nestedInFunc", nestedInFunc);
  registerPDLMatchConstraintFn("isDimDynamic", isDimDynamic);
  registerPDLMatchConstraintFn("isDimMultipleOf", isDimMultipleOf);
  registerPDLMatchConstraintFn("isDimStatic", isDimStatic);
  registerPDLMatchConstraintFn("isEquivalentToOp", isEquivalentToOp);

  declareDependentDialect<bufferization::BufferizationDialect>();
  declareDependentDialect<vector::VectorDialect>();
}

#define GET_OP_CLASSES
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.cpp.inc"

//===----------------------------------------------------------------------===//
// TrackingListener
//===----------------------------------------------------------------------===//

/// Find the linalg op that defines all values in range, potentially
/// transitively through tensor casts.
static linalg::LinalgOp findSingleLinalgOpDefiningAll(ValueRange range) {
  linalg::LinalgOp sourceOp = nullptr;
  for (Value value : range) {
    // See through tensor casts.
    //
    // TODO: we may need some generalization (interfaces?) of this for other
    // operations, especially multi-operand ones to understand which of their
    // operands may be coming from a Linalg op. Or a completely different
    // mechanism of tracking op replacement at creation, or even different
    // patterns that identify the "main" result of a transformation.
    while (auto castOp = value.getDefiningOp<tensor::CastOp>())
      value = castOp.source();

    if (auto currentSourceOp = value.getDefiningOp<linalg::LinalgOp>()) {
      if (!sourceOp || sourceOp == currentSourceOp) {
        sourceOp = currentSourceOp;
        continue;
      }

      LLVM_DEBUG(
          DBGS() << "different source linalg ops for replacing one op: \n"
                 << sourceOp << "\n"
                 << currentSourceOp << "\n");
    }
    LLVM_DEBUG(DBGS() << "replacing linalg op with unknown non-linalg op:\n"
                      << *value.getDefiningOp() << "\n");
    return nullptr;
  }
  return sourceOp;
}

/// Find the scf "for" op that defines all values in the range.
static scf::ForOp findSingleForOpDefiningAll(ValueRange range) {
  scf::ForOp forOp = nullptr;
  for (Value value : range) {
    if (auto currentSourceOp = value.getDefiningOp<scf::ForOp>()) {
      if (!forOp || forOp == currentSourceOp) {
        forOp = currentSourceOp;
        continue;
      }
      LLVM_DEBUG(
          DBGS() << "different source scf.for ops when replacing one op\n");
    }

    LLVM_DEBUG(
        DBGS()
        << "could not find a source scf.for when replacing another scf.for\n");
    return nullptr;
  }
  return forOp;
}

// Find a single op that defines all values in the range, optionally
// transitively through other operations in an op-specific way.
static Operation *findSingleDefiningOp(Operation *replacedOp,
                                       ValueRange range) {
  return llvm::TypeSwitch<Operation *, Operation *>(replacedOp)
      .Case<linalg::LinalgOp>([&](linalg::LinalgOp) -> Operation * {
        return findSingleLinalgOpDefiningAll(range);
      })
      .Case<scf::ForOp>([&](scf::ForOp) -> Operation * {
        return findSingleForOpDefiningAll(range);
      })
      .Default([](Operation *) -> Operation * { return nullptr; });
}

namespace detail {
class TrackingListener : public RewriteListener,
                         public transform::TransformState::Extension {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TrackingListener);

  explicit TrackingListener(transform::TransformState &state)
      : transform::TransformState::Extension(state) {}

  ~TrackingListener() override {
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
    assert(errorStateChecked && "must check listener error state");
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
  }

  void notifyOperationReplaced(Operation *op, ValueRange newValues) override {
    // Bail out if in error state.
    if (hadErrors)
      return;

    // Exit early if the op is not tracked.
    Value handle = getTransformState().getHandleForPayloadOp(op);
    if (!handle)
      return;

    Operation *replacement = findSingleDefiningOp(op, newValues);
    if (!replacement) {
      emitError(op) << "could not find replacement for tracked op";
      return;
    }

    LLVM_DEBUG(DBGS() << "replacing tracked " << *op << " with " << *replacement
                      << " for " << handle << "\n");
    mayFail(replacePayloadOp(op, replacement));
  }

  void notifyOperationRemoved(Operation *op) override {
    // Bail out if in error state.
    if (hadErrors)
      return;

    // Exit early if the op is not tracked.
    Value handle = getTransformState().getHandleForPayloadOp(op);
    if (!handle)
      return;

    LLVM_DEBUG(DBGS() << "removing tracked " << *op << " for " << handle
                      << "\n");
    mayFail(replacePayloadOp(op, nullptr));
  }

  LogicalResult checkErrorState() const {
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
    errorStateChecked = true;
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
    return failure(hadErrors);
  }

private:
  InFlightDiagnostic emitError(Operation *op, const llvm::Twine &message = {}) {
    mayFail(failure());
    return op->emitError(message);
  }

  void mayFail(LogicalResult result) {
    hadErrors |= result.failed();
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
    errorStateChecked = false;
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
  }

  bool hadErrors = false;

#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
  mutable bool errorStateChecked = false;
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
};
} // namespace detail

//===----------------------------------------------------------------------===//
// CanonicalizedSequenceOp
//===----------------------------------------------------------------------===//

/// Run enabling transformations (LICM and its variants, single-iteration loop
/// removal, CSE) on the given function.
static LogicalResult performEnablerTransformations(
    func::FuncOp func, RewriteListener &listener,
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
    func->walk(
        [](LoopLikeOpInterface loopLike) { moveLoopInvariantCode(loopLike); });
  }

  func.walk([](Operation *op) {
    (void)llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<AffineForOp, scf::ForOp>(
            [](auto loop) { return promoteIfSingleIteration(loop); })
        .Default([](Operation *) { return success(); });
  });

  if (options.hoistRedundantVectorTransfers)
    linalg::hoistRedundantVectorTransfers(func);
  if (options.hoistRedundantVectorTransfersOnTensor)
    linalg::hoistRedundantVectorTransfersOnTensor(func);

  return eliminateCommonSubexpressions(func, /*domInfo=*/nullptr, &listener);
}

/// Run enabling transformations on the given `containerOp` while preserving the
/// operation tracking information.
static LogicalResult performEnablerTransformations(
    Operation *containerOp, RewriteListener &listener,
    linalg::LinalgEnablingOptions options = linalg::LinalgEnablingOptions()) {
  auto res = containerOp->walk([&](func::FuncOp func) {
    if (failed(performEnablerTransformations(func, listener, options)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(res.wasInterrupted());
}

LogicalResult transform_ext::CanonicalizedSequenceOp::apply(
    transform::TransformResults &transformResults,
    transform::TransformState &state) {

  MLIRContext *ctx = getContext();
  RewritePatternSet patternList(ctx);
  for (Dialect *dialect : ctx->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patternList);
  for (RegisteredOperationName op : ctx->getRegisteredOperations())
    op.getCanonicalizationPatterns(patternList, ctx);
  FrozenRewritePatternSet patterns(std::move(patternList));

  transform::TransformState::RegionScope regionScope =
      state.make_region_scope(getBodyRegion());
  auto &listener = state.addExtension<::detail::TrackingListener>();
  auto detachListener = llvm::make_scope_exit(
      [&] { state.removeExtension<::detail::TrackingListener>(); });
  if (failed(mapBlockArguments(state)))
    return failure();

  auto checkedListenerTransform =
      [&](function_ref<LogicalResult(Operation *, RewriteListener &)>
              transform) {
        SmallVector<Operation *> roots;
        if (Value target = getTarget())
          llvm::append_range(roots, state.getPayloadOps(target));
        else
          roots.push_back(state.getTopLevel());

        for (Operation *target : roots) {
          // Make sure we always check the error state, no boolean
          // short-circuting.
          LogicalResult result = transform(target, listener);
          LogicalResult listenerResult = listener.checkErrorState();
          if (failed(result) || failed(listenerResult))
            return failure();
        }
        return success();
      };

  auto performCSE = [](Operation *root, RewriteListener &listener) {
    LogicalResult result =
        eliminateCommonSubexpressions(root, /*domInfo=*/nullptr, &listener);
    LLVM_DEBUG(
        DBGS() << (succeeded(result) ? "successfully performed" : "failed")
               << " CSE\n");
    return result;
  };
  auto performEnabler = [](Operation *root, RewriteListener &listener) {
    LogicalResult result = performEnablerTransformations(root, listener);
    LLVM_DEBUG(
        DBGS() << (succeeded(result) ? "successfully performed" : "failed")
               << " enabling transformations\n");
    return result;
  };
  auto performCanonicalization = [&patterns](Operation *root,
                                             RewriteListener &listener) {
    LogicalResult result =
        applyPatternsTrackAndFoldGreedily(root, listener, patterns);
    LLVM_DEBUG(
        DBGS() << (succeeded(result) ? "successfully performed" : "failed")
               << " canonicalization\n");
    return result;
  };

  LLVM_DEBUG(DBGS() << "begin canonicalizing sequence\n");
  if (failed(checkedListenerTransform(performCSE)))
    return failure();
  if (failed(checkedListenerTransform(performCanonicalization)))
    return failure();

  for (Operation &transform : getBodyBlock()->without_terminator()) {
    if (failed(state.applyTransform(
            cast<transform::TransformOpInterface>(transform)))) {
      LLVM_DEBUG(DBGS() << "failed: " << transform << "\n");
      return failure();
    }
    LLVM_DEBUG(DBGS() << "successfully performed: " << transform << "\n");

    if (failed(checkedListenerTransform(performCSE)))
      return failure();
    if (failed(checkedListenerTransform(performEnabler)))
      return failure();
    if (failed(checkedListenerTransform(performCanonicalization)))
      return failure();
  }

  LLVM_DEBUG(DBGS() << "end canonicalizing sequence\n");
  return success();
}

void transform_ext::CanonicalizedSequenceOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  SmallVector<MemoryEffects::EffectInstance> childEffects;
  walk([&](Operation *child) {
    // Skip self to avoid infinite recursion.
    if (child == getOperation())
      return;

    auto iface = dyn_cast<MemoryEffectOpInterface>(child);
    if (!iface)
      return;

    childEffects.clear();
    iface.getEffects(childEffects);
    llvm::append_range(effects, childEffects);
  });
}

//===----------------------------------------------------------------------===//
// TODO: WILL MIGRATE
//===----------------------------------------------------------------------===//

using namespace mlir::linalg;

/// Extracts a vector of int64_t from an array attribute. Asserts if the
/// attribute contains values other than integers.
static SmallVector<int64_t> extractI64Array(ArrayAttr attr) {
  SmallVector<int64_t> result;
  result.reserve(attr.size());
  for (APInt value : attr.getAsValueRange<IntegerAttr>())
    result.push_back(value.getSExtValue());
  return result;
}

/// Extracts a vector of unsigned from an array attribute. Asserts if the
/// attribute contains values other than intergers. May truncate.
static SmallVector<unsigned> extractUIntArray(ArrayAttr attr) {
  SmallVector<unsigned> result;
  result.reserve(attr.size());
  for (APInt value : attr.getAsValueRange<IntegerAttr>())
    result.push_back(value.getZExtValue());
  return result;
}

/// Apply a tiling transformation to all payload ops and store both the
/// tiled operation as well as the created tile loops.
static LogicalResult
applyTilingToAll(Operation *transformOp, Value target,
                 ArrayRef<int64_t> tileSizes,
                 mlir::transform::TransformResults &transformResults,
                 mlir::transform::TransformState &state,
                 std::function<FailureOr<TiledLinalgOp>(LinalgOp)> applyFn) {
  size_t numLoops = tileSizes.size() - llvm::count(tileSizes, 0);

  SmallVector<Operation *> tiledLinalgOps;
  SmallVector<SmallVector<Operation *>> loopOps(numLoops);

  for (Operation *target : state.getPayloadOps(target)) {
    auto linalgOp = cast<linalg::LinalgOp>(target);
    FailureOr<TiledLinalgOp> tiled = applyFn(linalgOp);
    if (failed(tiled))
      return failure();

    tiledLinalgOps.push_back(tiled->op);
    if (tiled->loops.size() != numLoops)
      // Not enough loops were generated. This usually means that the input size
      // was smaller than the tiling size.
      // TODO: LinalgTilingPattern should return failure().
      return failure();
    for (unsigned int i = 0; i < numLoops; ++i) {
      loopOps[i].push_back(tiled->loops[i]);
    }
  }

  transformResults.set(transformOp->getOpResult(0), tiledLinalgOps);
  for (unsigned int i = 0; i < numLoops; ++i) {
    transformResults.set(transformOp->getOpResult(i + 1), loopOps[i]);
  }
  return success();
}

/// Parse a tiling operation that returns the tiled op as well as the created
/// tile loops. The function counts the non-zero tile sizes to compute the
/// number of results.
static ParseResult parseTileOp(OpAsmParser &parser, OperationState &result,
                               StringRef sizesAttrName) {
  OpAsmParser::UnresolvedOperand targetOperand;
  SMLoc opLoc;
  if (parser.getCurrentLocation(&opLoc))
    return failure();
  if (parser.parseOperand(targetOperand))
    return parser.emitError(opLoc, "expected `target` operand");
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  Attribute sizesAttr = result.attributes.get(sizesAttrName);
  if (!sizesAttr) {
    return parser.emitError(
        opLoc, llvm::formatv("expected `{0}` attribute", sizesAttrName));
  }
  auto sizesArrayAttr = sizesAttr.dyn_cast<ArrayAttr>();
  if (!sizesArrayAttr) {
    return parser.emitError(
        opLoc,
        llvm::formatv("`{0}` attribute must be an array", sizesAttrName));
  }
  Type pdlOpType = parser.getBuilder().getType<pdl::OperationType>();
  size_t numExpectedLoops =
      sizesArrayAttr.size() - llvm::count(extractI64Array(sizesArrayAttr), 0);
  result.addTypes(SmallVector<Type>(numExpectedLoops + 1, pdlOpType));
  if (parser.resolveOperand(targetOperand, pdlOpType, result.operands))
    return failure();
  return success();
}

namespace {
class SimpleRewriter : public PatternRewriter {
public:
  explicit SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};
} // namespace

//===----------------------------------------------------------------------===//
// ScalarizeOp
//===----------------------------------------------------------------------===//

FailureOr<LinalgOp> transform_ext::ScalarizeOp::applyToOne(LinalgOp target) {
  LinalgTilingOptions tilingOptions;
  tilingOptions.scalarizeDynamicDims();
  // Tiling with "scalarize_dyn_dims" actually sets the same lambda as the tile
  // sizes and asserts that it is not already set.
  SmallVector<int64_t> emptyTileSizes;
  LinalgTilingPattern pattern(getContext(), tilingOptions);
  SimpleRewriter rewriter(getContext());
  rewriter.setInsertionPoint(target);
  FailureOr<TiledLinalgOp> result =
      pattern.returningMatchAndRewrite(target, rewriter);
  if (failed(result))
    return failure();
  return result->op;
}

//===---------------------------------------------------------------------===//
// FuseOp
//===---------------------------------------------------------------------===//

LogicalResult transform_ext::FuseOp::apply(
    mlir::transform::TransformResults &transformResults,
    mlir::transform::TransformState &state) {
  LinalgTilingAndFusionOptions fusionOptions;
  fusionOptions.tileSizes = extractI64Array(getTileSizes());
  fusionOptions.tileInterchange = extractI64Array(getTileInterchange());

  return applyTilingToAll(
      getOperation(), getTarget(), fusionOptions.tileSizes, transformResults,
      state, [&](LinalgOp linalgOp) -> FailureOr<TiledLinalgOp> {
        LinalgTileAndFuseTensorOpsPattern pattern(getContext(), fusionOptions);
        SimpleRewriter rewriter(getContext());
        rewriter.setInsertionPoint(linalgOp);
        FailureOr<TileLoopNest> tileLoopNest =
            pattern.returningMatchAndRewrite(linalgOp, rewriter);
        if (failed(tileLoopNest))
          return failure();

        TiledLinalgOp tiledLinalgOp;
        tiledLinalgOp.op = tileLoopNest->getRootOp();
        tiledLinalgOp.loops = {tileLoopNest->getLoopOps().begin(),
                               tileLoopNest->getLoopOps().end()};
        return tiledLinalgOp;
      });
}

LogicalResult transform_ext::FuseOp::verify() {
  SmallVector<int64_t> permutation = extractI64Array(getTileInterchange());
  auto sequence = llvm::to_vector(llvm::seq<int64_t>(0, permutation.size()));
  if (!std::is_permutation(sequence.begin(), sequence.end(),
                           permutation.begin(), permutation.end())) {
    return emitOpError() << "expects interchange to be a permutation, found "
                         << getTileInterchange();
  }
  return success();
}

ParseResult transform_ext::FuseOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  return parseTileOp(parser, result, "tile_sizes");
}

void transform_ext::FuseOp::print(OpAsmPrinter &p) {
  p << ' ';
  p << getTarget();
  p.printOptionalAttrDict((*this)->getAttrs());
}

//===---------------------------------------------------------------------===//
// GeneralizeOp
//===---------------------------------------------------------------------===//

FailureOr<LinalgOp> transform_ext::GeneralizeOp::applyToOne(LinalgOp target) {
  // Exit early if no transformation is needed.
  if (isa<GenericOp>(target))
    return target;

  LinalgGeneralizationPattern pattern(getContext());
  SimpleRewriter rewriter(getContext());
  rewriter.setInsertionPoint(target);
  FailureOr<GenericOp> result =
      pattern.returningMatchAndRewrite(target, rewriter);
  if (failed(result))
    return failure();
  return cast<LinalgOp>(result->getOperation());
}

//===---------------------------------------------------------------------===//
// InterchangeOp
//===---------------------------------------------------------------------===//

FailureOr<LinalgOp> transform_ext::InterchangeOp::applyToOne(LinalgOp target) {
  SmallVector<unsigned> interchangeVector =
      extractUIntArray(getIteratorInterchange());
  // Exit early if no transformation is needed.
  if (interchangeVector.empty())
    return target;

  auto genericTarget = dyn_cast<GenericOp>(target.getOperation());
  if (!genericTarget) {
    InFlightDiagnostic diag = emitOpError()
                              << "applies to " << GenericOp::getOperationName()
                              << " ops";
    diag.attachNote(target.getLoc()) << "attempted to apply to this op";
    return diag;
  }

  GenericOpInterchangePattern pattern(getContext(), interchangeVector);
  SimpleRewriter rewriter(getContext());
  rewriter.setInsertionPoint(target);
  FailureOr<GenericOp> result =
      pattern.returningMatchAndRewrite(genericTarget, rewriter);
  if (failed(result))
    return failure();
  return cast<LinalgOp>(result->getOperation());
}

LogicalResult transform_ext::InterchangeOp::verify() {
  SmallVector<unsigned> permutation =
      extractUIntArray(getIteratorInterchange());
  auto sequence = llvm::to_vector(llvm::seq<unsigned>(0, permutation.size()));
  if (!std::is_permutation(sequence.begin(), sequence.end(),
                           permutation.begin(), permutation.end())) {
    return emitOpError()
           << "expects iterator_interchange to be a permutation, found "
           << getIteratorInterchange();
  }
  return success();
}

//===---------------------------------------------------------------------===//
// PadOp
//===---------------------------------------------------------------------===//

FailureOr<LinalgOp> transform_ext::PadOp::applyToOne(LinalgOp target) {
  // Convert the integer packing flags to booleans.
  SmallVector<bool> packPaddings;
  for (int64_t packPadding : extractI64Array(getPackPaddings()))
    packPaddings.push_back(static_cast<bool>(packPadding));

  // Convert the padding values to attributes.
  SmallVector<Attribute> paddingValues;
  for (auto const &it :
       llvm::zip(getPaddingValues(), target->getOperandTypes())) {
    Attribute attr = std::get<0>(it);
    Type elementType = getElementTypeOrSelf(std::get<1>(it));
    // Try to parse string attributes to obtain an attribute of element type.
    if (auto stringAttr = attr.dyn_cast<StringAttr>()) {
      paddingValues.push_back(
          parseAttribute(attr.cast<StringAttr>(), elementType));
      if (!paddingValues.back()) {
        return target->emitOpError("expects a padding value ")
               << std::get<0>(it) << " that parses to " << elementType;
      }
      continue;
    }
    // Otherwise, add the attribute directly.
    if (attr.getType() != elementType) {
      return target->emitOpError("expects a padding value ")
             << attr << " of type " << elementType;
    }
    paddingValues.push_back(attr);
  }

  // Extract the transpose vectors.
  SmallVector<SmallVector<int64_t>> transposePaddings;
  for (Attribute transposeVector : getTransposePaddings().cast<ArrayAttr>())
    transposePaddings.push_back(
        extractI64Array(transposeVector.cast<ArrayAttr>()));

  LinalgPaddingOptions paddingOptions;
  paddingOptions.setPaddingValues(paddingValues);
  paddingOptions.setPaddingDimensions(extractI64Array(getPaddingDimensions()));
  paddingOptions.setPackPaddings(packPaddings);
  paddingOptions.setHoistPaddings(extractI64Array(getHoistPaddings()));
  paddingOptions.setTransposePaddings(transposePaddings);

  LinalgPaddingPattern pattern(getContext(), paddingOptions);
  SimpleRewriter rewriter(getContext());
  rewriter.setInsertionPoint(target);
  return pattern.returningMatchAndRewrite(target, rewriter);
}

LogicalResult transform_ext::PadOp::verify() {
  SmallVector<int64_t> packPaddings = extractI64Array(getPackPaddings());
  if (any_of(packPaddings, [](int64_t packPadding) {
        return packPadding != 0 && packPadding != 1;
      })) {
    return emitOpError()
           << "expects pack_paddings to contain booleans (0/1), found "
           << getPackPaddings();
  }
  SmallVector<int64_t> paddingDimensions =
      extractI64Array(getPaddingDimensions());
  if (any_of(paddingDimensions,
             [](int64_t paddingDimension) { return paddingDimension < 0; })) {
    return emitOpError()
           << "expects padding_dimensions to contain positive integers, found "
           << getPaddingDimensions();
  }
  SmallVector<int64_t> hoistPaddings = extractI64Array(getHoistPaddings());
  if (any_of(hoistPaddings,
             [](int64_t hoistPadding) { return hoistPadding < 0; })) {
    return emitOpError()
           << "expects hoist_paddings to contain positive integers, found "
           << getHoistPaddings();
  }
  ArrayAttr transposes = getTransposePaddings();
  for (Attribute attr : transposes) {
    SmallVector<int64_t> transpose = extractFromI64ArrayAttr(attr);
    auto sequence = llvm::to_vector(llvm::seq<int64_t>(0, transpose.size()));
    if (!std::is_permutation(sequence.begin(), sequence.end(),
                             transpose.begin(), transpose.end())) {
      return emitOpError()
             << "expects transpose_paddings to be a permutation, found "
             << attr;
    }
  }
  return success();
}

//===---------------------------------------------------------------------===//
// BufferizeOp
//===---------------------------------------------------------------------===//

static void applyBufferizationEnablingTransformations(ModuleOp moduleOp) {
  RewritePatternSet patterns(moduleOp.getContext());
  patterns.add<GeneralizePadOpPattern>(moduleOp.getContext());
  (void)applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));
}

LogicalResult
transform_ext::BufferizeOp::apply(mlir::transform::TransformResults &result,
                                  mlir::transform::TransformState &state) {
  bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  options.memCpyFn = [](OpBuilder &builder, Location loc, Value from,
                        Value to) {
    return success(linalg::makeMemRefCopyOp(builder, loc, from, to));
  };

  auto moduleOp = cast<ModuleOp>(state.getTopLevel());
  applyBufferizationEnablingTransformations(moduleOp);
  if (failed(runOneShotModuleBufferize(moduleOp, options)))
    return failure();

  // Perform buffer-level hoistings.
  state.getTopLevel()->walk(
      [&](func::FuncOp funcOp) { hoistRedundantVectorTransfers(funcOp); });
  return success();
}

//===---------------------------------------------------------------------===//
// LowerToLLVMOp
//===---------------------------------------------------------------------===//

LogicalResult
transform_ext::LowerToLLVMOp::apply(mlir::transform::TransformResults &result,
                                    mlir::transform::TransformState &state) {
  // TODO: it is feasible to scope lowering at arbitrary level and introduce
  // unrealized casts, but there needs to be the final module-wise cleanup in
  // the end. Keep module-level for now.
  PassManager pm(getContext());

  pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  if (getEnableAsync()) {
    pm.addPass(createAsyncToAsyncRuntimePass());
    pm.addPass(createAsyncRuntimeRefCountingPass());
    pm.addPass(createAsyncRuntimeRefCountingOptPass());
  }
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createConvertLinalgToLLVMPass());
  pm.addPass(createConvertVectorToLLVMPass(
      // clang-format off
      LowerVectorToLLVMOptions()
        .enableReassociateFPReductions(getReassociateFpReductions())
        .enableIndexOptimizations(getEnableIndexOptimizations())
        .enableArmNeon(getEnableArmNeon())
        .enableArmSVE(getEnableArmSve())
        .enableAMX(getEnableAmx())
        .enableX86Vector(getEnableX86vector())));
  // clang-format on
  pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
  pm.addPass(createMemRefToLLVMPass());
  if (getEnableAsync())
    pm.addPass(createConvertAsyncToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  if (failed(pm.run(state.getTopLevel())))
    return failure();

  // Make all arguments noalias for now.
  // FIXME: this is a terrible hack!
  state.getTopLevel()->walk([](LLVM::LLVMFuncOp funcOp) {
    for (int64_t i = 0; i < funcOp.getNumArguments(); ++i) {
      if (!funcOp.getFunctionType()
               .getParamType(i)
               .isa<LLVM::LLVMPointerType>())
        continue;
      funcOp.setArgAttr(i, "llvm.noalias", UnitAttr::get(funcOp.getContext()));
    }
  });
  return success();
}

//===---------------------------------------------------------------------===//
// DecomposeOp
//===---------------------------------------------------------------------===//

LogicalResult
transform_ext::DecomposeOp::apply(mlir::transform::TransformResults &results,
                                  mlir::transform::TransformState &state) {
  RewritePatternSet patterns(getContext());
  // TODO: make this targetable.
  populateDecomposeConvolutionPatterns(patterns, LinalgTransformationFilter());
  if (failed(applyPatternsAndFoldGreedily(state.getTopLevel(),
                                          std::move(patterns))))
    return failure();

  // TODO: make this chainable, it isn't in the original codegenstrategy.
  return success();
}

//===---------------------------------------------------------------------===//
// VectorizeOp
//===---------------------------------------------------------------------===//

static void
configureVectorizationPatterns(transform_ext::VectorizeOp vectorizeOp,
                               RewritePatternSet &patterns) {
  MLIRContext *ctx = vectorizeOp->getContext();
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
  vector::populateVectorReductionToContractPatterns(patterns);
  patterns.add<linalg::LinalgCopyVTRForwardingPattern,
               linalg::LinalgCopyVTWForwardingPattern>(ctx,
                                                       /*benefit=*/2);
  vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
  vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
  if (vectorizeOp.getVectorizePadding())
    linalg::populatePadOpVectorizationPatterns(patterns);
}

/// Applies the transformation specified by the given vectorize operation to the
/// given target operation AND some related operations.Populates `results` with
/// transformation operations for further transformations if the pattern applied
/// successfully (currently, the main "contraction" op after vectorization).
static FailureOr<LinalgOp>
executeTargetedVectorizeOp(LinalgOp target,
                           transform_ext::VectorizeOp vectorizeOp) {
  // TODO: this is copy-pasta from LinalgStrategyVectorizePass, it shouldn't be.
  MLIRContext *ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  configureVectorizationPatterns(vectorizeOp, patterns);
  LinalgVectorizationPattern pattern(target.getContext());
  auto functionalVectorize = [&](LinalgOp op, PatternRewriter &rewriter) {
    return pattern.matchAndRewrite(op, rewriter);
  };

  /// Apply the transformations in a scope.
  return linalg::transform::scoped(
      target,
      [&](linalg::transform::ScopeOp scope,
          Operation *op) -> FailureOr<LinalgOp> {
        if (failed(functional::applyAt(op, functionalVectorize)) ||
            failed(applyPatternsAndFoldGreedily(scope, std::move(patterns))))
          return failure();
        // FIXME: Vectorization doesn't return anything.
        return LinalgOp();
      });

  // TODO: vectorization may fail because the op is not vectorizable, unclear
  // what to do here. We should probably report it somehow, but we may also
  // want to go on and keep the original for continuation. Should we have
  // some notion of transformation optionality vs. mandatory (like lowering)?
  // How to find ops that were not replaced?
}

LogicalResult
transform_ext::VectorizeOp::apply(mlir::transform::TransformResults &results,
                                  mlir::transform::TransformState &state) {
  if (getTarget()) {
    SmallVector<Operation *> resultVector;
    LogicalResult res = mlir::transform::detail::applyTransformToEach(
        state.getPayloadOps(getTarget()), resultVector, [&](LinalgOp target) {
          return executeTargetedVectorizeOp(target, *this);
        });

    if (failed(res))
      return emitError() << "failed to apply";

    results.set(getResult(0).cast<OpResult>(), resultVector);
    return success();
  }

  MLIRContext *ctx = getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<LinalgVectorizationPattern>(ctx);
  configureVectorizationPatterns(*this, patterns);
  auto *listener = state.getExtension<::detail::TrackingListener>();
  if (!listener)
    return emitError() << "expected TrackingListener extension to be available";
  LogicalResult applicationResult = applyPatternsTrackAndFoldGreedily(
      state.getTopLevel(), *listener, std::move(patterns));
  LogicalResult listenerResult = listener->checkErrorState();
  if (failed(applicationResult) || failed(listenerResult))
    return emitError() << "failed to apply";
  return success();
}

ParseResult transform_ext::VectorizeOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  auto operationType = pdl::OperationType::get(parser.getContext());
  OpAsmParser::UnresolvedOperand target;
  OptionalParseResult parseResult = parser.parseOptionalOperand(target);
  if (parseResult.hasValue()) {
    if (parseResult.getValue().failed() ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.resolveOperand(target, operationType, result.operands) ||
        parser.addTypeToList(operationType, result.types)) {
      return failure();
    }
  } else {
    if (parser.parseOptionalAttrDict(result.attributes)) {
      return failure();
    }
  }
  return success();
}

void transform_ext::VectorizeOp::print(OpAsmPrinter &printer) {
  if (getTarget())
    printer << " " << getTarget() << " ";
  printer.printOptionalAttrDict(getOperation()->getAttrs());
}

//===---------------------------------------------------------------------===//
// LowerVectorsOp
//===---------------------------------------------------------------------===//

/// Returns true of the numbered vector lowering stage is included into the list
/// of stages specified on the given lowerVectors operation.
static bool stageIncluded(int stage,
                          transform_ext::LowerVectorsOp lowerVectorsOp) {
  for (auto s : lowerVectorsOp.getStages().getAsValueRange<IntegerAttr>()) {
    if (s.getSExtValue() == stage)
      return true;
  }
  return false;
}

// Applies the transformation specified by the given lower vectors operation
/// to the given function.
LogicalResult
transform_ext::LowerVectorsOp::apply(mlir::transform::TransformResults &results,
                                     mlir::transform::TransformState &state) {
  MLIRContext *ctx = getContext();
  RewritePatternSet patterns(ctx);

  vector::VectorTransposeLowering vectorTransposeLowering =
      llvm::StringSwitch<vector::VectorTransposeLowering>(
          getTransposeLowering())
          .Case("eltwise", vector::VectorTransposeLowering::EltWise)
          .Case("flat_transpose", vector::VectorTransposeLowering::Flat)
          .Case("shuffle", vector::VectorTransposeLowering::Shuffle)
          .Default(vector::VectorTransposeLowering::EltWise);
  vector::VectorMultiReductionLowering vectorMultiReductionLowering =
      llvm::StringSwitch<vector::VectorMultiReductionLowering>(
          getMultireductionLowering())
          .Case("innerreduction",
                vector::VectorMultiReductionLowering::InnerReduction)
          .Default(vector::VectorMultiReductionLowering::InnerParallel);
  vector::VectorContractLowering vectorContractLowering =
      llvm::StringSwitch<vector::VectorContractLowering>(
          getContractionLowering())
          .Case("matrixintrinsics", vector::VectorContractLowering::Matmul)
          .Case("dot", vector::VectorContractLowering::Dot)
          .Case("outerproduct", vector::VectorContractLowering::OuterProduct)
          .Default(vector::VectorContractLowering::OuterProduct);
  // TODO: fix the annoying name mismatch (vector-transfers vs vector-transfer).
  vector::VectorTransferSplit vectorTransferSplit =
      llvm::StringSwitch<vector::VectorTransferSplit>(getSplitTransfers())
          .Case("none", vector::VectorTransferSplit::None)
          .Case("linalg-copy", vector::VectorTransferSplit::LinalgCopy)
          .Case("vector-transfers", vector::VectorTransferSplit::VectorTransfer)
          .Default(vector::VectorTransferSplit::None);

  vector::VectorTransformsOptions vectorTransformOptions;
  vectorTransformOptions.setVectorTransformsOptions(vectorContractLowering)
      .setVectorMultiReductionLowering(vectorMultiReductionLowering)
      .setVectorTransposeLowering(vectorTransposeLowering)
      .setVectorTransferSplit(vectorTransferSplit);

  VectorTransferToSCFOptions vectorTransferToSCFOptions =
      VectorTransferToSCFOptions()
          .enableFullUnroll(getUnrollVectorTransfers())
          .enableLowerPermutationMaps();

  int maxTransferRank = 1;

  auto avx2LoweringOptions =
      x86vector::avx2::LoweringOptions().setTransposeOptions(
          x86vector::avx2::TransposeLoweringOptions()
              .lower4x8xf32(getTransposeAvx2Lowering())
              .lower8x8xf32(getTransposeAvx2Lowering()));

  // TODO: this is copy-pasta from LinalgStrategyLowerVectorsPass, shouldn't be.
  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  if (stageIncluded(1, *this)) {
    patterns.add<mlir::vector::ContractionOpToOuterProductOpLowering,
                 mlir::vector::ContractionOpToMatmulOpLowering,
                 mlir::vector::ContractionOpLowering>(vectorTransformOptions,
                                                      ctx);
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
  }
  if (stageIncluded(2, *this)) {
    vector::populateVectorMultiReductionLoweringPatterns(
        patterns, vectorTransformOptions.vectorMultiReductionLowering);
  }
  if (stageIncluded(3, *this)) {
    patterns.add<vector::VectorTransferFullPartialRewriter>(
        ctx, vectorTransformOptions);
  }
  if (stageIncluded(4, *this)) {
    vector::populateVectorTransferLoweringPatterns(patterns, maxTransferRank);
  }
  if (stageIncluded(5, *this)) {
    populateVectorToSCFConversionPatterns(
        patterns, vectorTransferToSCFOptions.setTargetRank(maxTransferRank));
  }
  if (stageIncluded(6, *this)) {
    vector::populateVectorShapeCastLoweringPatterns(patterns);
  }
  if (stageIncluded(7, (*this))) {
    vector::populateVectorTransposeLoweringPatterns(patterns,
                                                    vectorTransformOptions);
    if (getTransposeAvx2Lowering())
      x86vector::avx2::populateSpecializedTransposeLoweringPatterns(
          patterns, avx2LoweringOptions, /*benefit=*/10);
  }

  // TODO: these transformations are currently not targeted at concrete ops.
  // LinalgTransformationFilter filter = makeTransformationFilter(target);
  if (failed(applyPatternsAndFoldGreedily(state.getTopLevel(),
                                          std::move(patterns))))
    return failure();

  // TODO: make composable...
  return success();
}

//===---------------------------------------------------------------------===//
// GetParentLoopOp
//===---------------------------------------------------------------------===//

FailureOr<scf::ForOp>
transform_ext::GetParentLoopOp::applyToOne(Operation *source) {
  int64_t nLoops = getNumLoops();
  for (int64_t i = 0; i < nLoops; ++i) {
    source = source->getParentOfType<scf::ForOp>();
    if (!source) {
      emitError() << "the transformed op is enclosed by " << i << " loops, but "
                  << nLoops << " expected";
      return failure();
    }
  }
  return cast<scf::ForOp>(source);
}

void transform_ext::GetParentLoopOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), getTarget(),
                       mlir::transform::TransformMappingResource::get());
  effects.emplace_back(MemoryEffects::Allocate::get(), getTransformed(),
                       mlir::transform::TransformMappingResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), getTransformed(),
                       mlir::transform::TransformMappingResource::get());
  effects.emplace_back(MemoryEffects::Read::get(),
                       mlir::transform::PayloadIRResource::get());
}

//===---------------------------------------------------------------------===//
// UnrollLoopOp
//===---------------------------------------------------------------------===//

LogicalResult transform_ext::UnrollLoopOp::applyToOne(scf::ForOp loop) {
  return loopUnrollByFactor(loop, getFactor());
}

//===---------------------------------------------------------------------===//
// PeelLoopOp
//===---------------------------------------------------------------------===//

FailureOr<scf::ForOp> transform_ext::PeelLoopOp::applyToOne(scf::ForOp loop) {
  scf::ForOp result;
  IRRewriter rewriter(loop->getContext());
  LogicalResult status =
      scf::peelAndCanonicalizeForLoop(rewriter, loop, result);
  if (failed(status))
    return loop;
  return result;
}

//===---------------------------------------------------------------------===//
// PipelineLoopOp
//===---------------------------------------------------------------------===//

static void
loopScheduling(scf::ForOp forOp,
               std::vector<std::pair<Operation *, unsigned>> &schedule,
               unsigned iterationInterval, unsigned readLatency) {
  auto getLatency = [&](Operation *op) {
    if (isa<vector::TransferReadOp>(op))
      return readLatency;
    return unsigned(1);
  };

  DenseMap<Operation *, unsigned> opCycles;
  std::map<unsigned, std::vector<Operation *>> wrappedSchedule;
  for (Operation &op : forOp.getBody()->getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue;
    unsigned earlyCycle = 0;
    for (Value operand : op.getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def)
        continue;
      earlyCycle = std::max(earlyCycle, opCycles[def] + getLatency(def));
    }
    opCycles[&op] = earlyCycle;
    wrappedSchedule[earlyCycle % iterationInterval].push_back(&op);
  }
  for (auto it : wrappedSchedule) {
    for (Operation *op : it.second) {
      unsigned cycle = opCycles[op];
      schedule.push_back(std::make_pair(op, cycle / iterationInterval));
    }
  }
}

FailureOr<scf::ForOp>
transform_ext::PipelineLoopOp::applyToOne(scf::ForOp loop) {
  // TODO: make the pipelining pattern return the transformed loop.
  if (!getOperation()->getUses().empty()) {
    InFlightDiagnostic diag = emitError()
                              << "NYI: cannot target the result of pipelining";
    diag.attachNote(getOperation()->use_begin()->getOwner()->getLoc())
        << "use here";
    return failure();
  }

  scf::PipeliningOption schedule;
  schedule.getScheduleFn =
      [this](scf::ForOp forOp,
             std::vector<std::pair<Operation *, unsigned>> &schedule) mutable {
        loopScheduling(forOp, schedule, getIterationInterval(),
                       getReadLatency());
      };

  RewritePatternSet patterns(loop->getContext());
  scf::populateSCFLoopPipeliningPatterns(patterns, schedule);
  assert(patterns.getNativePatterns().size() == 1 &&
         "expected one pipelining pattern");

  SimpleRewriter rewriter(getContext());
  rewriter.setInsertionPoint(loop);
  RewritePattern *pattern = patterns.getNativePatterns().front().get();
  if (failed(pattern->matchAndRewrite(loop, rewriter)))
    return failure();

  return scf::ForOp();
}

//===---------------------------------------------------------------------===//
// OutlineLoopOp
//===---------------------------------------------------------------------===//

static scf::ExecuteRegionOp outlineInExecuteRegion(RewriterBase &b,
                                                   Operation *op) {
  if (op->getNumRegions() != 1)
    return nullptr;
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  scf::ExecuteRegionOp executeRegionOp =
      b.create<scf::ExecuteRegionOp>(op->getLoc(), op->getResultTypes());
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&executeRegionOp.getRegion().emplaceBlock());
    Operation *clonedOp = b.cloneWithoutRegions(*op);
    Region &clonedRegion = clonedOp->getRegions().front();
    assert(clonedRegion.empty() && "expected empty region");
    b.inlineRegionBefore(op->getRegions().front(), clonedRegion,
                         clonedRegion.end());
    b.create<scf::YieldOp>(op->getLoc(), clonedOp->getResults());
  }
  b.replaceOp(op, executeRegionOp.getResults());
  return executeRegionOp;
}

static FailureOr<func::FuncOp>
outlineLoop(scf::ForOp loop, StringRef funcName,
            mlir::transform::TransformState &state, Location errorLoc) {
  PatternRewriterListener rewriter(loop->getContext());
  auto *listener = state.getExtension<::detail::TrackingListener>();
  if (!listener) {
    return emitError(errorLoc)
           << "expected TrackingListener extension to be present";
  }
  rewriter.addListener(listener);
  Location loc = loop.getLoc();
  scf::ExecuteRegionOp exec = outlineInExecuteRegion(rewriter, loop);
  assert(exec && "failed to produce execute_region");
  FailureOr<func::FuncOp> outlined =
      outlineSingleBlockRegion(rewriter, loc, exec.getRegion(), funcName);
  if (failed(listener->checkErrorState()))
    return failure();
  return outlined;
}

LogicalResult
transform_ext::OutlineLoopOp::apply(mlir::transform::TransformResults &results,
                                    mlir::transform::TransformState &state) {
  SmallVector<Operation *> resultVector;
  auto res = mlir::transform::detail::applyTransformToEach(
      state.getPayloadOps(getTarget()), resultVector,
      [&](scf::ForOp loop) -> FailureOr<func::FuncOp> {
        return outlineLoop(loop, getFuncName(), state, getLoc());
      });
  if (failed(res))
    return emitError() << "failed to apply";
  results.set(getResult().cast<OpResult>(), resultVector);
  return success();
}

//===---------------------------------------------------------------------===//
// PrintOp
//===---------------------------------------------------------------------===//

LogicalResult
transform_ext::PrintOp::apply(mlir::transform::TransformResults &results,
                              mlir::transform::TransformState &state) {
  if (!getTarget()) {
    llvm::outs() << "[[[ IR printer: " << getName() << " top-level ]]]\n";
    state.getTopLevel()->dump();
    return success();
  }

  llvm::outs() << "[[[ IR printer: " << getName() << " single op ]]]\n";
  ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());
  targets.front()->dump();
  return success();
}

void transform_ext::PrintOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), getTarget(),
                       mlir::transform::TransformMappingResource::get());
  effects.emplace_back(MemoryEffects::Read::get(),
                       mlir::transform::PayloadIRResource::get());

  // There is no resource for stdout file descriptor, so just declare print
  // writes into the default resource.
  effects.emplace_back(MemoryEffects::Write::get());
}
