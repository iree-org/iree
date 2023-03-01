// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/ScopedTransform.h"
#include "iree-dialects/Transforms/ListenerCSE.h"
#include "iree-dialects/Transforms/TransformMatchers.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "transform-ops-ext"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using mlir::iree_compiler::IREE::LinalgExt::LinalgEnablingOptions;

//===----------------------------------------------------------------------===//
// Additional constraints for PDLMatchOp.
//===----------------------------------------------------------------------===//

/// Hook for PDL driver to check if an operation (`values[0]`) is directly
/// nested in a function with the name provided by an attribute
/// (`values[1]`).
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

/// Construct a IRMapping from `linalgOp` to `genericLinalgModelOp`.
/// Walk both ops and check whether all subops are the same.
static LogicalResult
haveIdenticalBodiesImpl(linalg::LinalgOp linalgOp,
                        linalg::LinalgOp genericLinalgModelOp) {
  IRMapping bvm;
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
  {
    OpOperandVector opInputs = linalgOp.getDpsInputOperands();
    OpOperandVector modelInputs = linalgModelOp.getDpsInputOperands();
    OpOperandVector opOutputs = linalgOp.getDpsInitOperands();
    OpOperandVector modelOutputs = linalgModelOp.getDpsInitOperands();
    auto notEqualFn = [](std::tuple<OpOperand *, OpOperand *> in) -> bool {
      return std::get<0>(in)->get() != std::get<1>(in)->get();
    };

    if (opInputs.size() != modelInputs.size() ||
        opOutputs.size() != modelOutputs.size() ||
        llvm::any_of(llvm::zip(opInputs, modelInputs), notEqualFn) ||
        llvm::any_of(llvm::zip(opOutputs, modelOutputs), notEqualFn) ||
        linalgOp.getIndexingMaps() != linalgModelOp.getIndexingMaps() ||
        linalgOp.getIteratorTypesArray() !=
            linalgModelOp.getIteratorTypesArray())
      return failure();
  }

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

mlir::transform_ext::StructuredTransformOpsExtension::
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
    // See through tensor casts and reshape ops.
    //
    // TODO: we may need some generalization (interfaces?) of this for other
    // operations, especially multi-operand ones to understand which of their
    // operands may be coming from a Linalg op. Or a completely different
    // mechanism of tracking op replacement at creation, or even different
    // patterns that identify the "main" result of a transformation.
    while (isa<tensor::CastOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp,
               tensor::InsertSliceOp>(value.getDefiningOp())) {
      value = llvm::TypeSwitch<Operation *, Value>(value.getDefiningOp())
                  .Case([](tensor::CastOp op) { return op.getSource(); })
                  .Case([](tensor::CollapseShapeOp op) { return op.getSrc(); })
                  .Case([](tensor::ExpandShapeOp op) { return op.getSrc(); })
                  .Case([](tensor::InsertSliceOp op) { return op.getSource(); })
                  .Default([](Operation *) {
                    llvm_unreachable("Wrong op type");
                    return Value();
                  });
    }

    if (auto currentSourceOp = value.getDefiningOp<linalg::LinalgOp>()) {
      if (!sourceOp || sourceOp == currentSourceOp) {
        sourceOp = currentSourceOp;
        continue;
      }

      LLVM_DEBUG(
          DBGS() << "different source linalg ops for replacing one op: \n"
                 << sourceOp << "\n"
                 << currentSourceOp << "\n");
      return nullptr;
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
      return nullptr;
    }

    LLVM_DEBUG(
        DBGS()
        << "could not find a source scf.for when replacing another scf.for\n");
    return nullptr;
  }
  return forOp;
}

/// Find the op that defines all values in the range.
static FailureOr<Operation *> findSingleOpDefiningAll(ValueRange range) {
  Operation *op = nullptr;
  for (Value value : range) {
    // Block arguments are just dropped.
    auto currentSourceOp = value.getDefiningOp();
    if (!currentSourceOp) {
      LLVM_DEBUG(DBGS() << "replacing tracked op with bbarg\n");
      continue;
    }

    if (!op || op == currentSourceOp) {
      op = currentSourceOp;
      continue;
    }

    LLVM_DEBUG(DBGS() << "different source op when replacing one op\n");
    return failure();
  }
  return op;
}

// Find a single op that defines all values in the range, optionally
// transitively through other operations in an op-specific way.
static FailureOr<Operation *> findSingleDefiningOp(Operation *replacedOp,
                                                   ValueRange range) {
  return llvm::TypeSwitch<Operation *, FailureOr<Operation *>>(replacedOp)
      .Case<linalg::LinalgOp>([&](linalg::LinalgOp) -> FailureOr<Operation *> {
        auto op = findSingleLinalgOpDefiningAll(range);
        if (!op)
          return failure();
        return op.getOperation();
      })
      .Case<scf::ForOp>([&](scf::ForOp) -> FailureOr<Operation *> {
        auto op = findSingleForOpDefiningAll(range);
        if (!op)
          return failure();
        return op.getOperation();
      })
      .Default([&](Operation *) -> FailureOr<Operation *> {
        return findSingleOpDefiningAll(range);
      });
}

void mlir::TrackingListener::notifyOperationReplaced(Operation *op,
                                                     ValueRange newValues) {
  // Bail out if in error state.
  if (hadErrors)
    return;

  // Exit early if the op is not tracked.
  SmallVector<Value> handles;
  if (failed(getTransformState().getHandlesForPayloadOp(op, handles)))
    return;

  FailureOr<Operation *> replacement = findSingleDefiningOp(op, newValues);
  if (failed(replacement)) {
    emitError(op) << "could not find replacement for tracked op";
    return;
  }

  if (*replacement == nullptr) {
    // TODO: Check if the handle is dead. Otherwise, the op should not be
    // dropped. This needs a change in the transform dialect interpreter.
    LLVM_DEBUG(DBGS() << "removing tracked @" << op << " : " << *op << "\n");
  } else {
    LLVM_DEBUG(DBGS() << "replacing tracked @" << op << " : " << *op << " with "
                      << **replacement << "\n");
  }
  mayFail(replacePayloadOp(op, *replacement));
}

void mlir::TrackingListener::notifyOperationRemoved(Operation *op) {
  // Bail out if in error state.
  if (hadErrors)
    return;

  // Exit early if the op is not tracked.
  SmallVector<Value> handles;
  if (failed(getTransformState().getHandlesForPayloadOp(op, handles)))
    return;

  LLVM_DEBUG(DBGS() << "removing tracked @" << op << " : " << *op << "\n");
  mayFail(replacePayloadOp(op, nullptr));
}

void mlir::TrackingListener::removeMappings(Operation *op) {
  // Bail if in error state.
  if (hadErrors)
    return;

  // Replacing the tracked op with null will stop the tracking.
  LLVM_DEBUG(DBGS() << "removing mappings @" << op << " : " << *op << "\n");
  mayFail(replacePayloadOp(op, nullptr));
}

//===----------------------------------------------------------------------===//
// CanonicalizedSequenceOp
//===----------------------------------------------------------------------===//

void ::transform_ext::CanonicalizedSequenceOp::build(
    OpBuilder &builder, OperationState &state,
    transform::FailurePropagationMode failurePropagationMode,
    ::transform_ext::CanonicalizedSequenceOp::BodyBuilderFn bodyBuilder) {
  assert(state.name.isRegistered() && "not registered!");
  assert(bodyBuilder && "requires a body builder");
  MLIRContext *ctx = builder.getContext();
  state.addAttribute(
      CanonicalizedSequenceOp::getFailurePropagationModeAttrName(state.name),
      transform::FailurePropagationModeAttr::get(ctx, failurePropagationMode));
  Region *bodyRegion = state.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  bodyBlock.addArgument(pdl::OperationType::get(ctx), state.location);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&bodyBlock);
  bodyBuilder(builder, state.location, bodyBlock.getArgument(0));
}

/// Run enabling transformations (LICM and its variants, single-iteration loop
/// removal, CSE) on the given function.
static LogicalResult performEnablerTransformations(
    func::FuncOp func, RewriterBase::Listener &listener,
    LinalgEnablingOptions options = LinalgEnablingOptions()) {
  MLIRContext *ctx = func->getContext();
  RewritePatternSet patterns(ctx);
  linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  GreedyRewriteConfig config;
  config.listener = &listener;
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config)))
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
    Operation *containerOp, RewriterBase::Listener &listener,
    LinalgEnablingOptions options = LinalgEnablingOptions()) {
  auto res = containerOp->walk([&](func::FuncOp func) {
    if (failed(performEnablerTransformations(func, listener, options)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(res.wasInterrupted());
}

/// Drop the association between payload operations and transform dialect
/// handles when it is no longer necessary in a canonicalized sequence.
/// Specifically, drop the association between payload operations and the
/// operand handles if all handles to them will not be used after the current
/// `transform`. Also drop the association between payload operations and result
/// handles if results are never read. Note that the operand part is specific to
/// sequence-like execution that is not guaranteed in the transform dialect in
/// general.
static void
forgetUnnecessaryHandles(transform::TransformState &state,
                         transform_ext::CanonicalizedSequenceOp sequence,
                         transform::TransformOpInterface transform) {
  auto *listener = state.getExtension<TrackingListener>();
  assert(transform->getParentOp() == sequence &&
         "only works for transform ops immediately nested in a canonicalized "
         "sequence");
  assert(listener && "expected tracking listener to be present");

  // Checks if the operation or its ancestor is before `transform` in its block
  // or is `transform` itself.
  auto userIsBefore = [&](Operation *user) {
    while (user && user->getParentOp() != sequence)
      user = user->getParentOp();
    if (!user)
      return false;
    return user->isBeforeInBlock(transform) || user == transform;
  };

  // Drop associations for operands that will not be read again. Ignore consumed
  // operands that have been deassociated already. Consider all handles to each
  // payload operation and only drop the association if all handles pointing to
  // the same operation will are not used after the current transform op. The
  // handle will be erased automatically after the last payload operation is
  // deassociated from it.
  llvm::SmallDenseSet<Operation *> seen;
  llvm::SmallDenseMap<Value, bool> handlesUsedAfterTransform;
  for (Value operand : transform->getOperands()) {
    if (transform::isHandleConsumed(operand, transform))
      continue;
    if (!operand.getType().isa<transform::TransformHandleTypeInterface>())
      continue;
    for (Operation *payload : state.getPayloadOps(operand)) {
      if (!payload || seen.contains(payload))
        continue;
      SmallVector<Value> allHandles;
      (void)state.getHandlesForPayloadOp(payload, allHandles);
      bool allHandlesUnused = llvm::all_of(allHandles, [&](Value handle) {
        if (!handlesUsedAfterTransform.count(handle)) {
          handlesUsedAfterTransform[handle] =
              !llvm::all_of(handle.getUsers(), userIsBefore);
        }
        return !handlesUsedAfterTransform[handle];
      });
      if (allHandlesUnused) {
        // Store for mapping removal, but not remove immediately because it
        // could invalidate other handles.
        seen.insert(payload);
      }
    }
  }

  // Drop associations for results that will never be read.
  for (Value result : transform->getResults()) {
    if (!result.getUses().empty())
      continue;
    if (!result.getType().isa<transform::TransformHandleTypeInterface>())
      continue;
    for (Operation *payload : state.getPayloadOps(result)) {
      if (!payload || seen.contains(payload))
        continue;
      // Store for mapping removal, but not remove immediately because it could
      // invalidate other handles.
      seen.insert(payload);
    }
  }

  // Remove everything in one bunch.
  for (Operation *payload : seen)
    listener->removeMappings(payload);
}

DiagnosedSilenceableFailure transform_ext::CanonicalizedSequenceOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {

  MLIRContext *ctx = getContext();
  RewritePatternSet patternList(ctx);
  for (Dialect *dialect : ctx->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patternList);
  for (RegisteredOperationName op : ctx->getRegisteredOperations())
    op.getCanonicalizationPatterns(patternList, ctx);
  FrozenRewritePatternSet patterns(std::move(patternList));

  transform::TransformState::RegionScope regionScope =
      state.make_region_scope(getBodyRegion());
  auto &listener = state.addExtension<::mlir::TrackingListener>();
  auto detachListener = llvm::make_scope_exit(
      [&] { state.removeExtension<::mlir::TrackingListener>(); });
  if (failed(mapBlockArguments(state)))
    return DiagnosedSilenceableFailure::definiteFailure();

  auto checkedListenerTransform =
      [&](function_ref<LogicalResult(Operation *, RewriterBase::Listener &)>
              transform) {
        SmallVector<Operation *> roots;
        if (Value root = getRoot())
          llvm::append_range(roots, state.getPayloadOps(root));
        else
          roots.push_back(state.getTopLevel());

        for (Operation *target : roots) {
          // Make sure we always check the error state, no boolean
          // short-circuting.
          if (failed(transform(target, listener))) {
            target->emitOpError("Transform application failed.");
            return failure();
          }
          if (failed(listener.checkErrorState())) {
            target->emitOpError("Listener failed.");
            return failure();
          }
        }
        return success();
      };

  auto performCSE = [](Operation *root, RewriterBase::Listener &listener) {
    LogicalResult result =
        eliminateCommonSubexpressions(root, /*domInfo=*/nullptr, &listener);
    LLVM_DEBUG(
        DBGS() << (succeeded(result) ? "successfully performed" : "failed")
               << " CSE\n");
    return result;
  };
  auto performEnabler = [](Operation *root, RewriterBase::Listener &listener) {
    LogicalResult result = performEnablerTransformations(root, listener);
    LLVM_DEBUG(
        DBGS() << (succeeded(result) ? "successfully performed" : "failed")
               << " enabling transformations\n");
    return result;
  };
  auto performCanonicalization = [&patterns](Operation *root,
                                             RewriterBase::Listener &listener) {
    GreedyRewriteConfig config;
    config.listener = &listener;
    // Manually gather list of ops because the other GreedyPatternRewriteDriver
    // overloads only accepts ops that are isolated from above.
    SmallVector<Operation *> ops;
    root->walk([&](Operation *op) {
      if (op != root)
        ops.push_back(op);
    });
    LogicalResult result = applyOpPatternsAndFold(ops, patterns, config);
    LLVM_DEBUG(
        DBGS() << (succeeded(result) ? "successfully performed" : "failed")
               << " canonicalization\n");
    return result;
  };

  LLVM_DEBUG(DBGS() << "begin canonicalizing sequence\n");
  if (failed(checkedListenerTransform(performCSE))) {
    return mlir::emitDefiniteFailure(
        *this, "Failed to performCSE beform transform sequence");
  }
  if (failed(checkedListenerTransform(performCanonicalization))) {
    return mlir::emitDefiniteFailure(
        *this, "Failed to performCanonicalization beform transform sequence");
  }

  // Apply the sequenced ops one by one.
  for (Operation &transform : getBodyBlock()->without_terminator()) {
    auto transformOp = cast<transform::TransformOpInterface>(transform);
    DiagnosedSilenceableFailure result = state.applyTransform(transformOp);
    if (result.isDefiniteFailure()) {
      LLVM_DEBUG(DBGS() << "failed: " << transform << "\n");
      return result;
    }
    if (result.isSilenceableFailure()) {
      LLVM_DEBUG(DBGS() << "failed silently: " << transform << "\n");
      if (getFailurePropagationMode() ==
          transform::FailurePropagationMode::Propagate)
        return result;
      (void)result.silence();
    }
    LLVM_DEBUG(DBGS() << "successfully performed: " << transform << "\n");

    // Canonicalization may replace payload operations associated with the
    // transform dialect handles. Post-canonicalize reassociation is fragile and
    // may fail. To make this less likely, drop any association that are no
    // longer necessary, i.e., if the operand is no longer used in the sequence
    // or elsewhere or if the result is never read.
    forgetUnnecessaryHandles(state, *this, transformOp);

    if (failed(checkedListenerTransform(performCSE))) {
      return mlir::emitDefiniteFailure(&transform,
                                       "Failed to performCSE after transform");
    }
    if (failed(checkedListenerTransform(performEnabler))) {
      return mlir::emitDefiniteFailure(
          &transform, "Failed to performEnabler after transform");
    }
    if (failed(checkedListenerTransform(performCanonicalization))) {
      return mlir::emitDefiniteFailure(
          &transform, "Failed to performCanonicalization after transform");
    }
  }

  // Forward the operation mapping for values yielded from the sequence to the
  // values produced by the sequence op.
  for (const auto &pair :
       llvm::zip(getBodyBlock()->getTerminator()->getOperands(),
                 getOperation()->getOpResults())) {
    Value terminatorOperand = std::get<0>(pair);
    OpResult result = std::get<1>(pair);
    results.set(result, state.getPayloadOps(terminatorOperand));
  }

  LLVM_DEBUG(DBGS() << "end canonicalizing sequence\n");
  return DiagnosedSilenceableFailure::success();
}

/// Returns `true` if the given op operand may be consuming the handle value in
/// the Transform IR. That is, if it may have a Free effect on it.
static bool isValueUsePotentialConsumer(OpOperand &use) {
  // Conservatively assume the effect being present in absence of the interface.
  auto memEffectInterface = dyn_cast<MemoryEffectOpInterface>(use.getOwner());
  if (!memEffectInterface)
    return true;

  SmallVector<MemoryEffects::EffectInstance, 2> effects;
  memEffectInterface.getEffectsOnValue(use.get(), effects);
  return llvm::any_of(effects, [](const MemoryEffects::EffectInstance &effect) {
    return isa<transform::TransformMappingResource>(effect.getResource()) &&
           isa<MemoryEffects::Free>(effect.getEffect());
  });
}

// TODO: Add declaration to TransformOps.h, then we do not have to duplicate
// this function.
static LogicalResult
checkDoubleConsume(Value value,
                   function_ref<InFlightDiagnostic()> reportError) {
  OpOperand *potentialConsumer = nullptr;
  for (OpOperand &use : value.getUses()) {
    if (!isValueUsePotentialConsumer(use))
      continue;

    if (!potentialConsumer) {
      potentialConsumer = &use;
      continue;
    }

    InFlightDiagnostic diag = reportError()
                              << " has more than one potential consumer";
    diag.attachNote(potentialConsumer->getOwner()->getLoc())
        << "used here as operand #" << potentialConsumer->getOperandNumber();
    diag.attachNote(use.getOwner()->getLoc())
        << "used here as operand #" << use.getOperandNumber();
    return diag;
  }

  return success();
}

LogicalResult transform_ext::CanonicalizedSequenceOp::verify() {
  // Check if the block argument has more than one consuming use.
  for (BlockArgument argument : getBodyBlock()->getArguments()) {
    auto report = [&]() {
      return (emitOpError() << "block argument #" << argument.getArgNumber());
    };
    if (failed(checkDoubleConsume(argument, report)))
      return failure();
  }

  // Check properties of the nested operations they cannot check themselves.
  for (Operation &child : *getBodyBlock()) {
    if (!isa<transform::TransformOpInterface>(child) &&
        &child != &getBodyBlock()->back()) {
      InFlightDiagnostic diag =
          emitOpError()
          << "expected children ops to implement TransformOpInterface";
      diag.attachNote(child.getLoc()) << "op without interface";
      return diag;
    }

    for (OpResult result : child.getResults()) {
      auto report = [&]() {
        return (child.emitError() << "result #" << result.getResultNumber());
      };
      if (failed(checkDoubleConsume(result, report)))
        return failure();
    }
  }

  if (getBodyBlock()->getTerminator()->getOperandTypes() !=
      getOperation()->getResultTypes()) {
    InFlightDiagnostic diag = emitOpError()
                              << "expects the types of the terminator operands "
                                 "to match the types of the result";
    diag.attachNote(getBodyBlock()->getTerminator()->getLoc()) << "terminator";
    return diag;
  }
  return success();
}

void transform_ext::CanonicalizedSequenceOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  auto *mappingResource = transform::TransformMappingResource::get();
  // Effects on root if present.
  if (getRoot())
    effects.emplace_back(MemoryEffects::Read::get(), getRoot(),
                         mappingResource);
  // Effects on results.
  for (Value result : getResults()) {
    effects.emplace_back(MemoryEffects::Allocate::get(), result,
                         mappingResource);
    effects.emplace_back(MemoryEffects::Write::get(), result, mappingResource);
  }

  for (Operation &op : *getBodyBlock()) {
    auto iface = dyn_cast<MemoryEffectOpInterface>(&op);
    if (!iface) {
      // TODO: fill all possible effects; or require ops to actually implement
      // the memory effect interface always
      assert(false);
    }
    if (getRoot()) {
      // Carry over all effects on the argument of the entry block as those on
      // the operand, this is the same value just remapped.
      SmallVector<MemoryEffects::EffectInstance, 2> nestedEffects;
      iface.getEffectsOnValue(getBodyBlock()->getArgument(0), nestedEffects);
      for (const auto &effect : nestedEffects)
        effects.emplace_back(effect.getEffect(), getRoot(),
                             effect.getResource());
    } else {
      // Otherwise, get all the effects.
      iface.getEffects(effects);
    }
  }
}

OperandRange transform_ext::CanonicalizedSequenceOp::getSuccessorEntryOperands(
    Optional<unsigned> index) {
  assert(index && index.value() == 0 && "unexpected region index");
  if (getOperation()->getNumOperands() == 1)
    return getOperation()->getOperands();
  return OperandRange(getOperation()->operand_end(),
                      getOperation()->operand_end());
}

void transform_ext::CanonicalizedSequenceOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  if (!index.has_value()) {
    Region *bodyRegion = &getBody();
    regions.emplace_back(bodyRegion, !operands.empty()
                                         ? bodyRegion->getArguments()
                                         : Block::BlockArgListType());
    return;
  }

  assert(*index == 0 && "unexpected region index");
  regions.emplace_back(getOperation()->getResults());
}

void transform_ext::CanonicalizedSequenceOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands, SmallVectorImpl<InvocationBounds> &bounds) {
  (void)operands;
  bounds.emplace_back(1, 1);
}

//===----------------------------------------------------------------------===//
// TODO: WILL MIGRATE
//===----------------------------------------------------------------------===//

using namespace mlir::linalg;

//===---------------------------------------------------------------------===//
// LowerToLLVMOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_ext::LowerToLLVMOp::apply(mlir::transform::TransformResults &result,
                                    mlir::transform::TransformState &state) {

  //===------------------------------------------------------------------===//
  // BEGIN: Copied from upstream, this needs to be retired once we have a
  // proper upstream transform op.
  //===------------------------------------------------------------------===//

  // TODO: it is feasible to scope lowering at arbitrary level and introduce
  // unrealized casts, but there needs to be the final module-wise cleanup in
  // the end. Keep module-level for now.
  PassManager pm(getContext());

  auto enableOpaquePointers = [](auto options) {
    options.useOpaquePointers = true;
    return options;
  };

  // Blanket-convert any remaining high-level vector ops to loops if any remain.
  pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());
  // Blanket-convert any remaining linalg ops to loops if any remain.
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  if (getEnableAsync()) {
    pm.addPass(createAsyncToAsyncRuntimePass());
    pm.addPass(createAsyncRuntimeRefCountingPass());
    pm.addPass(createAsyncRuntimeRefCountingOptPass());
  }
  pm.addPass(createCanonicalizerPass());
  // Blanket-convert any remaining affine ops if any remain.
  pm.addPass(createLowerAffinePass());
  // Convert SCF to CF (always needed).
  pm.addPass(createConvertSCFToCFPass());
  // Sprinkle some cleanups.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  // Blanket-convert any remaining linalg ops to LLVM if any remain.
  pm.addPass(createConvertLinalgToLLVMPass());
  {
    auto options = ConvertVectorToLLVMPassOptions();
    options.reassociateFPReductions = getReassociateFpReductions();
    options.force32BitVectorIndices = getEnableIndexOptimizations();
    options.armNeon = getEnableArmNeon();
    options.armSVE = getEnableArmSve();
    options.amx = getEnableAmx();
    options.x86Vector = getEnableX86vector();
    options.useOpaquePointers = true;
    pm.addPass(createConvertVectorToLLVMPass(options));
  }
  // Convert Math to LLVM (always needed).
  pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
  // Expand complicated MemRef operations before lowering them.
  pm.addPass(memref::createExpandStridedMetadataPass());
  // The expansion may create affine expressions. Get rid of them.
  pm.addPass(createLowerAffinePass());
  // Convert MemRef to LLVM (always needed).
  pm.addPass(createFinalizeMemRefToLLVMConversionPass(
      enableOpaquePointers(FinalizeMemRefToLLVMConversionPassOptions{})));
  if (getEnableAsync())
    pm.addPass(createConvertAsyncToLLVMPass());
  // Convert Func to LLVM (always needed).
  pm.addPass(createConvertFuncToLLVMPass(
      enableOpaquePointers(ConvertFuncToLLVMPassOptions{})));
  // Convert Index to LLVM (always needed).
  pm.addPass(createConvertIndexToLLVMPass());
  // Convert remaining unrealized_casts (always needed).
  pm.addPass(createReconcileUnrealizedCastsPass());

  if (failed(pm.run(state.getTopLevel())))
    return DiagnosedSilenceableFailure::definiteFailure();

  //===------------------------------------------------------------------===//
  // END: Copied from upstream, this needs to be retired once we have a
  // proper upstream transform op.
  //===------------------------------------------------------------------===//

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
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// MatchCallbackOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform_ext::MatchCallbackOp::apply(
    mlir::transform::TransformResults &results,
    mlir::transform::TransformState &state) {
  auto setEmptyResults = [&results, this] {
    for (OpResult value : getResults()) {
      results.set(value, {});
    }
  };
  auto errorOut = [this, &setEmptyResults] {
    setEmptyResults();
    return emitSilenceableError();
  };

  auto *registry = state.getExtension<transform_ext::MatchCallbacksRegistry>();
  if (!registry)
    return errorOut() << "match registry not available";

  const transform_ext::MatchCallbacksRegistry::MatchCallbackFn *callback =
      registry->get(getCallbackName());
  if (!callback) {
    return errorOut() << "callback '" << getCallbackName()
                      << "' not found in the registry";
  }

  MatchCallbackResult result;
  DiagnosedSilenceableFailure status =
      (*callback)(result, getLoc(), state, getInputs());
  if (!status.succeeded()) {
    setEmptyResults();
    if (status.isDefiniteFailure())
      return status;
    if (getFailurePropagationMode() ==
        mlir::transform::FailurePropagationMode::Propagate) {
      return emitSilenceableError() << "failed to match";
    } else {
      return DiagnosedSilenceableFailure::success();
    }
  }
  if (getNumResults() != result.getNumPayloadGroups()) {
    return errorOut()
           << "callback produced a different number of handles than expected ( "
           << result.getNumPayloadGroups() << " vs " << getNumResults() << " )";
  }

  for (OpResult value : getResults()) {
    results.set(value, result.getPayloadGroup(value.getResultNumber()));
  }
  return DiagnosedSilenceableFailure::success();
}

void transform_ext::MatchCallbackOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  mlir::transform::onlyReadsHandle(getInputs(), effects);
  mlir::transform::producesHandle(getOutputs(), effects);
  // TODO: it doesn't really modify the payload, we need a separate resource for
  // this mapping.
  mlir::transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// RegisterMatchCallbacksOp
//===---------------------------------------------------------------------===//

/// Match callback for "_test_match_callback" hook. Matches any payload
/// operations associated with operand handles unless they have the
/// "test.iree_transform_do_not_match" attribute, in which case produces a
/// silenceable failure.
static DiagnosedSilenceableFailure
testMatchCallbackCallback(transform_ext::MatchCallbackResult &res, Location loc,
                          const mlir::transform::TransformState &state,
                          ValueRange handles) {
  bool hadFailures = false;
  for (Value handle : handles) {
    if (llvm::any_of(state.getPayloadOps(handle), [](Operation *op) {
          return op->hasAttr("test.iree_transform_do_not_match");
        })) {
      res.addPayloadGroup(ArrayRef<Operation *>());
      hadFailures = true;
    } else {
      res.addPayloadGroup(state.getPayloadOps(handle));
    }
  }
  if (hadFailures)
    return emitSilenceableFailure(loc) << "failed to match";
  return DiagnosedSilenceableFailure::success();
}

static DiagnosedSilenceableFailure testRepeatedMatcherUseCallback(
    transform_ext::MatchCallbackResult &res, Location loc,
    const mlir::transform::TransformState &state, ValueRange handles) {
  if (handles.size() != 1 || state.getPayloadOps(handles[0]).size() != 1) {
    return emitSilenceableFailure(loc)
           << "expected one handle to one operation";
  }
  Operation *root = state.getPayloadOps(handles[0])[0];

  transform_ext::MatcherContext matcherContext;
  auto &operand = transform_ext::m_StructuredOp(matcherContext);
  auto &first = transform_ext::m_StructuredOp(matcherContext).input(0, operand);
  auto &second = transform_ext::m_StructuredOp(matcherContext)
                     .input(0, operand)
                     .input(1, first);

  WalkResult walkResult = root->walk([&](Operation *op) {
    second.resetCapture();
    if (!matchPattern(op, second))
      return WalkResult::advance();

    res.addPayloadGroup({first.getCaptured()});
    res.addPayloadGroup({second.getCaptured()});
    return WalkResult::interrupt();
  });

  if (walkResult.wasInterrupted())
    return DiagnosedSilenceableFailure::success();
  return emitSilenceableFailure(loc) << "failed to match";
}

static DiagnosedSilenceableFailure
testValueMatcherCallback(transform_ext::MatchCallbackResult &res, Location loc,
                         const mlir::transform::TransformState &state,
                         ValueRange handles) {
  if (handles.size() != 1 || state.getPayloadOps(handles[0]).size() != 1) {
    return emitSilenceableFailure(loc)
           << "expected one handle to one operation";
  }
  Operation *root = state.getPayloadOps(handles[0])[0];

  transform_ext::MatcherContext matcherContext;
  auto &operand = transform_ext::m_Value(matcherContext);
  auto &first = transform_ext::m_StructuredOp(matcherContext).input(0, operand);
  auto &second = transform_ext::m_StructuredOp(matcherContext)
                     .input(0, operand)
                     .input(1, first);

  WalkResult walkResult = root->walk([&](Operation *op) {
    second.resetCapture();
    if (!matchPattern(op, second))
      return WalkResult::advance();

    res.addPayloadGroup({first.getCaptured()});
    res.addPayloadGroup({second.getCaptured()});
    return WalkResult::interrupt();
  });

  if (walkResult.wasInterrupted())
    return DiagnosedSilenceableFailure::success();
  return emitSilenceableFailure(loc) << "failed to match";
}

/// Match callback for a reduction with optional leading and trailing
/// elementwise operations. Matches *the first* occurrence of such a reduction
/// within an op associated with the given handle.
///
/// Input handles:
///
///   - container op, must be associated with one operation.
///
/// Output handles:
///
///   - leading elementwise op, if any;
///   - the "fill" op preceding the reduction;
///   - reduction op;
///   - trailing elementwise op, if any.
static DiagnosedSilenceableFailure
reductionCallback(transform_ext::MatchCallbackResult &res, Location loc,
                  const mlir::transform::TransformState &state,
                  ValueRange handles) {
  if (handles.size() != 1 || state.getPayloadOps(handles[0]).size() != 1) {
    return emitSilenceableFailure(loc)
           << "expected one handle to one operation";
  }

  transform_ext::StructuredOpMatcher *pattern, *fill, *leading, *trailing;
  transform_ext::MatchedReductionCaptures ignore;
  transform_ext::MatcherContext matcherContext;
  makeReductionMatcher(matcherContext, pattern, fill, leading, trailing,
                       ignore);

  // TODO: need a mechanism for this to go around the entire IR,
  // potentially with list matches for each group.
  Operation *root = state.getPayloadOps(handles[0])[0];

  WalkResult walkResult = root->walk([&](Operation *op) {
    pattern->resetCapture();
    if (!matchPattern(op, *pattern))
      return WalkResult::advance();

    // TODO: notify properly.
    LLVM_DEBUG({
      DBGS() << "leading:\n";
      if (leading->getCaptured())
        DBGS() << leading->getCaptured() << "\n";
      DBGS() << "fill: " << fill->getCaptured() << "\n";
      DBGS() << "pattern: " << pattern->getCaptured() << "\n";
      DBGS() << "trailing:\n";
      if (trailing->getCaptured())
        DBGS() << trailing->getCaptured() << "\n";
    });

    res.addPotentiallyEmptyPayloadGroup(leading->getCaptured());
    res.addPayloadGroup({fill->getCaptured()});
    res.addPayloadGroup({pattern->getCaptured()});
    res.addPotentiallyEmptyPayloadGroup(trailing->getCaptured());
    return WalkResult::interrupt();
  });

  if (walkResult.wasInterrupted())
    return DiagnosedSilenceableFailure::success();
  return emitSilenceableFailure(loc) << "failed to match";
}

DiagnosedSilenceableFailure transform_ext::RegisterMatchCallbacksOp::apply(
    mlir::transform::TransformResults &results,
    mlir::transform::TransformState &state) {
  auto &registry = state.addExtension<transform_ext::MatchCallbacksRegistry>();
  registry.registerCallback("_test_match_callback", testMatchCallbackCallback);
  registry.registerCallback("_test_repeated_matcher_use_callback",
                            testRepeatedMatcherUseCallback);
  registry.registerCallback("_test_value_matcher_callback",
                            testValueMatcherCallback);
  registry.registerCallback("reduction", reductionCallback);
  return DiagnosedSilenceableFailure::success();
}

void transform_ext::RegisterMatchCallbacksOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // TODO: it doesn't really modify the payload, we need a separate resource for
  // this mapping.
  mlir::transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// TakeFirstOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_ext::TakeFirstOp::apply(mlir::transform::TransformResults &results,
                                  mlir::transform::TransformState &state) {
  SmallVector<Operation *> concatenated;
  bool found = false;
  for (Value handle : getInputs()) {
    ArrayRef<Operation *> payloads = state.getPayloadOps(handle);
    if (payloads.empty())
      continue;
    if (!found) {
      results.set(getFirst().cast<OpResult>(), payloads);
      found = true;
    } else {
      llvm::append_range(concatenated, payloads);
    }
  }

  if (!found)
    results.set(getFirst().cast<OpResult>(), {});
  results.set(getRest().cast<OpResult>(), concatenated);
  return DiagnosedSilenceableFailure::success();
}

void transform_ext::TakeFirstOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  mlir::transform::onlyReadsHandle(getInputs(), effects);
  mlir::transform::producesHandle(getFirst(), effects);
  mlir::transform::producesHandle(getRest(), effects);
}

//===---------------------------------------------------------------------===//
// EmitRemarkOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform_ext::EmitRemarkOp::applyToOne(
    Operation *target, mlir::transform::ApplyToEachResultList &results,
    mlir::transform::TransformState &state) {
  for (Operation *payload : state.getPayloadOps(getHandle())) {
    payload->emitRemark(getMessage());
  }
  return DiagnosedSilenceableFailure::success();
}

void transform_ext::EmitRemarkOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  mlir::transform::onlyReadsHandle(getHandle(), effects);
  mlir::transform::onlyReadsPayload(effects);
}
