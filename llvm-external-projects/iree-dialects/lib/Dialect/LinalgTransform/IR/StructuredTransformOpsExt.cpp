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
// ErrorCheckingTrackingListener
//===----------------------------------------------------------------------===//

void ErrorCheckingTrackingListener::notifyPayloadReplacementNotFound(
    Operation *op, ValueRange values) {
  // Certain ops can dropped safely.
  if (isa<scf::ForOp>(op)) {
    LLVM_DEBUG(DBGS() << "Silently dropping scf.for op mapping\n");
    return;
  }

  hadErrors = true;
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
  errorStateChecked = false;
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
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
  auto payloadOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(payloadOps) ||
      !isa<ModuleOp>(*payloadOps.begin()))
    return emitSilenceableError() << "expected single module target";
  ModuleOp moduleOp = cast<ModuleOp>(*payloadOps.begin());

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

  if (failed(pm.run(moduleOp)))
    return DiagnosedSilenceableFailure::definiteFailure();

  //===------------------------------------------------------------------===//
  // END: Copied from upstream, this needs to be retired once we have a
  // proper upstream transform op.
  //===------------------------------------------------------------------===//

  // Make all arguments noalias for now.
  // FIXME: this is a terrible hack!
  moduleOp->walk([](LLVM::LLVMFuncOp funcOp) {
    for (int64_t i = 0; i < funcOp.getNumArguments(); ++i) {
      if (!funcOp.getFunctionType()
               .getParamType(i)
               .isa<LLVM::LLVMPointerType>())
        continue;
      funcOp.setArgAttr(i, "llvm.noalias", UnitAttr::get(funcOp.getContext()));
    }
  });

  result.set(getTransformed().cast<OpResult>(), payloadOps);
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
  if (handles.size() != 1 ||
      !llvm::hasSingleElement(state.getPayloadOps(handles[0]))) {
    return emitSilenceableFailure(loc)
           << "expected one handle to one operation";
  }
  Operation *root = *state.getPayloadOps(handles[0]).begin();

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
  if (handles.size() != 1 ||
      !llvm::hasSingleElement(state.getPayloadOps(handles[0]))) {
    return emitSilenceableFailure(loc)
           << "expected one handle to one operation";
  }
  Operation *root = *state.getPayloadOps(handles[0]).begin();

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
  if (handles.size() != 1 ||
      !llvm::hasSingleElement(state.getPayloadOps(handles[0]))) {
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
  Operation *root = *state.getPayloadOps(handles[0]).begin();

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

/// Match callback for a convolution with optional fill and trailing
/// elementwise operations. Matches *the first* occurrence of such a convolution
/// within an op associated with the given handle.
///
/// Input handles:
///
///   - container op, must be associated with one operation.
///
/// Output handles:
///
///   - the "fill" op preceding the convolution, if present;
///   - convolution op;
///   - trailing elementwise op, if any.
static DiagnosedSilenceableFailure
convolutionCallback(transform_ext::MatchCallbackResult &res, Location loc,
                    const mlir::transform::TransformState &state,
                    ValueRange handles) {
  if (handles.size() != 1 ||
      !llvm::hasSingleElement(state.getPayloadOps(handles[0]))) {
    return emitSilenceableFailure(loc)
           << "expected one handle to one operation";
  }

  transform_ext::StructuredOpMatcher *pattern, *fill, *trailing;
  transform_ext::MatchedConvolutionCaptures ignore;
  transform_ext::MatcherContext matcherContext;
  makeConvolutionMatcher(matcherContext, pattern, fill, trailing, ignore);

  // TODO: need a mechanism for this to go around the entire IR,
  // potentially with list matches for each group.
  Operation *root = *state.getPayloadOps(handles[0]).begin();

  WalkResult walkResult = root->walk([&](Operation *op) {
    pattern->resetCapture();
    if (!matchPattern(op, *pattern))
      return WalkResult::advance();

    // TODO: notify properly.
    LLVM_DEBUG({
      DBGS() << "fill:\n";
      if (fill->getCaptured())
        DBGS() << fill->getCaptured() << "\n";
      DBGS() << "pattern: " << pattern->getCaptured() << "\n";
      DBGS() << "trailing:\n";
      if (trailing->getCaptured())
        DBGS() << trailing->getCaptured() << "\n";
    });

    res.addPotentiallyEmptyPayloadGroup(fill->getCaptured());
    res.addPayloadGroup({pattern->getCaptured()});
    res.addPotentiallyEmptyPayloadGroup(trailing->getCaptured());
    return WalkResult::interrupt();
  });

  if (walkResult.wasInterrupted())
    return DiagnosedSilenceableFailure::success();
  return emitSilenceableFailure(loc) << "failed to match";
}

/// Match callback for a matmul with fill and optional trailing
/// elementwise operations. Matches *the first* occurrence of such a convolution
/// within an op associated with the given handle.
///
/// Input handles:
///
///   - container op, must be associated with one operation.
///
/// Output handles:
///
///   - the "fill" op preceding the convolution, if present;
///   - convolution op;
///   - trailing elementwise op, if any.
static DiagnosedSilenceableFailure
matmulCallback(transform_ext::MatchCallbackResult &res, Location loc,
               const mlir::transform::TransformState &state,
               ValueRange handles) {
  if (handles.size() != 1 ||
      !llvm::hasSingleElement(state.getPayloadOps(handles[0]))) {
    return emitSilenceableFailure(loc)
           << "expected one handle to one operation";
  }

  transform_ext::StructuredOpMatcher *pattern, *fill, *trailing;
  transform_ext::MatchedMatmulCaptures ignore;
  transform_ext::MatcherContext matcherContext;
  makeMatmulMatcher(matcherContext, pattern, fill, trailing, ignore);

  // TODO: need a mechanism for this to go around the entire IR,
  // potentially with list matches for each group.
  Operation *root = *state.getPayloadOps(handles[0]).begin();

  WalkResult walkResult = root->walk([&](Operation *op) {
    pattern->resetCapture();
    if (!matchPattern(op, *pattern))
      return WalkResult::advance();

    // TODO: notify properly.
    LLVM_DEBUG({
      DBGS() << "fill:\n";
      if (fill->getCaptured())
        DBGS() << fill->getCaptured() << "\n";
      DBGS() << "pattern: " << pattern->getCaptured() << "\n";
      DBGS() << "trailing:\n";
      if (trailing->getCaptured())
        DBGS() << trailing->getCaptured() << "\n";
    });

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
  registry.registerCallback("convolution", convolutionCallback);
  registry.registerCallback("matmul", matmulCallback);
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
    auto payloads = state.getPayloadOps(handle);
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
