// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"

#include "iree-dialects/Transforms/TransformMatchers.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/PDLExtension/PDLExtensionOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
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

/// Hook for PDL driver to check if an operation (`pdlValues[0]`) is directly
/// nested in a function with the name provided by an attribute
/// (`pdlValues[1]`).
/// TODO: PDL needs user-defined "questions".
static LogicalResult nestedInFunc(PatternRewriter &rewriter,
                                  PDLResultList &pdlResults,
                                  ArrayRef<PDLValue> pdlValues) {
  assert(pdlValues.size() == 2 && "expected 2 PDL values");
  Operation *operation = pdlValues[0].cast<Operation *>();
  Attribute attr = pdlValues[1].cast<Attribute>();

  auto func = operation->getParentOfType<mlir::FunctionOpInterface>();
  if (!func)
    return rewriter.notifyMatchFailure(operation, "not nested in a function");
  auto functionSymbol = dyn_cast<SymbolRefAttr>(attr);
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
    SmallVector<Value> opInputs = linalgOp.getDpsInputs();
    SmallVector<Value> modelInputs = linalgModelOp.getDpsInputs();
    ValueRange opOutputs = linalgOp.getDpsInits();
    ValueRange modelOutputs = linalgModelOp.getDpsInits();
    auto notEqualFn = [](std::tuple<Value, Value> in) -> bool {
      return std::get<0>(in) != std::get<1>(in);
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
        llvm::map_to_vector<4>(linalgOp.getBlock()->getArguments(),
                               [](Value v) { return v.getLoc(); }));
    ImplicitLocOpBuilder b(linalgModelOp.getLoc(), rewriter);
    auto regionBuilder = linalgModelOp.getRegionBuilder();
    llvm::ArrayRef<mlir::NamedAttribute> attrs = {};
    regionBuilder(b, *bodyBlock, attrs, /*emitError=*/{});
  }

  return haveEquivalentBodies(linalgOp, linalgModelOp, rewriter);
}

/// Check whether the unique Operation* stored in `pdlValues[0]` (assumed) is
/// equivalent to the unique StringRefAttr passed in `pdlValues[1]` (assumed).
/// Equivalence is achieved when either:
///   1. `pdlValues[0]` has the name stored in `pdlValues[1]`.
///   2. `pdlValues[0]` and `pdlValues[1]` are both linalg ops and their
///      structured interfaces as well as their bodies are equivalent.
///      Structured interfaces equivalence is a simple attribute level check.
///      Body equivalence is more involved and currently limited:
///        a. the current impl constructs an instance of the op whose name is
///           specified in `pdlValues[1]` and checks for exact body equality.
///        b. a more advanced version would "subtract" the bodies and fold, cse
///           and canonicalize to fixed point. If the result is "all zeros",
///           then the bodies would be equivalent (really isomorphic).
///   3. other cases TBD (e.g. vector.generic when available).
static LogicalResult isEquivalentToOp(PatternRewriter &rewriter,
                                      PDLResultList &pdlResults,
                                      ArrayRef<PDLValue> pdlValues) {
  assert(pdlValues.size() == 2 && "expected 2 PDL values");
  Operation *operation = pdlValues[0].cast<Operation *>();
  Attribute attribute = pdlValues[1].cast<Attribute>();

  auto modelOpNameAttr = dyn_cast<StringAttr>(attribute);
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
///   1. `pdlValues[0]` is an operands range
///   2. `pdlValues[1]` contains a DictAttr with `operand_number`, `dim` and
///      `divisor` IntegerAttr entries.
/// Succeed if `operands`[`operand_number`] is a ranked type whose `dim` is a
/// multiple of `divisor`.
/// Note: 0 is the convention to express "do not tile", it is considered to
/// divide everything.
static LogicalResult isDimMultipleOf(PatternRewriter &rewriter,
                                     PDLResultList &pdlResults,
                                     ArrayRef<PDLValue> pdlValues) {
  assert(pdlValues.size() == 2 && "expected 2 PDL values");
  ValueRange operands = pdlValues[0].cast<ValueRange>();
  Attribute attribute = pdlValues[1].cast<Attribute>();

  auto dict = dyn_cast<DictionaryAttr>(attribute);
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
    shapedType = dyn_cast<ShapedType>(operands[operandNumber].getType());
  if (!shapedType || shapedType.getRank() <= dim)
    return failure();
  return success(divisor == 0 || (shapedType.getShape()[dim] > 0 &&
                                  shapedType.getShape()[dim] % divisor == 0));
}

/// Assume that:
///   1. `pdlValues[0]` is an operands range
///   2. `pdlValues[1]` contains a DictAttr with `operand_number` and `dim`
///       IntegerAttr entries.
/// Succeed if `value`[`operand_number`] is a ranked type whose `dim` is
/// dynamic.
static LogicalResult isDimStatic(PatternRewriter &rewriter,
                                 PDLResultList &pdlResults,
                                 ArrayRef<PDLValue> pdlValues) {
  assert(pdlValues.size() == 2 && "expected 2 PDL values");
  ValueRange operands = pdlValues[0].cast<ValueRange>();
  Attribute attribute = pdlValues[1].cast<Attribute>();

  auto dict = dyn_cast<DictionaryAttr>(attribute);
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
    shapedType = dyn_cast<ShapedType>(operands[operandNumber].getType());
  return success(shapedType && !shapedType.isDynamicDim(dim));
}

/// Assume that:
///   1. `pdlValues[0]` is an operands range
///   2. `pdlValues[1]` contains a DictAttr with `operand_number` and `dim`
///       IntegerAttr entries.
/// Succeed if `value`[`operand_number`] is a ranked type whose `dim` is
/// dynamic.
static LogicalResult isDimDynamic(PatternRewriter &rewriter,
                                  PDLResultList &pdlResults,
                                  ArrayRef<PDLValue> pdlValues) {
  assert(pdlValues.size() == 2 && "expected 2 PDL values");
  ValueRange operands = pdlValues[0].cast<ValueRange>();
  Attribute attribute = pdlValues[1].cast<Attribute>();

  auto dict = dyn_cast<DictionaryAttr>(attribute);
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
    shapedType = dyn_cast<ShapedType>(operands[operandNumber].getType());
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

  addDialectDataInitializer<transform::PDLMatchHooks>(
      [&](transform::PDLMatchHooks &hooks) {
        llvm::StringMap<PDLConstraintFunction> constraints;
        constraints.try_emplace("nestedInFunc", nestedInFunc);
        constraints.try_emplace("isDimDynamic", isDimDynamic);
        constraints.try_emplace("isDimMultipleOf", isDimMultipleOf);
        constraints.try_emplace("isDimStatic", isDimStatic);
        constraints.try_emplace("isEquivalentToOp", isEquivalentToOp);
        hooks.mergeInPDLMatchHooks(std::move(constraints));
      });

  declareDependentDialect<bufferization::BufferizationDialect>();
  declareDependentDialect<vector::VectorDialect>();
}

#define GET_OP_CLASSES
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.cpp.inc"

//===----------------------------------------------------------------------===//
// ErrorCheckingTrackingListener
//===----------------------------------------------------------------------===//

void ErrorCheckingTrackingListener::notifyPayloadReplacementNotFound(
    Operation *op, ValueRange values, DiagnosedSilenceableFailure &&diag) {
  // Certain ops can dropped safely.
  if (isa<scf::ForOp>(op)) {
    LLVM_DEBUG(DBGS() << "Silently dropping scf.for op mapping\n");
    return;
  }

  SmallVector<Diagnostic> diags;
  diag.takeDiagnostics(diags);
  if (!status.succeeded())
    status.takeDiagnostics(diags);
  status = DiagnosedSilenceableFailure::silenceableFailure(std::move(diags));

  status = emitSilenceableFailure(
      getTransformOp(), "!!! tracking listener failed to find replacement op");
  status.attachNote(op->getLoc()) << "replaced op";
  for (Value v : values)
    status.attachNote(v.getLoc()) << "replacement value";
}

//===---------------------------------------------------------------------===//
// MatchCallbackOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform_ext::MatchCallbackOp::apply(
    mlir::transform::TransformRewriter &rewriter,
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
  mlir::transform::onlyReadsHandle(getInputsMutable(), effects);
  mlir::transform::producesHandle(getOutputs(), effects);
  // TODO: it doesn't really modify the payload, we need a separate resource for
  // this mapping.
  mlir::transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// Callbacks for tests driven by RegisterMatchCallbacksOp
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

static DiagnosedSilenceableFailure testShapedValueMatcherCallback(
    transform_ext::MatchCallbackResult &res, Location loc,
    const mlir::transform::TransformState &state, ValueRange handles) {
  if (handles.size() != 1 ||
      !llvm::hasSingleElement(state.getPayloadOps(handles[0]))) {
    return emitSilenceableFailure(loc)
           << "expected one handle to one operation";
  }
  Operation *root = *state.getPayloadOps(handles[0]).begin();

  int64_t rank;
  SmallVector<int64_t> dims;
  transform_ext::MatcherContext matcherContext;
  auto &value = transform_ext::m_ShapedValue(matcherContext);
  value.rank(transform_ext::CaptureRank(rank))
      .dim(transform_ext::AllDims(), transform_ext::CaptureDims(dims));
  auto &opMatcher =
      transform_ext::m_Operation<linalg::GenericOp>(matcherContext);
  opMatcher.result(0, value);

  WalkResult walkResult = root->walk([&](Operation *op) {
    opMatcher.resetCapture();
    if (!matchPattern(op, opMatcher))
      return WalkResult::advance();

    op->emitRemark() << "rank: " << rank;
    std::string message;
    llvm::raw_string_ostream os(message);
    llvm::interleaveComma(dims, os);
    os.flush();
    op->emitRemark() << "dimensions: " << message;

    res.addPayloadGroup({opMatcher.getCaptured()});
    return WalkResult::interrupt();
  });

  if (walkResult.wasInterrupted())
    return DiagnosedSilenceableFailure::success();
  return emitSilenceableFailure(loc) << "failed to match";
}

//===---------------------------------------------------------------------===//
// Callbacks for codegen driven by RegisterMatchCallbacksOp.
//===---------------------------------------------------------------------===//

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
  makeConvolutionMatcher(matcherContext, pattern, fill, trailing, ignore,
                         /*mustMatchEntireFunc=*/true);

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
                  ValueRange handles, bool mustMatchEntireFunc) {
  if (handles.size() != 1 ||
      !llvm::hasSingleElement(state.getPayloadOps(handles[0]))) {
    return emitSilenceableFailure(loc)
           << "expected one handle to one operation";
  }

  transform_ext::StructuredOpMatcher *pattern, *fill, *leading, *trailing;
  transform_ext::MatchedReductionCaptures ignore;
  transform_ext::MatcherContext matcherContext;
  makeReductionMatcher(matcherContext, pattern, fill, leading, trailing, ignore,
                       mustMatchEntireFunc);

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
  makeMatmulMatcher(matcherContext, pattern, fill, trailing, ignore,
                    /*mustMatchEntireFunc=*/true);

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

/// Match callback for linalg.batch_matmul and its linalg.generic equivalent fed
/// by a linalg.fill.
///
/// Input handles:
///
///   - the container op, must be associated with one operation.
///
/// Output handles:
///
///   - the fill op initializing the output;
///   - the main compute op.
static DiagnosedSilenceableFailure
batchMatmulCallback(transform_ext::MatchCallbackResult &res, Location loc,
                    const mlir::transform::TransformState &state,
                    ValueRange handles) {
  if (handles.size() != 1 ||
      !llvm::hasSingleElement(state.getPayloadOps(handles[0]))) {
    return emitSilenceableFailure(loc)
           << "expected one handle to one operation";
  }

  transform_ext::StructuredOpMatcher *pattern, *fill;
  transform_ext::MatchedMatmulCaptures ignore;
  transform_ext::MatcherContext matcherContext;
  transform_ext::makeBatchMatmulMatcher(matcherContext, pattern, fill, ignore,
                                        /*mustMatchEntireFunc*/ true);

  // TODO: need a mechanism for this to go around the entire IR,
  // potentially with list matches for each group.
  Operation *root = *state.getPayloadOps(handles[0]).begin();

  WalkResult walkResult = root->walk([&](Operation *op) {
    pattern->resetCapture();
    if (!matchPattern(op, *pattern))
      return WalkResult::advance();

    // TODO: notify properly
    LLVM_DEBUG({
      DBGS() << "fill:" << fill->getCaptured() << "\n";
      DBGS() << "pattern: " << pattern->getCaptured() << "\n";
    });

    res.addPayloadGroup({fill->getCaptured()});
    res.addPayloadGroup({pattern->getCaptured()});
    return WalkResult::interrupt();
  });

  if (walkResult.wasInterrupted())
    return DiagnosedSilenceableFailure::success();
  return emitSilenceableFailure(loc) << "failed to match batch matmul";
}

/// Match callback for a tensor.pad. Matches *the first* occurrence of such pad
/// within an op associated with the given handle.
///
/// Input handles:
///
///   - the container op, must be associated with one operation.
///
/// Output handles:
///
///   - the pad op.
static DiagnosedSilenceableFailure
padCallback(transform_ext::MatchCallbackResult &res, Location loc,
            const mlir::transform::TransformState &state, ValueRange handles,
            bool mustMatchEntireFunc) {
  if (handles.size() != 1 ||
      !llvm::hasSingleElement(state.getPayloadOps(handles[0]))) {
    return emitSilenceableFailure(loc)
           << "expected one handle to one operation";
  }

  transform_ext::CapturingOpMatcher *pattern;
  transform_ext::MatchedPadCaptures ignore;
  transform_ext::MatcherContext matcherContext;
  makePadMatcher(matcherContext, pattern, ignore, mustMatchEntireFunc);

  Operation *root = *state.getPayloadOps(handles[0]).begin();

  WalkResult walkResult = root->walk([&](Operation *op) {
    pattern->resetCapture();
    if (!matchPattern(op, *pattern))
      return WalkResult::advance();

    // TODO: notify properly.
    LLVM_DEBUG({
      DBGS() << "pad:\n";
      if (pattern->getCaptured())
        DBGS() << pattern->getCaptured() << "\n";
    });

    res.addPayloadGroup({pattern->getCaptured()});
    return WalkResult::interrupt();
  });

  if (walkResult.wasInterrupted())
    return DiagnosedSilenceableFailure::success();
  return emitSilenceableFailure(loc) << "failed to match";
}

//===---------------------------------------------------------------------===//
// RegisterMatchCallbacksOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform_ext::RegisterMatchCallbacksOp::apply(
    mlir::transform::TransformRewriter &rewriter,
    mlir::transform::TransformResults &results,
    mlir::transform::TransformState &state) {
  auto &registry = state.addExtension<transform_ext::MatchCallbacksRegistry>();
  registry.registerCallback("_test_match_callback", testMatchCallbackCallback);
  registry.registerCallback("_test_repeated_matcher_use_callback",
                            testRepeatedMatcherUseCallback);
  registry.registerCallback("_test_value_matcher_callback",
                            testValueMatcherCallback);
  registry.registerCallback("_test_shaped_value_matcher_callback",
                            testShapedValueMatcherCallback);
  registry.registerCallback("convolution", convolutionCallback);
  registry.registerCallback("matmul", matmulCallback);
  registry.registerCallback("batch_matmul", batchMatmulCallback);
  registry.registerCallback("pad", wrapAsEntireFuncMatch(padCallback));
  registry.registerCallback("reduction",
                            wrapAsEntireFuncMatch(reductionCallback));
  registry.registerCallback("reduction_partial",
                            wrapAsPartialMatch(reductionCallback));
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
transform_ext::TakeFirstOp::apply(mlir::transform::TransformRewriter &rewriter,
                                  mlir::transform::TransformResults &results,
                                  mlir::transform::TransformState &state) {
  SmallVector<Operation *> concatenated;
  bool found = false;
  for (Value handle : getInputs()) {
    auto payloads = state.getPayloadOps(handle);
    if (payloads.empty())
      continue;
    if (!found) {
      results.set(cast<OpResult>(getFirst()), payloads);
      found = true;
    } else {
      llvm::append_range(concatenated, payloads);
    }
  }

  if (!found)
    results.set(cast<OpResult>(getFirst()), {});
  results.set(cast<OpResult>(getRest()), concatenated);
  return DiagnosedSilenceableFailure::success();
}

void transform_ext::TakeFirstOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  mlir::transform::onlyReadsHandle(getInputsMutable(), effects);
  mlir::transform::producesHandle(getOperation()->getOpResults(), effects);
}

//===---------------------------------------------------------------------===//
// EmitRemarkOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform_ext::EmitRemarkOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    mlir::transform::ApplyToEachResultList &results,
    mlir::transform::TransformState &state) {
  for (Operation *payload : state.getPayloadOps(getHandle())) {
    payload->emitRemark(getMessage());
  }
  return DiagnosedSilenceableFailure::success();
}

void transform_ext::EmitRemarkOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  mlir::transform::onlyReadsHandle(getHandleMutable(), effects);
  mlir::transform::onlyReadsPayload(effects);
}
