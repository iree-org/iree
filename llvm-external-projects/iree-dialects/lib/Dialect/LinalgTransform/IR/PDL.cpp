//===-- PDL.cpp - Interoperability with PDL -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PDL.h"

#include "Transforms/Functional.h"
#include "llvm/ADT/ScopeExit.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Rewrite/PatternApplicator.h"

namespace mlir {
namespace linalg {

/// Return ops that match any of the patterns.
static SmallVector<Operation *> getMatchingOps(
    Operation *parent, const FrozenRewritePatternSet &patterns) {
  PatternApplicator applicator(patterns);
  applicator.applyDefaultCostModel();

  // TODO: The C++ functional API needs better interoperability with PDL.
  return functional::applyForEachIn(
      parent,
      [&](Operation *op, PatternRewriter &rewriter) -> FailureOr<Operation *> {
        if (succeeded(applicator.matchAndRewrite(op, rewriter))) return op;
        return failure();
      });
}

/// Hook for PDL driver to check if an operation (`value`) is directly nested in
/// a function with the name provided as constant parameter.
/// TODO: PDL needs user-defined "questions".
static LogicalResult nestedInFunc(PDLValue value, ArrayAttr constantParams,
                                  PatternRewriter &rewriter) {
  auto *operation = value.cast<Operation *>();
  auto func = operation->getParentOfType<FuncOp>();
  assert(constantParams.size() == 1 &&
         "expected a constant param with function name");
  auto functionSymbol = constantParams[0].dyn_cast<SymbolRefAttr>();
  assert(functionSymbol && "expected a function name");

  if (!func)
    return rewriter.notifyMatchFailure(operation, "not nested in a function");
  return success(functionSymbol.getLeafReference() == func.getName());
}

/// PDL rewrite hook that does nothing.
static void noOpRewriter(ArrayRef<PDLValue> args, ArrayAttr constantParams,
                         PatternRewriter &rewriter, PDLResultList &results) {
  assert(args.size() == 1 && "expected one argument");
#ifndef NDEBUG
  args.front().cast<Operation *>()->setAttr("iree_linalg_transform.matched",
                                            rewriter.getUnitAttr());
#endif
}

/// Construct a BlockAndValueMapping from `linalgOp` to `genericLinalgModelOp`.
/// Walk both ops and check whether all subops are the same.
static LogicalResult haveIdenticalBodiesImpl(LinalgOp linalgOp,
                                             LinalgOp genericLinalgModelOp) {
  BlockAndValueMapping bvm;
  bvm.map(linalgOp.getBlock()->getArguments(),
          genericLinalgModelOp.getBlock()->getArguments());
  SmallVector<Operation *> linalgBodyOps;
  linalgOp.getBlock()->walk(
      [&](Operation *op) { linalgBodyOps.push_back(op); });

  unsigned idx = 0;
  WalkResult res = genericLinalgModelOp.getBlock()->walk([&](Operation *op) {
    Operation *linalgSubOp = linalgBodyOps[idx++];
    if (op->getName() != linalgSubOp->getName()) return WalkResult::interrupt();
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
static LogicalResult haveEquivalentBodies(LinalgOp linalgOp,
                                          LinalgOp genericLinalgModelOp,
                                          PatternRewriter &rewriter) {
  if (succeeded(haveIdenticalBodiesImpl(linalgOp, genericLinalgModelOp)))
    return success();
  // TODO: haveEquivalentBodiesImpl, see e.g.
  // https://gist.github.com/nicolasvasilache/39e89e18c46e02335c16db6ec20a07e3
  return failure();
}

/// Succeed when `linalgOp` and `linalgModelOp` are deemed equivalent.
static LogicalResult isEquivalentToOpImpl(LinalgOp linalgOp,
                                          LinalgOp linalgModelOp,
                                          PatternRewriter &rewriter) {
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

/// Check whether the unique Operation* `op` stored in `value` (assumed) is
/// equivalent to the unique StringRefAttr operation name passed in
/// `constantParams`.
/// Equivalence is achieved when either:
///   1. `op` has the name stored in `constantParams`.
///   2. `op` and `constantParams` are both linalg ops and their structured
///      interfaces as well as their bodies are equivalent.
///      Structured interfaces equivalence is a simple attribute level check.
///      Body equivalence is more involved and currently limited:
///        a. the current impl constructs an instance of the op whose name is
///           specified in `constantParams` and checks for exact body equality.
///        b. a more advanced version would "subtract" the bodies and fold, cse
///           and canonicalize to fixed point. If the result is "all zeros",
///           then the bodies would be equivalent (really isomorphic).
///   3. other cases TBD (e.g. vector.generic when available).
static LogicalResult isEquivalentToOp(PDLValue value, ArrayAttr constantParams,
                                      PatternRewriter &rewriter) {
  auto *maybeOp = value.dyn_cast<Operation *>();
  if (!maybeOp)
    return failure();  // TODO: notifyMatchFailure needs an Operation* handle.
  Operation *op = maybeOp;

  ArrayRef<Attribute> attrs = constantParams.getValue();
  if (attrs.size() != 1)
    return failure();  // TODO: notifyMatchFailure needs an Operation* handle.
  auto modelOpNameAttr = attrs.front().dyn_cast<StringAttr>();
  if (!modelOpNameAttr)
    return failure();  // TODO: notifyMatchFailure needs an Operation* handle.
  auto modelOpName = modelOpNameAttr.strref();

  // 1. If op has name `modelOpName`, the match is trivial.
  if (op->getName().getStringRef() == modelOpName) return success();

  // 2. Linalg vs Linalg.
  // Create op from `constantParams`.
  OperationState modelOpState(op->getLoc(), modelOpName, op->getOperands(),
                              op->getResultTypes(), op->getAttrs());
  modelOpState.addRegion();
  Operation *modelOp = rewriter.createOperation(modelOpState);
  auto g1 = llvm::make_scope_exit([&]() { rewriter.eraseOp(modelOp); });
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  LinalgOp linalgModelOp = dyn_cast<LinalgOp>(modelOp);
  if (linalgOp && linalgModelOp)
    return isEquivalentToOpImpl(linalgOp, linalgModelOp, rewriter);

  // 3. TBD
  return failure();
}

/// Assume that:
///   1. `value` is an operands range
///   2. `constantParams` contains a DictAttr with `operand_number`, `dim` and
///      `divisor` IntegerAttr entries.
/// Succeed if `value`[`operand_number`] is a ranked type whose `dim` is a
/// multiple of `divisor`.
/// Note: 0 is the convention to express "do not tile", it is considered to
/// divide everything.
static LogicalResult isDimMultipleOf(PDLValue value, ArrayAttr constantParams,
                                     PatternRewriter &rewriter) {
  auto maybeOperands = value.dyn_cast<ValueRange>();
  if (!maybeOperands)
    return failure();  // TODO: notifyMatchFailure needs an Operation* handle.
  auto operands = *maybeOperands;

  auto dict = constantParams.begin()->dyn_cast<DictionaryAttr>();
  if (!dict)
    return failure();  // TODO: notifyMatchFailure needs an Operation* handle.

  int64_t dim;
  auto dimAttr = dict.getAs<IntegerAttr>("dim");
  if (!dimAttr)
    return failure();  // TODO: notifyMatchFailure needs an Operation* handle.
  dim = dimAttr.getInt();

  int64_t divisor;
  auto divisorAttr = dict.getAs<IntegerAttr>("divisor");
  if (!divisorAttr)
    return failure();  // TODO: notifyMatchFailure needs an Operation* handle.
  divisor = divisorAttr.getInt();

  int64_t operandNumber;
  auto operandNumberAttr = dict.getAs<IntegerAttr>("operand_number");
  if (!operandNumberAttr)
    return failure();  // TODO: notifyMatchFailure needs an Operation* handle.
  operandNumber = operandNumberAttr.getInt();

  ShapedType shapedType;
  if (static_cast<int64_t>(operands.size()) > operandNumber)
    shapedType = operands[operandNumber].getType().dyn_cast<ShapedType>();
  if (!shapedType || shapedType.getRank() <= dim) return failure();
  return success(divisor == 0 || (shapedType.getShape()[dim] > 0 &&
                                  shapedType.getShape()[dim] % divisor == 0));
}

/// Assume that:
///   1. `value` is an operands range
///   2. `constantParams` contains a DictAttr with `operand_number` and `dim`
///       IntegerAttr entries.
/// Succeed if `value`[`operand_number`] is a ranked type whose `dim` is
/// dynamic.
static LogicalResult isDimStatic(PDLValue value, ArrayAttr constantParams,
                                 PatternRewriter &rewriter) {
  auto maybeOperands = value.dyn_cast<ValueRange>();
  if (!maybeOperands)
    return failure();  // TODO: notifyMatchFailure needs an Operation* handle.
  auto operands = *maybeOperands;

  auto dict = constantParams.begin()->dyn_cast<DictionaryAttr>();
  if (!dict)
    return failure();  // TODO: notifyMatchFailure needs an Operation* handle.

  int64_t dim;
  auto dimAttr = dict.getAs<IntegerAttr>("dim");
  if (!dimAttr)
    return failure();  // TODO: notifyMatchFailure needs an Operation* handle.
  dim = dimAttr.getInt();

  int64_t operandNumber;
  auto operandNumberAttr = dict.getAs<IntegerAttr>("operand_number");
  if (!operandNumberAttr)
    return failure();  // TODO: notifyMatchFailure needs an Operation* handle.
  operandNumber = operandNumberAttr.getInt();

  ShapedType shapedType;
  if (static_cast<int64_t>(operands.size()) > operandNumber)
    shapedType = operands[operandNumber].getType().dyn_cast<ShapedType>();
  return success(shapedType && !shapedType.isDynamicDim(dim));
}

/// Assume that:
///   1. `value` is an operands range
///   2. `constantParams` contains a DictAttr with `operand_number` and `dim`
///       IntegerAttr entries.
/// Succeed if `value`[`operand_number`] is a ranked type whose `dim` is
/// dynamic.
static LogicalResult isDimDynamic(PDLValue value, ArrayAttr constantParams,
                                  PatternRewriter &rewriter) {
  auto maybeOperands = value.dyn_cast<ValueRange>();
  if (!maybeOperands)
    return failure();  // TODO: notifyMatchFailure needs an Operation* handle.
  auto operands = *maybeOperands;

  auto dict = constantParams.begin()->dyn_cast<DictionaryAttr>();
  if (!dict)
    return failure();  // TODO: notifyMatchFailure needs an Operation* handle.

  int64_t dim;
  auto dimAttr = dict.getAs<IntegerAttr>("dim");
  if (!dimAttr)
    return failure();  // TODO: notifyMatchFailure needs an Operation* handle.
  dim = dimAttr.getInt();

  int64_t operandNumber;
  auto operandNumberAttr = dict.getAs<IntegerAttr>("operand_number");
  if (!operandNumberAttr)
    return failure();  // TODO: notifyMatchFailure needs an Operation* handle.
  operandNumber = operandNumberAttr.getInt();

  ShapedType shapedType;
  if (static_cast<int64_t>(operands.size()) > operandNumber)
    shapedType = operands[operandNumber].getType().dyn_cast<ShapedType>();
  return success(shapedType && shapedType.isDynamicDim(dim));
}

FailureOr<SmallVector<Operation *>> findMatchingOps(Operation *op,
                                                    SymbolRefAttr pattern,
                                                    ModuleOp module) {
  auto patternOp = module.lookupSymbol<pdl::PatternOp>(pattern);
  if (!patternOp)
    return {op->emitError("could not find a pattern named: ") << pattern};

  // Clone the pattern operation into the temporary module used by the driver
  // as it might be referenced multiple times.
  OwningOpRef<ModuleOp> pdlModuleOp = ModuleOp::create(patternOp.getLoc());
  OpBuilder::atBlockBegin(pdlModuleOp->getBody()).clone(*patternOp);

  // Build the PDL module.
  PDLPatternModule pdlModule(std::move(pdlModuleOp));
  pdlModule.registerConstraintFunction("nestedInFunc", nestedInFunc);
  pdlModule.registerConstraintFunction("isDimDynamic", isDimDynamic);
  pdlModule.registerConstraintFunction("isDimMultipleOf", isDimMultipleOf);
  pdlModule.registerConstraintFunction("isDimStatic", isDimStatic);
  pdlModule.registerConstraintFunction("isEquivalentToOp", isEquivalentToOp);
  pdlModule.registerRewriteFunction("iree_linalg_transform.apply",
                                    noOpRewriter);

  RewritePatternSet patterns(std::move(pdlModule));
  return getMatchingOps(module, std::move(patterns));
}

}  // namespace linalg
}  // namespace mlir
