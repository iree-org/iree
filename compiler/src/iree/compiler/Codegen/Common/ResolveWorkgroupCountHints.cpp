// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//=== ResolveWorkgroupCountHints.cpp -------------------------------------===//
//
// Resolves `iree_codegen.workgroup_count_hint` ops by materializing a globally
// agreeable workgroup count per export. This pass performs a depth first walk
// of the target variant's call graph constructing the necessary information to
// materialize workgroup count hints for that function.
//
// Workgroup counts are materialized by again walking the callgraph starting
// from each exported function and mapping the required operands to construct
// the hints for a function using the operands of the callsite we're traversing
// from.
//
// Additionally any conditional statements (scf.if ops) transitively wrapping
// a workgroup count hint are tracked and materialized in the workgroup count
// region as well. This avoids pessimizing the workgroup count for situations
// where we are switching implementations on uniform host provided parameters
// (workloads/device queries).
//
// TODO: Implement device query support.
//
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/HAL/IR/HALTraits.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_RESOLVEWORKGROUPCOUNTHINTSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

class ResolveWorkgroupCountHintsPass final
    : public impl::ResolveWorkgroupCountHintsPassBase<
          ResolveWorkgroupCountHintsPass> {
public:
  using Base::Base;
  void runOnOperation() override;
};

//===---------------------------------------------------------------------===//
// Utilities
//===---------------------------------------------------------------------===//

static Value negateValue(RewriterBase &rewriter, Location loc, Value v) {
  auto one =
      arith::ConstantIntOp::create(rewriter, loc, rewriter.getI1Type(), 1);
  return arith::XOrIOp::create(rewriter, loc, v, one);
}

static Value andValues(RewriterBase &rewriter, Location loc, Value l, Value r) {
  return arith::AndIOp::create(rewriter, loc, l, r);
}

static SmallVector<Value, 3> maxValueVectors(RewriterBase &rewriter,
                                             Location loc, ArrayRef<Value> l,
                                             ArrayRef<Value> r) {
  SmallVector<Value, 3> max;
  for (auto [lv, rv] : llvm::zip_equal(l, r)) {
    max.push_back(arith::MaxSIOp::create(rewriter, loc, lv, rv));
  }
  return max;
}

using OperandWalkFn = std::function<LogicalResult(OpOperand &)>;
static LogicalResult walkOperandsAndCapturesImpl(Operation *root,
                                                 Operation *curr,
                                                 DominanceInfo &dominance,
                                                 OperandWalkFn callback) {
  for (OpOperand &operand : curr->getOpOperands()) {
    if (dominance.properlyDominates(operand.get(), root)) {
      if (failed(callback(operand))) {
        return failure();
      }
    }
  }

  for (Region &region : root->getRegions()) {
    for (Block &body : region.getBlocks()) {
      for (Operation &op : body.getOperations()) {
        if (failed(
                walkOperandsAndCapturesImpl(root, &op, dominance, callback))) {
          return failure();
        }
      }
    }
  }
  return success();
}

// Recursive walk of all operands and implicit captures of |root|. This
// allows fetching all values transitively required by |root|. If
// |skipCaptures| is set, this will skip lookup of implicit captures for
// |root| only.
static LogicalResult walkOperandsAndCaptures(Operation *root,
                                             OperandWalkFn callback,
                                             bool skipCaptures) {
  for (OpOperand &operand : root->getOpOperands()) {
    if (failed(callback(operand))) {
      return failure();
    }
  }

  if (skipCaptures) {
    return success();
  }

  // Dominance for checking operand capture is maybe more expensive for the
  // first query of an operation per block, but caches the info for subsequent
  // queries.
  DominanceInfo dominance(root);
  for (Region &region : root->getRegions()) {
    for (Block &body : region.getBlocks()) {
      for (Operation &op : body.getOperations()) {
        if (failed(
                walkOperandsAndCapturesImpl(root, &op, dominance, callback))) {
          return failure();
        }
      }
    }
  }
  return success();
}

static LogicalResult getBackwardOrdinalSliceImpl(
    Operation *op, DenseSet<Operation *> &visited,
    SetVector<Operation *> *backwardSlice,
    SetVector<BlockArgument> &funcArgLeaves,
    SetVector<IREE::TensorExt::DispatchWorkloadOrdinalOp> &ordinalLeaves,
    bool skipCaptures = false, bool inclusive = true) {
  if (auto ordinal = dyn_cast<IREE::TensorExt::DispatchWorkloadOrdinalOp>(op)) {
    ordinalLeaves.insert(ordinal);
    return success();
  }

  auto processOperand = [&](OpOperand &operand) {
    if (Operation *definingOp = operand.get().getDefiningOp()) {
      if (backwardSlice->count(definingOp) == 0 &&
          visited.insert(definingOp).second) {
        // Memory effecting ops or hal.interface ops are illegal outside of
        // executables. This is possible if the code branches on a value
        // resident on the device rather than a host provided param.
        // In such cases we will ignore those conditionals. If a hint op
        // directly depends on an illegal op that is a hard error however.
        if (!isMemoryEffectFree(definingOp) ||
            definingOp->hasTrait<OpTrait::IREE::HAL::ExecutableInterfaceOp>()) {
          return failure();
        }
        return getBackwardOrdinalSliceImpl(definingOp, visited, backwardSlice,
                                           funcArgLeaves, ordinalLeaves);
      }

      visited.erase(definingOp);
    } else if (auto blockArg = dyn_cast<BlockArgument>(operand.get())) {
      // Function op arguments are terminal. Everything else is treated as a
      // failure.
      auto funcOp =
          dyn_cast<FunctionOpInterface>(blockArg.getOwner()->getParentOp());
      if (!funcOp) {
        return failure();
      }
      funcArgLeaves.insert(blockArg);
    } else {
      return failure();
    }

    return success();
  };
  if (failed(walkOperandsAndCaptures(op, processOperand, skipCaptures))) {
    return failure();
  }

  if (inclusive) {
    backwardSlice->insert(op);
  }
  return success();
}

// Looks up the backward slice of |op| including implicit captures and populates
// |funcArgLeaves| with all required function arguments of the immediately
// enclosing function and |ordinalLeaves| with all defining
// `tensor_ext.dispatch.workload.ordinal` ops.
//
// Returns failure if a valid backward slice could not be found. This occurs if
// the slice includes a non-function block argument or an unsupported operation
// (memory effecting or non-uniform like a workgroup id).
//
// `getBackwardsSlice` silently skips or passes through block arguments meaning
// we can't capture function args. Roll our own impl that does what we need.
static LogicalResult getBackwardOrdinalSlice(
    Operation *op, SetVector<Operation *> &backwardSlice,
    SetVector<BlockArgument> &funcArgLeaves,
    SetVector<IREE::TensorExt::DispatchWorkloadOrdinalOp> &ordinalLeaves,
    bool skipCaptures = true, bool inclusive = false) {
  DenseSet<Operation *> visited;
  LogicalResult res =
      getBackwardOrdinalSliceImpl(op, visited, &backwardSlice, funcArgLeaves,
                                  ordinalLeaves, skipCaptures, inclusive);
  if (succeeded(res)) {
    // Topological sort for cloning later. On failure the slice is not used
    // so no need.
    topologicalSort(backwardSlice);
  }
  return res;
}

//===---------------------------------------------------------------------===//
// State + Processing
//===---------------------------------------------------------------------===//

/// All the state and processing functions needed to materialize workgroup
/// counts are defined in this section. A structure for the state as well as
/// as well as processing functions that populate that state are provided for
/// the following IR constructs:
///
///  - `scf.if`
///  - `iree_codegen.workgroup_count_hint`
///  - `OpOperand`
///  - `CallOpInterface`
///  - `FunctionOpInterface`

/// ================
/// *** `scf.if` ***
/// ================
///   - ConditionSlice
///   - processCondition
///
/// Processes + holds state for `scf.if` ops wrapping `workgroup_count_hint`
/// ops. This allows optimizing the workgroup count for cases where the count
/// is switched on kernel uniform values. For example:
///
/// ```
/// %cond = hal.interface.constant.load ordinal(0) : i1
/// scf.if %cond {
///   iree_codegen.workgroup_count_hint(1, 2, 3)
/// } else {
///   iree_codegen.workgroup_count_hint(4, 5, 6)
/// }
/// ```
///
/// This will generate (effectively):
///
/// ```
/// hal.executable.export count(%cond: i1)
///   %x, %y, %z = scf.if %cond {
///     scf.yield %c1, %c2, %c3
///   } else {
///     scf.yield %c4, %c5, %c6
///   }
///   hal.return %x, %y, %z
/// ```
///
/// As opposed to the more pessimistic:
///
/// ```
/// hal.executable.export count(%cond: i1)
///   %x, %y, %z = max(1, 4), max(2, 5), max(3, 6) = 4, 5, 6
///   hal.return %x, %y, %z
/// ```
struct ConditionSlice {
  // The condition on which to predicate.
  Value cond;
  // The program slice that produces this condition.
  SetVector<Operation *> slice;
  // Arguments of the parent function that are inputs to the slice.
  SetVector<BlockArgument> funcArgLeaves;
  // Workload ordinals that are inputs to the slice.
  SetVector<IREE::TensorExt::DispatchWorkloadOrdinalOp> ordinalLeaves;
  // Whether or not to negate the condition (i.e. else branch).
  bool negate = false;
};

static void processCondition(scf::IfOp ifOp, bool isInElseBranch,
                             std::vector<ConditionSlice> &slices) {
  ConditionSlice slice;
  slice.negate = isInElseBranch;
  LogicalResult res = getBackwardOrdinalSlice(
      ifOp, slice.slice, slice.funcArgLeaves, slice.ordinalLeaves);
  if (succeeded(res)) {
    slice.cond = ifOp.getCondition();
    slices.push_back(std::move(slice));
  }
}

// Populates |slices| with all conditional descriptors wrapping |base| within
// the current function. Skips all conditionals not statically resolvable in
// terms of function args, ordinals, and legal ops.
static void processConditionsFromBase(Operation *base,
                                      std::vector<ConditionSlice> &slices) {
  Operation *parent = base->getParentOp();
  Operation *child = base;
  while (parent && !isa<FunctionOpInterface>(parent)) {
    if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
      processCondition(ifOp,
                       /*isInElseRegion=*/child->getBlock()->getParent() ==
                           &ifOp.getElseRegion(),
                       slices);
    }

    // We need to walk the ancestor chain one at a time to track the child (and
    // thus which branch we come up in).
    child = parent;
    parent = parent->getParentOp();
  }
}

/// ===========================================
/// *** `iree_codegen.workgroup_count_hint` ***
/// ===========================================
///   - HintSlice
///   - processHint
///
/// Processes + holds state for `workgroup_count_hint` ops. Hints are processed
/// individually and only hold the state needed to resolve it in the context of
/// the function it is contained within. This populates:
///
///  - The backward program slice of the operands to the hint op.
///  - All operands of the enclosing function the backward slice depends on.
///  - All `iree_tensor_ext.dispatch.workload.ordinal` ops the slice depends on.
///  - The ConditionSlices of all `scf.if` ops that are ancestors of the hint.
struct HintSlice {
  IREE::Codegen::WorkgroupCountHintOp hintOp;
  // The program slice that produces the operands to the hint op.
  SetVector<Operation *> slice;
  // Arguments of the parent function that are inputs to the slice.
  SetVector<BlockArgument> funcArgLeaves;
  // Workload ordinals that are inputs to the slice.
  SetVector<IREE::TensorExt::DispatchWorkloadOrdinalOp> ordinalLeaves;
  // List of conditions on which to predicate this count hint.
  std::vector<ConditionSlice> conditions;
};

static LogicalResult processHint(IREE::Codegen::WorkgroupCountHintOp hintOp,
                                 HintSlice &slice) {
  processConditionsFromBase(hintOp, slice.conditions);
  if (failed(getBackwardOrdinalSlice(hintOp, slice.slice, slice.funcArgLeaves,
                                     slice.ordinalLeaves))) {
    return hintOp->emitOpError(
        "failed to resolve workgroup count hint in terms of workload ordinals");
  }
  slice.hintOp = hintOp;
  return success();
}

/// ===================
/// *** `OpOperand` ***
/// ===================
///   - OperandSlice
///   - processOperand

/// Descriptor of a program slice for the producers of an individual operand.
struct OperandSlice {
  OpOperand *operand;
  // The program slice that produces this operand.
  SetVector<Operation *> slice;
  // Arguments of the parent function that are inputs to the slice.
  SetVector<BlockArgument> funcArgLeaves;
  // Workload ordinals that are inputs to the slice.
  SetVector<IREE::TensorExt::DispatchWorkloadOrdinalOp> ordinalLeaves;
};

// If possible, appends to |slices| with a new backwards program slice that
// produces |operand|. On failure |slices| remains unchanged.
static LogicalResult processOperand(OpOperand &operand,
                                    std::vector<OperandSlice> &slices) {
  OperandSlice slice;
  slice.operand = &operand;
  if (auto blockArg = dyn_cast<BlockArgument>(operand.get())) {
    auto funcOp =
        dyn_cast<FunctionOpInterface>(blockArg.getOwner()->getParentOp());
    if (!funcOp) {
      return failure();
    }
    slice.funcArgLeaves.insert(blockArg);
    slices.push_back(std::move(slice));
    return success();
  }

  Operation *definingOp = operand.get().getDefiningOp();
  if (!isMemoryEffectFree(definingOp) ||
      definingOp->hasTrait<OpTrait::IREE::HAL::ExecutableInterfaceOp>()) {
    return failure();
  }

  LogicalResult res = getBackwardOrdinalSlice(
      definingOp, slice.slice, slice.funcArgLeaves, slice.ordinalLeaves,
      /*skipCaptures=*/false, /*inclusive=*/true);
  if (succeeded(res)) {
    slices.push_back(std::move(slice));
  }
  return res;
}

/// =========================
/// *** `CallOpInterface` ***
/// =========================
///   - CallSlice
///   - processCall

/// Descriptor of a program slice for a function call in terms of workload
/// ordinals and the parent function. Backward slices are stored per operand
/// in case when materializing a function, one of the optional operands wasn't
/// resolvable in terms of workload ordinals and legal ops. This allows us to
/// skip dependent conditionals per callsite rather than globally pessimizing.
/// The per operand approach potentially introduces repeat IR, however CSE
/// should be able to clean it up as only pure ops are legal.
struct CallSlice {
  // Cache the callee to avoid repeat symbol table lookups.
  FunctionOpInterface callee;
  // List of program slices for each optional operand.
  std::vector<OperandSlice> optionalOperands;
  // List of program slices for each required operand.
  std::vector<OperandSlice> requiredOperands;
  // List of conditions on which to predicate this function call.
  std::vector<ConditionSlice> conditions;
};

/// =============================
/// *** `FunctionOpInterface` ***
/// =============================
///   - FuncSlice
///   - processFunc
struct FunctionSlice {
  // List of operands that may optionally resolve in terms of workload ordinals.
  // These are operands that are only used as arguments to conditions.
  SetVector<BlockArgument> optionalOperands;
  // List of operands that are required to resolve in terms of workload
  // ordinals. These are operands for workgroup count hints.
  SetVector<BlockArgument> requiredOperands;
  // List of workgroup count hints contained within this function.
  std::vector<HintSlice> hints;
  // List of function calls contained within this function.
  std::vector<CallSlice> calls;
  bool valid = false;
  bool required = true;
};

/// Implementation of call processing. Assumes the call transitively calls a
/// `workgroup_count_hint` op and returns failure if the call can't be resolved
/// in terms of valid operands the same way a hint op is (function args and
/// ordinal ops).
static LogicalResult processCall(CallOpInterface callOp,
                                 FunctionOpInterface callee, CallSlice &slice,
                                 FunctionSlice &funcSlice) {
  assert(funcSlice.valid && funcSlice.required &&
         "unexpected processing of an invalid or non-required function call");

  MutableOperandRange callOperands = callOp.getArgOperandsMutable();
  processConditionsFromBase(callOp, slice.conditions);

  FunctionOpInterface::BlockArgListType funcArgs = callee.getArguments();
  for (BlockArgument argument : funcArgs) {
    OpOperand &operand = callOperands[argument.getArgNumber()];
    if (funcSlice.requiredOperands.contains(argument)) {
      if (failed(processOperand(operand, slice.requiredOperands))) {
        callOp->emitOpError("failed to resolve operand ")
            << operand.getOperandNumber()
            << " required to resolve workgroup count in terms of workload "
               "ordinals.";
        return failure();
      }
    } else if (funcSlice.optionalOperands.contains(argument)) {
      if (failed(processOperand(operand, slice.optionalOperands))) {
        continue;
      }
    }
  }
  return success();
}

/// Recursive walk of the callgraph, populating the state of each function along
/// the way. A function is required to materialize if it directly or
/// transitively calls a `workgroup_count_hint` op. Calls that don't call
/// functions that may reach a `workgroup_count_hint` aren't added to the list
/// of calls in the FunctionSlice created for |func|.
static LogicalResult
processFunction(FunctionOpInterface func, SymbolTableCollection &symbolTables,
                llvm::DenseMap<FunctionOpInterface, FunctionSlice> &slices) {
  auto res = slices.try_emplace(func);
  if (!res.second) {
    // Recursion check. We don't support recursively called hints so fail if
    // this slice hasn't been marked valid (i.e. re-visited while processing
    // itself).
    if (!res.first->second.valid) {
      func->emitOpError(
          "detected recursive call when resolving workgroup count hints");
      return failure();
    }
    return success();
  }

  FunctionSlice &slice = res.first->second;
  // Ignore external functions. Since workgroup count hints don't have execution
  // semantics we can safely assume external functions don't include them. If we
  // wanted a way to define workgroup counts externally we would need to
  // introduce a special function op for it.
  if (func.isExternal()) {
    slice.valid = true;
    slice.required = false;
    return success();
  }

  if (func.walk([&](IREE::Codegen::WorkgroupCountHintOp hintOp) {
            HintSlice &hintSlice = slice.hints.emplace_back();
            if (failed(processHint(hintOp, hintSlice))) {
              return WalkResult::interrupt();
            }
            slice.requiredOperands.insert_range(hintSlice.funcArgLeaves);
            for (ConditionSlice &condition : hintSlice.conditions) {
              slice.optionalOperands.insert_range(condition.funcArgLeaves);
            }
            return WalkResult::advance();
          })
          .wasInterrupted()) {
    return failure();
  }

  if (func.walk([&](CallOpInterface callOp) {
            CallInterfaceCallable callable = callOp.getCallableForCallee();
            // Fail on indirect calls for now.
            if (isa<Value>(callable)) {
              return WalkResult::interrupt();
            }
            auto targetSymbol = cast<SymbolRefAttr>(callable);
            FunctionOpInterface callee = cast<FunctionOpInterface>(
                symbolTables.lookupNearestSymbolFrom(callOp, targetSymbol));
            if (failed(processFunction(callee, symbolTables, slices))) {
              return WalkResult::interrupt();
            }
            FunctionSlice &transitiveSlice = slices[callee];
            // We can skip the processing of a call if the callee isn't required
            // for workgroup count.
            if (!transitiveSlice.required) {
              return WalkResult::advance();
            }
            CallSlice &callSlice = slice.calls.emplace_back();
            if (failed(
                    processCall(callOp, callee, callSlice, transitiveSlice))) {
              return WalkResult::interrupt();
            }
            callSlice.callee = callee;
            for (ConditionSlice &condition : callSlice.conditions) {
              slice.optionalOperands.insert_range(condition.funcArgLeaves);
            }
            for (OperandSlice &optionalOperand : callSlice.optionalOperands) {
              slice.optionalOperands.insert_range(
                  optionalOperand.funcArgLeaves);
            }
            for (OperandSlice &requiredOperand : callSlice.requiredOperands) {
              slice.requiredOperands.insert_range(
                  requiredOperand.funcArgLeaves);
            }
            return WalkResult::advance();
          })
          .wasInterrupted()) {
    return failure();
  }

  // Subtract off any required operands that are also marked as optional
  // operands.
  slice.optionalOperands.set_subtract(slice.requiredOperands);

  // Mark the function slice as valid. This way if the overall walk revisits it
  // we will skip reprocessing it.
  slice.valid = true;

  // If this function transitively calls no hints then it isn't required to
  // construct the workgroup count region. This culls processing of calls to
  // this function.
  if (slice.hints.empty() && slice.calls.empty()) {
    slice.required = false;
  }
  return success();
}

//===---------------------------------------------------------------------===//
// Workgroup Count Materialization
//===---------------------------------------------------------------------===//

/// Materializes a program slice. All |ordinals| are remapped to corresponding
/// values in |workloadVals|. All dependent function arguments should already
/// have been remapped and can be queried directly from |map|.
static LogicalResult materializeSlice(
    RewriterBase &rewriter, IRMapping &map, ValueRange workloadVals,
    SetVector<Operation *> &slice,
    SetVector<IREE::TensorExt::DispatchWorkloadOrdinalOp> &ordinals) {
  return materializeSliceFromOrdinals(
      rewriter, map, workloadVals, ordinals.getArrayRef(), slice.getArrayRef());
}

/// Materializes a single combined `scf.if` based on the list of
/// ConditionSlices. If no slices successfully materialize, this produces no
/// new IR.
static FailureOr<Operation *>
materializeConditions(RewriterBase &rewriter, IRMapping &map, Location loc,
                      ValueRange workloadVals,
                      std::vector<ConditionSlice> &conditions) {
  Value acc;
  for (ConditionSlice &condition : conditions) {
    if (!llvm::all_of(condition.funcArgLeaves,
                      [&](BlockArgument b) { return map.contains(b); })) {
      continue;
    }

    if (failed(materializeSlice(rewriter, map, workloadVals, condition.slice,
                                condition.ordinalLeaves))) {
      return failure();
    }

    Value newCond = map.lookupOrNull(condition.cond);
    if (!newCond) {
      return failure();
    }

    if (condition.negate) {
      newCond = negateValue(rewriter, loc, newCond);
    }

    if (!acc) {
      acc = newCond;
    } else {
      acc = andValues(rewriter, loc, acc, newCond);
    }
  }

  // Use a placeholder for the values returned by the scf.if. The builder for
  // scf.if infers the return types from the types yielded by its body, so we
  // need a placeholder producer at the time when we construct the if (which is
  // before its contents have been cloned in).
  Operation *maybePlaceholder = nullptr;
  if (acc) {
    scf::IfOp::create(
        rewriter, loc, acc, /*thenBuilder=*/
        [&](OpBuilder &b, Location l) {
          Type indexType = b.getIndexType();
          maybePlaceholder = UnrealizedConversionCastOp::create(
              b, l, TypeRange{indexType, indexType, indexType}, ValueRange());
          scf::YieldOp::create(b, l, maybePlaceholder->getResults());
        },
        /*elseBuilder=*/
        [&](OpBuilder &b, Location l) {
          auto zero = arith::ConstantIndexOp::create(b, l, 0);
          scf::YieldOp::create(b, l, {zero, zero, zero});
        });
  }
  return maybePlaceholder;
}

/// Materializes the IR for the provided HintSlice using the given mapping at
/// the current insertion point. This amounts to cloning all of the IR cached
/// in `hintSlice.slice` and remapping function args and ordinals based on the
/// provided |map| + |workloadVals|.
static FailureOr<SmallVector<Value, 3>> materializeHint(RewriterBase &rewriter,
                                                        IRMapping &map,
                                                        ValueRange workloadVals,
                                                        HintSlice &hintSlice) {
  Location loc = hintSlice.hintOp.getLoc();
  // If an scf.if wrapping the cloned hint producers is generated, a placeholder
  // `unrealized_conversion_cast` op is generated as a placeholder for the
  // values returned by the cloned backwards slice described in |hintSlice|.
  // The IR would look something like:
  //
  // scf.if %cond {
  //   %x, %y, %z = builtin.unrealized_conversion_cast // no operands
  //   scf.yield %x, %y, %z
  // } else {
  //   %c0 = arith.constant 0 : index
  //   scf.yield %c0, %c0, %c0
  // }
  FailureOr<Operation *> maybePlaceholder = materializeConditions(
      rewriter, map, loc, workloadVals, hintSlice.conditions);
  if (failed(maybePlaceholder)) {
    hintSlice.hintOp->emitOpError("failed to materialize conditions.");
    return failure();
  }

  Operation *placeholder = maybePlaceholder.value();
  if (placeholder) {
    rewriter.setInsertionPoint(placeholder);
  }

  if (failed(materializeSlice(rewriter, map, workloadVals, hintSlice.slice,
                              hintSlice.ordinalLeaves))) {
    hintSlice.hintOp->emitOpError("failed to materialize operand slice.");
    return failure();
  }

  SmallVector<Value, 3> sizes;
  for (OpFoldResult size : hintSlice.hintOp.getMixedSizes()) {
    if (auto v = dyn_cast<Value>(size)) {
      sizes.push_back(map.lookup(v));
    } else {
      sizes.push_back(getValueOrCreateConstantIndexOp(rewriter, loc, size));
    }
  }

  while (sizes.size() < 3) {
    sizes.push_back(arith::ConstantIndexOp::create(rewriter, loc, 1));
  }

  if (placeholder) {
    Operation *parent = placeholder->getParentOp();
    SmallVector<Value, 3> newSizes = parent->getResults();
    rewriter.replaceOp(placeholder, sizes);
    sizes = newSizes;
    // Put the insertion point back to immediately after the enclosing `scf.if`.
    rewriter.setInsertionPointAfter(parent);
  }

  return sizes;
}

// Forward declare for the recursive call.
static FailureOr<SmallVector<Value, 3>>
materializeFunc(RewriterBase &rewriter, IRMapping &map, ValueRange workloadVals,
                FunctionSlice &funcSlice,
                llvm::DenseMap<FunctionOpInterface, FunctionSlice> &sliceMap);

/// Materializes the IR for the provided CallSlice using the given mapping at
/// the current insertion point. This amounts to cloning all of the IR cached
/// in `operandSlice.slice` for each required and optional operand.
/// If one of the optional operands is not resolvable in terms of clonable ops
/// (e.g. hal.interface.* ops or memory effecting ops that can't live on the
/// host), then the conditional op(s) that depend on that optional operands
/// aren't generated.
static FailureOr<SmallVector<Value, 3>>
materializeCall(RewriterBase &rewriter, IRMapping &map, ValueRange workloadVals,
                CallSlice &callSlice,
                llvm::DenseMap<FunctionOpInterface, FunctionSlice> &sliceMap) {
  Location loc = callSlice.callee.getLoc();
  FailureOr<Operation *> maybePlaceholder = materializeConditions(
      rewriter, map, loc, workloadVals, callSlice.conditions);
  if (failed(maybePlaceholder)) {
    callSlice.callee->emitOpError(
        "failed to materialize conditions for function call.");
    return failure();
  }

  Operation *placeholder = maybePlaceholder.value();
  if (placeholder) {
    rewriter.setInsertionPoint(placeholder);
  }

  FunctionOpInterface::BlockArgListType funcArgs =
      callSlice.callee.getArguments();
  SmallVector<BlockArgument> argsToUnmap;
  auto pushOperandSlice = [&](OperandSlice &operandSlice) {
    if (failed(materializeSlice(rewriter, map, workloadVals, operandSlice.slice,
                                operandSlice.ordinalLeaves))) {
      return failure();
    }
    // Link the cloned slice for this operand to the basic block arg of the
    // callee. The function arg is added to `argsToUnmap` so that we can remove
    // the mapping after processing this callsite (and risking stale mappings).
    Value curr = map.lookup(operandSlice.operand->get());
    BlockArgument bbArg = funcArgs[operandSlice.operand->getOperandNumber()];
    map.map(bbArg, curr);
    argsToUnmap.push_back(bbArg);
    return success();
  };

  for (OperandSlice &operandSlice : callSlice.requiredOperands) {
    assert(llvm::all_of(operandSlice.funcArgLeaves,
                        [&](BlockArgument b) { return map.contains(b); }) &&
           "unexpected unmapped required operand");
    if (failed(pushOperandSlice(operandSlice))) {
      callSlice.callee->emitOpError(
          "failed to materialize operand slice for function call.");
      return failure();
    }
  }

  for (OperandSlice &operandSlice : callSlice.optionalOperands) {
    if (!llvm::all_of(operandSlice.funcArgLeaves,
                      [&](BlockArgument b) { return map.contains(b); })) {
      continue;
    }
    if (failed(pushOperandSlice(operandSlice))) {
      return failure();
    }
  }

  FailureOr<SmallVector<Value, 3>> sizes = materializeFunc(
      rewriter, map, workloadVals, sliceMap[callSlice.callee], sliceMap);
  if (failed(sizes)) {
    return failure();
  }

  // Erase the arg mapping for the function args. Each call to a function
  // provides a different mapping and may have a different set of optional
  // args.
  for (Value v : argsToUnmap) {
    map.erase(v);
  }

  assert(sizes->size() == 3 &&
         "function materialization produced too few values");

  if (placeholder) {
    Operation *parent = placeholder->getParentOp();
    SmallVector<Value, 3> newSizes = parent->getResults();
    rewriter.replaceOp(placeholder, *sizes);
    *sizes = newSizes;
    rewriter.setInsertionPointAfter(parent);
  }

  return *sizes;
}

/// Materializes the IR for all contained hint and required call ops at the
/// current insertion point and then takes the maximum values of each
/// elementwise. For example:
///
/// ```
/// func.func @entry_point() {
///   iree_codegen.workgroup_count_hint %x, %y, %z
///   iree_codegen.workgroup_count_hint %a, %b, %c
///   func.call @constant_hint()
/// }
/// func.func @constant_hint() {
///   iree_codegen.workgroup_count_hint 1, 2, 3
/// }
/// ```
///
/// This would resolve to:
///
/// ```
/// hal.executable.export @entry_point count(%x, %y, %z, %a, %b, %c) {
///   %wx = max(%x, %a, 1)
///   %wy = max(%y, %b, 2)
///   %wz = max(%z, %c, 3)
///   hal.return %wx, %wy, %wz
/// }
/// ```
static FailureOr<SmallVector<Value, 3>>
materializeFunc(RewriterBase &rewriter, IRMapping &map, ValueRange workloadVals,
                FunctionSlice &funcSlice,
                llvm::DenseMap<FunctionOpInterface, FunctionSlice> &sliceMap) {
  SmallVector<Value, 3> results;
  for (HintSlice &hint : funcSlice.hints) {
    FailureOr<SmallVector<Value, 3>> newVals =
        materializeHint(rewriter, map, workloadVals, hint);
    if (failed(newVals)) {
      return failure();
    }

    if (results.empty()) {
      results = newVals.value();
    } else {
      results = maxValueVectors(rewriter, hint.hintOp.getLoc(), results,
                                newVals.value());
    }
  }

  for (CallSlice &call : funcSlice.calls) {
    FailureOr<SmallVector<Value, 3>> newVals =
        materializeCall(rewriter, map, workloadVals, call, sliceMap);
    if (failed(newVals)) {
      return failure();
    }

    if (results.empty()) {
      results = newVals.value();
    } else {
      results = maxValueVectors(rewriter, call.callee.getLoc(), results,
                                newVals.value());
    }
  }

  if (results.empty()) {
    // Something went wrong with error catching during analysis or
    // materialization.
    return failure();
  }
  return results;
}

/// Replaces the given `workgroup_count_from_slice` op with the program slice
/// described by the FunctionSlice |root|.
static LogicalResult replaceFromSliceOpWithFunctionSlice(
    RewriterBase &rewriter,
    IREE::TensorExt::DispatchWorkgroupCountFromSliceOp fromSliceOp,
    FunctionSlice &root,
    llvm::DenseMap<FunctionOpInterface, FunctionSlice> &sliceMap) {
  ValueRange workloadVals = fromSliceOp.getOperands();
  IRMapping map;
  FailureOr<SmallVector<Value, 3>> results =
      materializeFunc(rewriter, map, workloadVals, root, sliceMap);
  if (failed(results)) {
    return failure();
  }
  rewriter.replaceOp(fromSliceOp, results.value());
  return success();
}

} // namespace

//===---------------------------------------------------------------------===//
// Pass Impl
//===---------------------------------------------------------------------===//

void ResolveWorkgroupCountHintsPass::runOnOperation() {
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp innerModuleOp = variantOp.getInnerModule();

  // Run the analysis. If a function that contains a `workgroup_count_hint`
  // fails processing, this results in a pass failure. Diagnostic information
  // is emitted by the root cause of the failure so no failure message is
  // needed here.
  llvm::DenseMap<FunctionOpInterface, FunctionSlice> slices;
  SymbolTableCollection symbolTables;
  if (innerModuleOp
          .walk([&](FunctionOpInterface func) -> WalkResult {
            // Always skip on success. This skips walking inside the body of
            // each function and saves on the outer most walk. `processFunction`
            // walks the body so this still ends up as a full module walk.
            if (slices.contains(func)) {
              return WalkResult::skip();
            }
            return failed(processFunction(func, symbolTables, slices))
                       ? WalkResult::interrupt()
                       : WalkResult::skip();
          })
          .wasInterrupted()) {
    return signalPassFailure();
  }

  // Rewrite all `workgroup_count_from_slice` ops based on the result of the
  // analysis.
  IRRewriter rewriter(variantOp);
  for (auto exportOp : variantOp.getOps<IREE::HAL::ExecutableExportOp>()) {
    auto rootFuncOp = llvm::dyn_cast_if_present<FunctionOpInterface>(
        symbolTables.lookupNearestSymbolFrom(innerModuleOp,
                                             exportOp.getSymNameAttr()));
    if (!rootFuncOp || rootFuncOp.isExternal()) {
      // Skip external functions.
      continue;
    }

    auto sliceIt = slices.find(rootFuncOp);
    bool hasSlice = sliceIt != slices.end();
    IREE::TensorExt::DispatchWorkgroupCountFromSliceOp fromSliceOp;
    if (!exportOp.getWorkgroupCountBody() ||
        !exportOp.getWorkgroupCountBody()
             ->walk([&](Operation *op) -> WalkResult {
               fromSliceOp =
                   dyn_cast<IREE::TensorExt::DispatchWorkgroupCountFromSliceOp>(
                       op);
               return fromSliceOp ? WalkResult::interrupt()
                                  : WalkResult::advance();
             })
             .wasInterrupted()) {
      // If the workgroup count was already materialized pass through.
      continue;
    }

    if (!hasSlice || !sliceIt->second.required || !sliceIt->second.valid) {
      // If there is an unresolved `workgroup_count_from_slice` op. The default
      // behavior is to convert this to {1, 1, 1}.
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(fromSliceOp);
      auto one =
          arith::ConstantIndexOp::create(rewriter, fromSliceOp.getLoc(), 1);
      rewriter.replaceOp(fromSliceOp, {one, one, one});
      continue;
    }

    if (hasSlice && !sliceIt->second.valid) {
      // Something went wrong. Should have been caught on processing.
      return signalPassFailure();
    }

    FunctionSlice &root = sliceIt->second;
    rewriter.setInsertionPoint(fromSliceOp);
    if (failed(replaceFromSliceOpWithFunctionSlice(rewriter, fromSliceOp, root,
                                                   slices))) {
      return signalPassFailure();
    }

    if (exportOp.getWorkgroupCountBody()
            ->walk([](Operation *op) {
              return isa<IREE::TensorExt::DispatchWorkgroupCountFromSliceOp>(op)
                         ? WalkResult::interrupt()
                         : WalkResult::advance();
            })
            .wasInterrupted()) {
      exportOp->emitOpError(
          "failed to convert all `workgroup_count_from_slice` ops.");
      return signalPassFailure();
    }
  }

  // Erase all ordinals and hints.
  SmallVector<IREE::TensorExt::DispatchWorkloadOrdinalOp> ordinals;
  SmallVector<IREE::Codegen::WorkgroupCountHintOp> hints;
  variantOp.walk([&](Operation *op) {
    if (auto ordinal =
            dyn_cast<IREE::TensorExt::DispatchWorkloadOrdinalOp>(op)) {
      ordinals.push_back(ordinal);
    } else if (auto hint = dyn_cast<IREE::Codegen::WorkgroupCountHintOp>(op)) {
      hints.push_back(hint);
    }
  });

  for (auto ordinal : ordinals) {
    rewriter.replaceOp(ordinal, ordinal.getOperand());
  }
  for (auto hint : hints) {
    rewriter.eraseOp(hint);
  }
}

} // namespace mlir::iree_compiler
