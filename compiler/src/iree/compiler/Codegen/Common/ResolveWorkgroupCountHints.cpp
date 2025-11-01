// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//=== ResolveWorkgroupCountHints.cpp
//---------------------------------------===//
//
// While lowering executable target, the pipelines used are run at a
// func-like op granularity. Each of these func-like operations set the
// workgroup size, and subgroup size as required (as part of the
// `TranslationInfo`). Eventually these have to be reconciled and set
// appropriately on the surrounding HAL ops for the host runtime to pick them
// up. In case of inconsistencies, this pass will throw an error.
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_RESOLVEWORKGROUPCOUNTHINTSPASS
#include <mlir/Analysis/TopologicalSortUtils.h>
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

class ResolveWorkgroupCountHintsPass final
    : public impl::ResolveWorkgroupCountHintsPassBase<
          ResolveWorkgroupCountHintsPass> {
public:
  using Base::Base;
  void runOnOperation() override;
};

using OperandWalkFn = std::function<LogicalResult(OpOperand &)>;
static LogicalResult walkOperandsAndCaptures(Operation *root, Operation *curr,
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
        if (failed(walkOperandsAndCaptures(root, &op, dominance, callback))) {
          return failure();
        }
      }
    }
  }
  return success();
}

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

  DominanceInfo dominance(root);
  for (Region &region : root->getRegions()) {
    for (Block &body : region.getBlocks()) {
      for (Operation &op : body.getOperations()) {
        if (failed(walkOperandsAndCaptures(root, &op, dominance, callback))) {
          return failure();
        }
      }
    }
  }
  return success();
}

// `getBackwardsSlice` silently skips or passes through block arguments meaning
// we can't capture function args. Roll our own impl that does what we need.
static LogicalResult getBackwardOrdinalSlice(
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
        return getBackwardOrdinalSlice(definingOp, visited, backwardSlice,
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

static LogicalResult getBackwardOrdinalSlice(
    Operation *op, SetVector<Operation *> &backwardSlice,
    SetVector<BlockArgument> &funcArgLeaves,
    SetVector<IREE::TensorExt::DispatchWorkloadOrdinalOp> &ordinalLeaves,
    bool skipCaptures = true, bool inclusive = false) {
  DenseSet<Operation *> visited;
  LogicalResult res =
      getBackwardOrdinalSlice(op, visited, &backwardSlice, funcArgLeaves,
                              ordinalLeaves, skipCaptures, inclusive);
  if (succeeded(res)) {
    topologicalSort(backwardSlice);
  }
  return res;
}

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
                             SmallVector<ConditionSlice> &slices) {
  ConditionSlice slice;
  slice.negate = isInElseBranch;
  LogicalResult res = getBackwardOrdinalSlice(
      ifOp, slice.slice, slice.funcArgLeaves, slice.ordinalLeaves);
  if (succeeded(res)) {
    slices.push_back(std::move(slice));
  }
}

static void processConditionsFromBase(Operation *base,
                                      SmallVector<ConditionSlice> &slices) {
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

// Descriptor of a program slice for a `workgroup_count_hint` op in terms of
// workload ordinals and the parent function.
struct HintSlice {
  IREE::Codegen::WorkgroupCountHintOp hintOp;
  // The program slice that produces the operands to the hint op.
  SetVector<Operation *> slice;
  // Arguments of the parent function that are inputs to the slice.
  SetVector<BlockArgument> funcArgLeaves;
  // Workload ordinals that are inputs to the slice.
  SetVector<IREE::TensorExt::DispatchWorkloadOrdinalOp> ordinalLeaves;
  // List of conditions on which to predicate this count hint.
  SmallVector<ConditionSlice> conditions;
};

static LogicalResult processHint(IREE::Codegen::WorkgroupCountHintOp hintOp,
                                 HintSlice &slice) {
  processConditionsFromBase(hintOp, slice.conditions);
  return getBackwardOrdinalSlice(hintOp, slice.slice, slice.funcArgLeaves,
                                 slice.ordinalLeaves);
}

// Descriptor of a program slice for the producers of an individual operand.
struct OperandSlice {
  OpOperand *operand;
  // The program slice that produces this operand.
  SetVector<Operation *> slice;
  // Arguments of the parent function that are inputs to the slice.
  SetVector<BlockArgument> funcArgLeaves;
  // Workload ordinals that are inputs to the slice.
  SetVector<IREE::TensorExt::DispatchWorkloadOrdinalOp> ordinalLeaves;
};

static LogicalResult processOperand(OpOperand &operand,
                                    SmallVector<OperandSlice> slices) {
  if (auto blockArg = dyn_cast<BlockArgument>(operand.get())) {
    auto funcOp =
        dyn_cast<FunctionOpInterface>(blockArg.getOwner()->getParentOp());
    if (!funcOp) {
      return failure();
    }
    OperandSlice slice;
    slice.operand = &operand;
    slice.funcArgLeaves.insert(blockArg);
    slices.push_back(std::move(slice));
  }

  OperandSlice slice;
  LogicalResult res = getBackwardOrdinalSlice(
      operand.get().getDefiningOp(), slice.slice, slice.funcArgLeaves,
      slice.ordinalLeaves, /*skipCaptures=*/false, /*inclusive=*/true);
  if (succeeded(res)) {
    slices.push_back(std::move(slice));
  }
  return res;
}

// Descriptor of a program slice for a function call in terms of workload
// ordinals and the parent function.
struct CallSlice {
  // List of program slices for each optional operand.
  SmallVector<OperandSlice> optionalOperands;
  // List of program slices for each required operand.
  SmallVector<OperandSlice> requiredOperands;
  // List of conditions on which to predicate this function call.
  SmallVector<ConditionSlice> conditions;
};

struct FunctionSlice {
  // List of operands that may optionally resolve in terms of workload ordinals.
  // These are operands that are only used as arguments to conditions.
  SetVector<BlockArgument> optionalOperands;
  // List of operands that are required to resolve in terms of workload
  // ordinals. These are operands for workgroup count hints.
  SetVector<BlockArgument> requiredOperands;
  // List of workgroup count hints contained within this function.
  SmallVector<HintSlice> hints;
  // List of function calls contained within this function.
  SmallVector<CallSlice> calls;
  bool valid = false;
  bool required = false;
};

static LogicalResult processCall(CallOpInterface callOp,
                                 FunctionOpInterface func, CallSlice &slice,
                                 FunctionSlice &funcSlice) {
  assert(funcSlice.valid && funcSlice.required &&
         "unexpected processing of an invalid or non-required function call");

  MutableOperandRange callOperands = callOp.getArgOperandsMutable();

  FunctionOpInterface::BlockArgListType funcArgs = func.getArguments();
  int64_t argStart = funcArgs.front().getArgNumber();
  for (BlockArgument argument : func.getArguments()) {
    OpOperand &operand = callOperands[argument.getArgNumber() + argStart];
    if (funcSlice.requiredOperands.contains(argument)) {
      if (failed(processOperand(operand, slice.requiredOperands))) {
        return failure();
      }
    } else if (funcSlice.optionalOperands.contains(argument)) {
      if (failed(processOperand(operand, slice.optionalOperands))) {
        continue;
      }
    }
  }
  return failure();
}

static LogicalResult
processFunction(FunctionOpInterface func, SymbolTableCollection &symbolTables,
                llvm::DenseMap<FunctionOpInterface, FunctionSlice> &slices) {
  auto res = slices.try_emplace(func);
  if (!res.second) {
    // Recursion check. We don't support recursively called hints so fail if
    // this slice hasn't been marked valid (i.e. re-visited while processing
    // itself).
    return success(res.first->second.valid);
  }

  FunctionSlice &slice = res.first->second;
  // Ignore external functions. Since workgroup count hints don't have execution
  // semantics we can safely assume external functions don't include them. If we
  // wanted a way to define workgroup counts externally we would need to
  // introduce a special function op for it.
  if (func.isExternal()) {
    slice.valid = true;
    slice.required = false;
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
            if (failed(processCall(callOp, func, callSlice, transitiveSlice))) {
              return WalkResult::interrupt();
            }
            auto argOperands = callOp.getArgOperands();
            auto functionArgs = callee.getArguments();
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

static LogicalResult materializeSlice(
    RewriterBase &rewriter, IRMapping &map, ValueRange workloadVals,
    SetVector<Operation *> &slice,
    SetVector<IREE::TensorExt::DispatchWorkloadOrdinalOp> &ordinals) {
  for (auto ordinalOp : ordinals) {
    // Map `tensor_ext.dispatch.constant_ordinal` op with the corresponding
    // operand of the `tensor_ext.dispatch.workgroup_count_default` operation.
    int64_t ordinal = ordinalOp.getOrdinal().getSExtValue();
    if (ordinal >= workloadVals.size()) {
      return ordinalOp.emitOpError(
          "ordinal number is higher than the number of workloads captured in "
          "the workgroup count region");
    }
    map.map(ordinalOp.getResult(),
            workloadVals[ordinalOp.getOrdinal().getSExtValue()]);
  }
  for (auto op : slice) {
    // TODO(#13038) This is a WAR for the these ops ending up in workgroup count
    // computation. They should not. Some pre-processing at MaterializeEncoding
    // time might make these go away.
    if (isa<IREE::Codegen::QueryTileSizesOp>(op)) {
      Value constVal =
          arith::ConstantIndexOp::create(rewriter, op->getLoc(), 16);
      for (auto result : op->getResults()) {
        map.map(result, constVal);
      }
      continue;
    }
    rewriter.clone(*op, map);
  }
  return success();
}

static Value negateValue(RewriterBase &rewriter, Value v) {
  // TODO
  return v;
}

static Value andValues(RewriterBase &rewriter, Value l, Value r) {
  // TODO
  return l;
}

static FailureOr<Operation *>
materializeConditions(RewriterBase &rewriter, IRMapping &mapping, Location loc,
                      ValueRange workloadVals,
                      SmallVector<ConditionSlice> &conditions) {
  auto isConditionSupported = [&](ConditionSlice &condition) {
    return llvm::all_of(condition.funcArgLeaves,
                        [&](BlockArgument b) { return mapping.contains(b); });
  };

  Value acc;
  for (ConditionSlice &condition : conditions) {
    if (!llvm::all_of(condition.funcArgLeaves,
                      [&](BlockArgument b) { return mapping.contains(b); })) {
      continue;
    }

    if (failed(materializeSlice(rewriter, mapping, workloadVals,
                                condition.slice, condition.ordinalLeaves))) {
      return failure();
    }

    Value newCond = mapping.lookupOrNull(condition.cond);
    if (!newCond) {
      return failure();
    }

    if (condition.negate) {
      newCond = negateValue(rewriter, newCond);
    }

    if (!acc) {
      acc = newCond;
    } else {
      acc = andValues(rewriter, acc, newCond);
    }
  }

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
        })
        .getResult(0);
  }
  return maybePlaceholder;
}

static FailureOr<SmallVector<Value>> materializeHint(RewriterBase &rewriter,
                                                     IRMapping &map,
                                                     ValueRange workloadVals,
                                                     HintSlice &hintSlice) {
  Location loc = hintSlice.hintOp.getLoc();
  FailureOr<Operation *> maybePlaceholder = materializeConditions(
      rewriter, map, loc, workloadVals, hintSlice.conditions);
  if (failed(maybePlaceholder)) {
    return failure();
  }

  Operation *placeholder = maybePlaceholder.value();
  if (placeholder) {
    rewriter.setInsertionPoint(placeholder);
  }

  if (failed(materializeSlice(rewriter, map, workloadVals, hintSlice.slice,
                              hintSlice.ordinalLeaves))) {
    return failure();
  }

  SmallVector<Value> sizes;
  for (OpFoldResult size : hintSlice.hintOp.getMixedSizes()) {
    if (auto v = dyn_cast<Value>(size)) {
      Value replacement = map.lookupOrNull(v);
      if (!replacement) {
        return failure();
      }
      sizes.push_back(v);
    } else {
      sizes.push_back(getValueOrCreateConstantIndexOp(rewriter, loc, size));
    }
  }

  if (placeholder) {
    SmallVector<Value> newSizes = placeholder->getParentOp()->getResults();
    rewriter.replaceOp(placeholder, sizes);
    sizes = newSizes;
  }

  return sizes;
}

static LogicalResult replaceFromSliceOpWithFunctionSlice(
    RewriterBase &rewriter,
    IREE::TensorExt::DispatchWorkgroupCountFromSliceOp fromSliceOp,
    FunctionSlice &root) {
  ValueRange workloadVals = fromSliceOp.getOperands();
  IRMapping map;
  SmallVector<OpFoldResult> results;
  // // Since the workgroup count at HAL level is in x, y, z form, process the
  // // workload in reverse.
  // for (auto ofr : llvm::reverse(workgroupCount)) {
  //   if (auto val = dyn_cast<Value>(ofr)) {
  //     results.push_back(getAsOpFoldResult(map.lookup(val)));
  //   } else {
  //     results.push_back(ofr);
  //   }
  // }
}

} // namespace

void ResolveWorkgroupCountHintsPass::runOnOperation() {
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp innerModuleOp = variantOp.getInnerModule();
  MLIRContext *context = &getContext();
  (void)context;

  llvm::DenseMap<FunctionOpInterface, FunctionSlice> slices;

  // Construct the call-graph for the inner module.
  CallGraph callGraph(innerModuleOp);
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
    if (!exportOp.getWorkgroupCountBody()
             ->walk([&](Operation *op) -> WalkResult {
               fromSliceOp =
                   dyn_cast<IREE::TensorExt::DispatchWorkgroupCountFromSliceOp>(
                       op);
               return fromSliceOp ? WalkResult::interrupt()
                                  : WalkResult::advance();
             })
             .wasInterrupted()) {
      if (hasSlice && sliceIt->second.required) {
        exportOp->emitOpError("exporting function with a workgroup count hint "
                              "yet `workgroup_count_from_slice` not found.");
        return signalPassFailure();
      }
      continue;
    }

    if (!hasSlice || !sliceIt->second.required || !sliceIt->second.valid) {
      exportOp->emitOpError(
          "exporting function with `workgroup_count_from_slice` yet no "
          "`workgroup_count_hint` found.");
      return signalPassFailure();
    }

    FunctionSlice &root = sliceIt->second;
    rewriter.setInsertionPoint(fromSliceOp);
    if (failed(
            replaceFromSliceOpWithFunctionSlice(rewriter, fromSliceOp, root))) {
      return signalPassFailure();
    }

    if (!exportOp.getWorkgroupCountBody()
             ->walk([](Operation *op) {
               return isa<IREE::TensorExt::DispatchWorkgroupCountFromSliceOp>(
                          op)
                          ? WalkResult::interrupt()
                          : WalkResult::advance();
             })
             .wasInterrupted()) {
      exportOp->emitOpError(
          "failed to convert all `workgroup_count_from_slice` ops.");
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler
