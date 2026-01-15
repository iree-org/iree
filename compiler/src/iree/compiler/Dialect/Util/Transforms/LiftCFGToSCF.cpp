// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/CFGToSCF.h"

namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DEF_LIFTCFGTOSCFPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// IREE::Util implementation of CFGToSCFInterface
//===----------------------------------------------------------------------===//

// Interface implementation for transforming CFG to SCF in IREE util dialect.
// Similar to the upstream variant but supports our unreachable ops, does not
// assume the `func` dialect is used, and can be used to support custom
// behavior like tied operands.
struct UtilToSCFInterface : public mlir::CFGToSCFInterface {
  // Creates scf.if or scf.index_switch for structured branches.
  FailureOr<Operation *> createStructuredBranchRegionOp(
      OpBuilder &builder, Operation *controlFlowCondOp, TypeRange resultTypes,
      MutableArrayRef<Region> regions) override {
    if (auto condBranchOp = dyn_cast<cf::CondBranchOp>(controlFlowCondOp)) {
      return createStructuredCondBranchRegionOp(builder, condBranchOp,
                                                resultTypes, regions);
    } else if (auto switchOp = dyn_cast<cf::SwitchOp>(controlFlowCondOp)) {
      return createStructuredSwitchRegionOp(builder, switchOp, resultTypes,
                                            regions);
    }
    return controlFlowCondOp->emitOpError(
        "cannot convert unknown control flow op to structured control flow");
  }

  // Handles cf.cond_br -> scf.if conversion.
  static FailureOr<Operation *> createStructuredCondBranchRegionOp(
      OpBuilder &builder, cf::CondBranchOp condBranchOp, TypeRange resultTypes,
      MutableArrayRef<Region> regions) {
    assert(regions.size() == 2 && "cond_br must have exactly 2 successors");
    auto ifOp = scf::IfOp::create(builder, condBranchOp.getLoc(), resultTypes,
                                  condBranchOp.getCondition());
    ifOp.getThenRegion().takeBody(regions[0]);
    ifOp.getElseRegion().takeBody(regions[1]);
    return ifOp.getOperation();
  }

  // Handles cf.switch -> scf.index_switch conversion.
  static FailureOr<Operation *>
  createStructuredSwitchRegionOp(OpBuilder &builder, cf::SwitchOp switchOp,
                                 TypeRange resultTypes,
                                 MutableArrayRef<Region> regions) {
    // Convert the switch flag to index type if needed.
    // cf.switch expects i32 (for some reason), scf.index_switch expects index
    // type (what cf should have).
    Value indexFlag = switchOp.getFlag();
    if (!indexFlag.getType().isIndex()) {
      indexFlag = arith::IndexCastOp::create(builder, switchOp.getLoc(),
                                             builder.getIndexType(), indexFlag);
    }

    // Build case values array.
    SmallVector<int64_t> cases;
    if (auto caseValues = switchOp.getCaseValues()) {
      llvm::append_range(
          cases, llvm::map_range(*caseValues, [](const llvm::APInt &apInt) {
            return apInt.getZExtValue();
          }));
    }
    assert(regions.size() == cases.size() + 1 &&
           "switch must have n+1 regions for n cases");

    // Move regions.
    auto indexSwitchOp =
        scf::IndexSwitchOp::create(builder, switchOp.getLoc(), resultTypes,
                                   indexFlag, cases, cases.size());
    indexSwitchOp.getDefaultRegion().takeBody(regions[0]);
    for (auto &&[targetRegion, sourceRegion] : llvm::zip_equal(
             indexSwitchOp.getCaseRegions(), llvm::drop_begin(regions))) {
      targetRegion.takeBody(sourceRegion);
    }
    return indexSwitchOp.getOperation();
  }

  // Creates scf.yield terminators for structured regions.
  LogicalResult createStructuredBranchRegionTerminatorOp(
      Location loc, OpBuilder &builder, Operation *branchRegionOp,
      Operation *replacedControlFlowOp, ValueRange results) override {
    scf::YieldOp::create(builder, loc, results);
    return success();
  }

  // Creates scf.while for loops.
  FailureOr<Operation *>
  createStructuredDoWhileLoopOp(OpBuilder &builder, Operation *replacedOp,
                                ValueRange loopValuesInit, Value condition,
                                ValueRange loopValuesNextIter,
                                Region &&loopBody) override {
    Location loc = replacedOp->getLoc();

    // Create the while loop with initial values.
    auto whileOp = scf::WhileOp::create(builder, loc, loopValuesInit.getTypes(),
                                        loopValuesInit);

    // Move the loop body to the before region.
    whileOp.getBefore().takeBody(loopBody);

    // Create the condition operation at the end of the before region.
    builder.setInsertionPointToEnd(&whileOp.getBefore().back());

    // Our getCFGSwitchValue returns an index type but scf.condition expects
    // i1. Convert index to i1 (value is guaranteed to be 0 or 1).
    // First cast index to i64 then truncate to i1.
    Value conditionI64 = arith::IndexCastUIOp::create(
        builder, loc, builder.getI64Type(), condition);
    Value conditionI1 = arith::TruncIOp::create(
        builder, loc, builder.getI1Type(), conditionI64);

    scf::ConditionOp::create(builder, loc, conditionI1, loopValuesNextIter);

    // Create the after region with a simple yield.
    Block *afterBlock = builder.createBlock(&whileOp.getAfter());
    afterBlock->addArguments(loopValuesInit.getTypes(),
                             SmallVector<Location>(loopValuesInit.size(), loc));
    scf::YieldOp::create(builder, loc, afterBlock->getArguments());

    return whileOp.getOperation();
  }

  // Creates constants for multiplexer dispatch using an index type.
  Value getCFGSwitchValue(Location loc, OpBuilder &builder,
                          unsigned value) override {
    return arith::ConstantIndexOp::create(builder, loc, value);
  }

  // Creates intermediate switch operations (will be lifted later).
  void createCFGSwitchOp(Location loc, OpBuilder &builder, Value flag,
                         ArrayRef<unsigned> caseValues,
                         BlockRange caseDestinations,
                         ArrayRef<ValueRange> caseArguments, Block *defaultDest,
                         ValueRange defaultArgs) override {
    // Convert index to i32 for cf.switch (it requires i32, unfortunately).
    Value flagI32 =
        arith::IndexCastOp::create(builder, loc, builder.getI32Type(), flag);
    SmallVector<int32_t> caseValuesI32;
    for (unsigned val : caseValues) {
      caseValuesI32.push_back(static_cast<int32_t>(val));
    }

    // Create cf.switch temporarily (will be transformed recursively).
    cf::SwitchOp::create(builder, loc, flagI32, defaultDest, defaultArgs,
                         caseValuesI32, caseDestinations, caseArguments);
  }

  // Gets undefined value for unused block arguments.
  Value getUndefValue(Location loc, OpBuilder &builder, Type type) override {
    return ub::PoisonOp::create(builder, loc, type, nullptr);
  }

  // Creates appropriate terminator for infinite loops.
  // We insert util.scf.unreachable to mark the unreachable code then
  // create a return with poison values (like upstream) to allow full CFG->SCF
  // transformation.
  FailureOr<Operation *> createUnreachableTerminator(Location loc,
                                                     OpBuilder &builder,
                                                     Region &region) override {
    auto callableOp = dyn_cast<CallableOpInterface>(region.getParentOp());
    if (!callableOp) {
      return emitError(loc) << "expected callable op (e.g. util.func or "
                               "util.initializer) as parent of region";
    }

    // Insert util.scf.unreachable to mark that this is an infinite loop.
    IREE::Util::SCFUnreachableOp::create(builder, loc, "infinite loop");

    // Return poisoned values. We don't use util.unreachable as that breaks the
    // lifting and instead we let canonicalizers fix things after we are fully
    // converted.
    return IREE::Util::ReturnOp::create(
               builder, loc,
               IREE::Util::SCFUnreachableOp::createPoisonValues(
                   builder, loc, callableOp.getResultTypes()))
        .getOperation();
  }
};

//===----------------------------------------------------------------------===//
// LiftCFGToSCFPass
//===----------------------------------------------------------------------===//

struct LiftCFGToSCFPass final : impl::LiftCFGToSCFPassBase<LiftCFGToSCFPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    // Process all callable ops (util.func, util.intializer, etc).
    // Note that we only walk the top-level ops so that we don't go into
    // executables.
    //
    // TODO(benvanik): see if we could do this in parallel by scoping more
    // narrow - not sure if there are function signature changes we want to
    // make.
    UtilToSCFInterface interface;
    bool changed = false;
    for (auto callableOp : moduleOp.getOps<mlir::CallableOpInterface>()) {
      // Skip if the region is null or empty (usually a declaration/extern).
      Region *region = callableOp.getCallableRegion();
      if (!region || region->empty()) {
        continue;
      }

      // Use analysis cache for dominance information.
      auto &domInfo = callableOp.getOperation() != moduleOp
                          ? getChildAnalysis<DominanceInfo>(callableOp)
                          : getAnalysis<DominanceInfo>();

      // Walk all operations (including nested) and transform their regions.
      // This includes the callable's own region.
      // We walk ops in post-order to process innermost regions first.
      auto visitor = [&](Operation *innerOp) -> WalkResult {
        for (Region &reg : innerOp->getRegions()) {
          if (reg.empty()) {
            continue;
          }
          // Transform CFG in each region and track if we made any changes.
          FailureOr<bool> regionChanged =
              transformCFGToSCF(reg, interface, domInfo);
          if (failed(regionChanged)) {
            innerOp->emitError()
                << "Failed to lift control flow to SCF. This may indicate "
                   "irreducible control flow (loops with multiple entry "
                   "points). Consider restructuring the code to use "
                   "single-entry loops";
            return WalkResult::interrupt();
          }
          changed |= *regionChanged;
        }
        return WalkResult::advance();
      };
      if (callableOp->walk<WalkOrder::PostOrder>(visitor).wasInterrupted()) {
        return signalPassFailure();
      }
    }

    // Preserve analyses if no changes were made.
    if (!changed) {
      markAllAnalysesPreserved();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Util
