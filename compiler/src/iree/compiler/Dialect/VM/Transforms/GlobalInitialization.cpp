// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir::iree_compiler::IREE::VM {

#define GEN_PASS_DEF_GLOBALINITIALIZATIONPASS
#include "iree/compiler/Dialect/VM/Transforms/Passes.h.inc"

// Finds a function with |name| and returns it ready for appending.
// The returned op builder will be set at an insertion point where new
// operations can be added that are guaranteed to execute in the CFG. The
// caller must insert a return op at the insertion point when done.
static std::tuple<IREE::VM::FuncOp, OpBuilder>
appendOrCreateInitFuncOp(IREE::VM::ModuleOp moduleOp, StringRef name,
                         SymbolTable &symbolTable, OpBuilder &moduleBuilder) {
  IREE::VM::FuncOp funcOp = symbolTable.lookup<IREE::VM::FuncOp>(name);
  OpBuilder funcBuilder(moduleOp.getContext());
  if (!funcOp) {
    // Create a new empty function.
    funcOp =
        IREE::VM::FuncOp::create(moduleBuilder, moduleBuilder.getUnknownLoc(),
                                 name, moduleBuilder.getFunctionType({}, {}));
    funcBuilder = OpBuilder::atBlockEnd(funcOp.addEntryBlock());
    return std::make_tuple(funcOp, funcBuilder);
  }

  // Function already exists; we need to append to it. The function may have
  // an arbitrarily complex CFG and we need to route all returns to a new
  // block that will always execute before the initializer returns.

  // Create the target block we'll be inserting into.
  // It'll be empty and the caller must insert the terminator.
  auto *newBlock = funcOp.addBlock();

  // Find all extant return points and redirect them to the new block.
  auto returnOps = llvm::to_vector(funcOp.getOps<IREE::VM::ReturnOp>());
  for (auto returnOp :
       llvm::make_early_inc_range(funcOp.getOps<IREE::VM::ReturnOp>())) {
    OpBuilder builder(returnOp);
    IREE::VM::BranchOp::create(builder, returnOp.getLoc(), newBlock);
    returnOp.erase();
  }

  // Return the builder at the end of the new block.
  funcBuilder.setInsertionPointToEnd(newBlock);
  return std::make_tuple(funcOp, funcBuilder);
}

// Adds a vm.export for |funcOp| if there is not one already.
static void exportFuncIfNeeded(IREE::VM::ModuleOp moduleOp,
                               IREE::VM::FuncOp funcOp) {
  // Check for an existing export.
  for (auto exportOp : moduleOp.getOps<IREE::VM::ExportOp>()) {
    if (exportOp.getExportName() == funcOp.getName()) {
      // Already has an export.
      return;
    }
  }

  // Functions are private, exports are public.
  funcOp.setPrivate();

  // Add vm.export if needed.
  OpBuilder moduleBuilder(funcOp);
  IREE::VM::ExportOp::create(moduleBuilder, funcOp.getLoc(), funcOp);
}

// Updates the mutability of globals based on whether they are stored anywhere
// in the program. The mutability here is not for program analysis but because
// the runtime needs to allocate rwdata for the global instead of embedding it
// as a rodata constant.
static void fixupGlobalMutability(Operation *moduleOp,
                                  SymbolTable &symbolTable) {
  Explorer explorer(moduleOp, TraversalAction::SHALLOW);
  explorer.setOpInterfaceAction<mlir::FunctionOpInterface>(
      TraversalAction::RECURSE);
  explorer.initialize();
  SmallVector<Operation *> deadOps;
  explorer.forEachGlobal([&](const Explorer::GlobalInfo *globalInfo) {
    if (globalInfo->uses.empty())
      return;
    // TODO(benvanik): verify we want this behavior - we likely want to change
    // this to be mutable only if stores exist outside of initializers.
    //
    // If there are stores mark the global as mutable. We need to update all
    // of the loads if this changes anything.
    bool hasStores = !globalInfo->getStores().empty();
    bool didChange = globalInfo->op.isGlobalMutable() != hasStores;
    if (didChange) {
      globalInfo->op.setGlobalMutable(hasStores);
      for (auto loadOp : globalInfo->getLoads()) {
        loadOp.setGlobalImmutable(!hasStores);
      }
    }
  });
}

// Finds all global variables and moves their inital values/initializer calls
// into a single function. Relies on the inliner to later make the uber function
// better.
//
// Note that this may not generate ideal initialization behavior. For example,
// if there are 100 global refs of constant buffers this will lower 100
// individual initializers down to calls when clearly there should be only a
// single buffer allocated and sliced for all the globals. Once we are at this
// point in the lowering though we cannot know that so we rely on dialects
// providing their own initialization functions for those cases.
//
// TODO(benvanik): combine i32 initializers to store more efficiently.
class GlobalInitializationPass
    : public IREE::VM::impl::GlobalInitializationPassBase<
          GlobalInitializationPass> {
  void runOnOperation() override {
    IREE::VM::ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    // Create the __init and __deinit functions. They may be empty if there are
    // no globals but that's fine.
    OpBuilder moduleBuilder = OpBuilder::atBlockEnd(&moduleOp.getBlock());
    moduleBuilder.setInsertionPoint(moduleOp.getBlock().getTerminator());

    // Find/create the init/deinit functions.
    // We will erase them if they end up empty.
    IREE::VM::FuncOp initFuncOp;
    OpBuilder initBuilder(&getContext());
    std::tie(initFuncOp, initBuilder) = appendOrCreateInitFuncOp(
        moduleOp, "__init", symbolTable, moduleBuilder);
    IREE::VM::FuncOp deinitFuncOp;
    OpBuilder deinitBuilder(&getContext());
    std::tie(deinitFuncOp, deinitBuilder) = appendOrCreateInitFuncOp(
        moduleOp, "__deinit", symbolTable, moduleBuilder);

    // Build out the functions with logic from all globals and initializers.
    // We use a two-phase approach to ensure correct initialization order:
    // Phase 1: Initialize all globals with initial values in module order.
    // Phase 2: Execute all initializers in module order.
    //
    // This ensures that initializers can safely reference globals even if the
    // initializer appears before the global in module order, which can happen
    // after passes like CombineInitializersPass reorder operations.
    InlinerInterface inlinerInterface(&getContext());

    // Phase 1: Initialize all globals with initial values.
    // This ensures all globals reach a valid state before any initializers run.
    for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOpInterface>()) {
      if (isa<IREE::VM::RefType>(globalOp.getGlobalType())) {
        if (failed(appendRefInitialization(globalOp, initBuilder))) {
          globalOp.emitOpError() << "ref-type global unable to be initialized";
          return signalPassFailure();
        }
      } else {
        if (failed(appendPrimitiveInitialization(globalOp, initBuilder))) {
          globalOp.emitOpError() << "primitive global unable to be initialized";
          return signalPassFailure();
        }
      }
    }

    // Phase 2: Execute all initializers in module order.
    // Initializers can now safely reference globals initialized in phase 1.
    SmallVector<Operation *> deadOps;
    for (auto initializerOp : moduleOp.getOps<IREE::VM::InitializerOp>()) {
      if (failed(appendInitializer(initializerOp, inlinerInterface,
                                   initBuilder))) {
        initializerOp.emitOpError() << "unable to be initialized";
        return signalPassFailure();
      }
      deadOps.push_back(initializerOp);
      initBuilder.setInsertionPointToEnd(&initFuncOp.back());
    }
    for (auto *deadOp : deadOps) {
      deadOp->erase();
    }

    // Add returns to the initializers.
    IREE::VM::ReturnOp::create(initBuilder, initBuilder.getUnknownLoc());
    IREE::VM::ReturnOp::create(deinitBuilder, deinitBuilder.getUnknownLoc());

    // Correct mutability of all globals.
    fixupGlobalMutability(moduleOp, symbolTable);

    // If we didn't need to initialize anything then we can elide the functions
    // and otherwise we need to ensure they are exported.
    exportFuncIfNeeded(moduleOp, initFuncOp);
    exportFuncIfNeeded(moduleOp, deinitFuncOp);
  }

  LogicalResult
  appendPrimitiveInitialization(IREE::Util::GlobalOpInterface globalOp,
                                OpBuilder &builder) {
    auto initialValue = globalOp.getGlobalInitialValue();
    Value value = {};
    if (initialValue) {
      LogicalResult constResult = success();
      std::tie(constResult, value) =
          createConst(globalOp.getLoc(), initialValue, builder);
      if (failed(constResult)) {
        return globalOp.emitOpError()
               << "unable to create initializer constant for global";
      }
      globalOp.setGlobalInitialValue({});
    }
    if (!value) {
      // Globals are zero-initialized by default so we can just strip the
      // initial value/initializer and avoid the work entirely.
      return success();
    }
    return storePrimitiveGlobal(globalOp.getLoc(), globalOp.getGlobalName(),
                                value, builder);
  }

  // Returns {} if the constant is zero.
  std::pair<LogicalResult, Value> createConst(Location loc, Attribute value,
                                              OpBuilder &builder) {
    if (auto integerAttr = dyn_cast<IntegerAttr>(value)) {
      if (integerAttr.getValue().isZero()) {
        // Globals are zero-initialized by default.
        return {success(), {}};
      }
      switch (integerAttr.getType().getIntOrFloatBitWidth()) {
      case 32:
        return {success(),
                builder.createOrFold<IREE::VM::ConstI32Op>(loc, integerAttr)};
      case 64:
        return {success(),
                builder.createOrFold<IREE::VM::ConstI64Op>(loc, integerAttr)};
      default:
        return {failure(), {}};
      }
    } else if (auto floatAttr = dyn_cast<FloatAttr>(value)) {
      if (floatAttr.getValue().isZero()) {
        // Globals are zero-initialized by default.
        return {success(), {}};
      }
      switch (floatAttr.getType().getIntOrFloatBitWidth()) {
      case 32:
        return {success(),
                builder.createOrFold<IREE::VM::ConstF32Op>(loc, floatAttr)};
      case 64:
        return {success(),
                builder.createOrFold<IREE::VM::ConstF64Op>(loc, floatAttr)};
      default:
        return {failure(), {}};
      }
    }
    return {failure(), {}};
  }

  // Stores a value to a global; the global must be mutable.
  LogicalResult storePrimitiveGlobal(Location loc, StringRef symName,
                                     Value value, OpBuilder &builder) {
    if (auto integerType = dyn_cast<IntegerType>(value.getType())) {
      switch (integerType.getIntOrFloatBitWidth()) {
      case 32:
        IREE::VM::GlobalStoreI32Op::create(builder, loc, value, symName);
        return success();
      case 64:
        IREE::VM::GlobalStoreI64Op::create(builder, loc, value, symName);
        return success();
      default:
        return failure();
      }
    } else if (auto floatType = dyn_cast<FloatType>(value.getType())) {
      switch (floatType.getIntOrFloatBitWidth()) {
      case 32:
        IREE::VM::GlobalStoreF32Op::create(builder, loc, value, symName);
        return success();
      case 64:
        IREE::VM::GlobalStoreF64Op::create(builder, loc, value, symName);
        return success();
      default:
        return failure();
      }
    }
    return failure();
  }

  LogicalResult appendRefInitialization(IREE::Util::GlobalOpInterface globalOp,
                                        OpBuilder &builder) {
    // NOTE: nothing yet, though if we had attribute initialization we'd do it
    // here (for example, #vm.magic.initial.ref<foo>).
    return success();
  }

  LogicalResult appendInitializer(InitializerOp initializerOp,
                                  InlinerInterface &inlinerInterface,
                                  OpBuilder &builder) {
    auto result = mlir::inlineRegion(
        inlinerInterface, InlinerConfig{}.getCloneCallback(),
        &initializerOp.getBody(), builder.getInsertionBlock(),
        builder.getInsertionPoint(),
        /*inlinedOperands=*/ValueRange{},
        /*resultsToReplace=*/ValueRange{}, /*inlineLoc=*/std::nullopt,
        /*shouldCloneInlinedRegion=*/false);
    builder.setInsertionPointToEnd(builder.getInsertionBlock());
    return result;
  }
};

} // namespace mlir::iree_compiler::IREE::VM
