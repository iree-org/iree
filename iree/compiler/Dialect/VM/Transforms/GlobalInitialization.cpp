// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

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
// TODO(benvanik): add initializer functions to make dialect init possible.
// TODO(benvanik): combine i32 initializers to store more efficiently.
class GlobalInitializationPass
    : public PassWrapper<GlobalInitializationPass, OperationPass<ModuleOp>> {
 public:
  StringRef getArgument() const override {
    return "iree-vm-global-initialization";
  }

  StringRef getDescription() const override {
    return "Creates module-level global init/deinit functions";
  }

  void runOnOperation() override {
    // Create the __init and __deinit functions. They may be empty if there are
    // no globals but that's fine.
    OpBuilder moduleBuilder = OpBuilder::atBlockEnd(&getOperation().getBlock());
    moduleBuilder.setInsertionPoint(getOperation().getBlock().getTerminator());
    auto initFuncOp =
        moduleBuilder.create<FuncOp>(moduleBuilder.getUnknownLoc(), "__init",
                                     moduleBuilder.getFunctionType({}, {}));
    OpBuilder initBuilder = OpBuilder::atBlockEnd(initFuncOp.addEntryBlock());
    auto deinitFuncOp =
        moduleBuilder.create<FuncOp>(moduleBuilder.getUnknownLoc(), "__deinit",
                                     moduleBuilder.getFunctionType({}, {}));
    OpBuilder deinitBuilder =
        OpBuilder::atBlockEnd(deinitFuncOp.addEntryBlock());

    // Build out the functions with logic from all globals.
    // Note that the initialization order here is undefined (in that it's just
    // module op order). If we ever want to make this more deterministic we
    // could gather the ops, sort them (by some rule), and then build the
    // initialization function.
    for (auto &op : getOperation().getBlock().getOperations()) {
      if (auto globalOp = dyn_cast<GlobalRefOp>(op)) {
        if (failed(appendRefInitialization(globalOp, initBuilder))) {
          globalOp.emitOpError() << "unable to be initialized";
          return signalPassFailure();
        }
      } else if (auto globalOp = dyn_cast<VMGlobalOp>(op)) {
        if (failed(appendPrimitiveInitialization(globalOp, initBuilder))) {
          globalOp.emitOpError() << "unable to be initialized";
          return signalPassFailure();
        }
      }
    }

    initBuilder.create<ReturnOp>(initBuilder.getUnknownLoc());
    deinitBuilder.create<ReturnOp>(deinitBuilder.getUnknownLoc());

    // If we didn't need to initialize anything then we can elide the functions.
    if (initFuncOp.getBlocks().front().getOperations().size() > 1) {
      initFuncOp.setPrivate();
      moduleBuilder.create<ExportOp>(moduleBuilder.getUnknownLoc(), initFuncOp);
    } else {
      initFuncOp.erase();
    }
    if (deinitFuncOp.getBlocks().front().getOperations().size() > 1) {
      deinitFuncOp.setPrivate();
      moduleBuilder.create<ExportOp>(moduleBuilder.getUnknownLoc(),
                                     deinitFuncOp);
    } else {
      deinitFuncOp.erase();
    }
  }

 private:
  LogicalResult appendPrimitiveInitialization(VMGlobalOp globalOp,
                                              OpBuilder &builder) {
    auto initialValue =
        globalOp.getInitialValueAttr().getValueOr<Attribute>({});
    Value value = {};
    if (initialValue) {
      LogicalResult constResult = success();
      std::tie(constResult, value) =
          createConst(globalOp.getLoc(), initialValue, builder);
      if (failed(constResult)) {
        return globalOp.emitOpError()
               << "unable to create initializer constant for global";
      }
      globalOp.clearInitialValue();
    } else if (globalOp.getInitializerAttr().hasValue()) {
      auto callOp = builder.create<CallOp>(
          globalOp.getLoc(), globalOp.getInitializerAttr().getValue(),
          ArrayRef<Type>{globalOp.getStorageType()}, ArrayRef<Value>{});
      value = callOp.getResult(0);
      globalOp.clearInitializer();
    }
    if (!value) {
      // Globals are zero-initialized by default so we can just strip the
      // initial value/initializer and avoid the work entirely.
      return success();
    }
    globalOp.makeMutable();
    return storePrimitiveGlobal(globalOp.getLoc(), globalOp.getSymbolName(),
                                value, builder);
  }

  // Returns {} if the constant is zero.
  std::pair<LogicalResult, Value> createConst(Location loc, Attribute value,
                                              OpBuilder &builder) {
    if (auto integerAttr = value.dyn_cast<IntegerAttr>()) {
      if (integerAttr.getValue().isNullValue()) {
        // Globals are zero-initialized by default.
        return {success(), {}};
      }
      switch (integerAttr.getType().getIntOrFloatBitWidth()) {
        case 32:
          return {success(),
                  builder.createOrFold<ConstI32Op>(loc, integerAttr)};
        case 64:
          return {success(),
                  builder.createOrFold<ConstI64Op>(loc, integerAttr)};
        default:
          return {failure(), {}};
      }
    } else if (auto floatAttr = value.dyn_cast<FloatAttr>()) {
      if (floatAttr.getValue().isZero()) {
        // Globals are zero-initialized by default.
        return {success(), {}};
      }
      switch (floatAttr.getType().getIntOrFloatBitWidth()) {
        case 32:
          return {success(), builder.createOrFold<ConstF32Op>(loc, floatAttr)};
        case 64:
          return {success(), builder.createOrFold<ConstF64Op>(loc, floatAttr)};
        default:
          return {failure(), {}};
      }
    }
    return {failure(), {}};
  }

  // Stores a value to a global; the global must be mutable.
  LogicalResult storePrimitiveGlobal(Location loc, StringRef symName,
                                     Value value, OpBuilder &builder) {
    if (auto integerType = value.getType().dyn_cast<IntegerType>()) {
      switch (integerType.getIntOrFloatBitWidth()) {
        case 32:
          builder.create<GlobalStoreI32Op>(loc, value, symName);
          return success();
        case 64:
          builder.create<GlobalStoreI64Op>(loc, value, symName);
          return success();
        default:
          return failure();
      }
    } else if (auto floatType = value.getType().dyn_cast<FloatType>()) {
      switch (floatType.getIntOrFloatBitWidth()) {
        case 32:
          builder.create<GlobalStoreF32Op>(loc, value, symName);
          return success();
        case 64:
          builder.create<GlobalStoreF64Op>(loc, value, symName);
          return success();
        default:
          return failure();
      }
    }
    return failure();
  }

  LogicalResult appendRefInitialization(GlobalRefOp globalOp,
                                        OpBuilder &builder) {
    if (globalOp.initializer().hasValue()) {
      auto callOp = builder.create<CallOp>(
          globalOp.getLoc(), globalOp.initializerAttr(),
          ArrayRef<Type>{globalOp.type()}, ArrayRef<Value>{});
      builder.create<GlobalStoreRefOp>(globalOp.getLoc(), callOp.getResult(0),
                                       globalOp.sym_name());
      globalOp.clearInitializer();
      globalOp.makeMutable();
    }
    return success();
  }
};

std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createGlobalInitializationPass() {
  return std::make_unique<GlobalInitializationPass>();
}

static PassRegistration<GlobalInitializationPass> pass;

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
