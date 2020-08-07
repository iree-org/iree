// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
      moduleBuilder.create<ExportOp>(moduleBuilder.getUnknownLoc(), initFuncOp);
    } else {
      initFuncOp.erase();
    }
    if (deinitFuncOp.getBlocks().front().getOperations().size() > 1) {
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
    if (auto intValue = value.dyn_cast<IntegerAttr>()) {
      if (intValue.getValue().isNullValue()) {
        // Globals are zero-initialized by default.
        return {success(), {}};
      }
      switch (intValue.getValue().getBitWidth()) {
        case 32:
          return {success(), builder.createOrFold<ConstI32Op>(loc, intValue)};
        case 64:
          return {success(), builder.createOrFold<ConstI64Op>(loc, intValue)};
        default:
          return {failure(), {}};
      }
    }
    return {failure(), {}};
  }

  // Stores a value to a global; the global must be mutable.
  LogicalResult storePrimitiveGlobal(Location loc, StringRef symName,
                                     Value value, OpBuilder &builder) {
    if (auto intType = value.getType().dyn_cast<IntegerType>()) {
      switch (intType.getIntOrFloatBitWidth()) {
        case 32:
          builder.create<GlobalStoreI32Op>(loc, value, symName);
          return success();
        case 64:
          builder.create<GlobalStoreI64Op>(loc, value, symName);
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

std::unique_ptr<OperationPass<ModuleOp>> createGlobalInitializationPass() {
  return std::make_unique<GlobalInitializationPass>();
}

static PassRegistration<GlobalInitializationPass> pass(
    "iree-vm-global-initialization",
    "Creates module-level global init/deinit functions");

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
