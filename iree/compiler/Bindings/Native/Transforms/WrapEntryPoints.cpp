// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace ABI {

// Wraps all entry points in a function that is compatible with the
// expected invocation semantics of bindings following the native IREE ABI.
class WrapEntryPointsPass
    : public PassWrapper<WrapEntryPointsPass, OperationPass<ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<StandardOpsDialect, mlir::arith::ArithmeticDialect,
                    IREE::HAL::HALDialect,
                    // TODO: memref is here because the **tensor** dim op was
                    // moved there for some reason. When that goes away we can
                    // drop this dependency.
                    memref::MemRefDialect>();
  }

  StringRef getArgument() const override {
    return "iree-abi-wrap-entry-points";
  }

  StringRef getDescription() const override {
    return "Wraps all entry points in a function that is compatible with the "
           "expected invocation semantics of bindings following the native "
           "IREE ABI.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    SmallVector<FuncOp, 4> entryFuncOps;
    for (auto funcOp : moduleOp.getOps<FuncOp>()) {
      if (funcOp.isPublic() && !funcOp->hasAttr("iree.abi.stub")) {
        entryFuncOps.push_back(funcOp);
      }
    }

    // Create a wrapper function for each entry point.
    for (auto entryFuncOp : entryFuncOps) {
      // Rename the original function so that our wrapper can use the original
      // name in its public definition.
      auto publicName = entryFuncOp.getName().str();
      auto privateName = "_" + publicName;
      mlir::StringAttr privateNameAttr =
          mlir::StringAttr::get(entryFuncOp.getContext(), privateName);
      entryFuncOp.setName(privateNameAttr);
      entryFuncOp.setPrivate();

      // Create the wrapper function that conforms to the IREE native ABI and
      // marshals arguments/results to the original function.
      auto wrapperFuncOp = createWrapperFunc(entryFuncOp);
      wrapperFuncOp.setPublic();
      wrapperFuncOp.setName(
          mlir::StringAttr::get(entryFuncOp.getContext(), publicName));
      moduleOp.insert(Block::iterator(entryFuncOp), wrapperFuncOp);

      wrapperFuncOp.getOperation()->setAttr("iree.abi.stub",
                                            UnitAttr::get(&getContext()));
    }
  }

 private:
  Type mapToABIType(Type type) {
    if (type.isa<TensorType>()) {
      return IREE::HAL::BufferViewType::get(type.getContext());
    }
    return type;
  }

  // Creates the corresponding wrapper function for the given entry point.
  //
  // We do this by creating a new function just for the bindings and calling the
  // existing entry point. This allows us to support multiple binding schemes as
  // transforms from other bindings can also perform their own equivalent
  // wrapping.
  //
  // NOTE: today we only support a single entry point; with minor tweaks we
  // could fix this up to support multiple if we wanted.
  FuncOp createWrapperFunc(FuncOp entryFuncOp) {
    // Convert argument types to those required by the binding ABI.
    //
    // NOTE: this is where we could change our signature to provide additional
    // values from the runtime bindings as may be required - like semaphores for
    // async behavior or cancellation.
    auto entryFuncType = entryFuncOp.getType();
    SmallVector<Type> inputTypes;
    for (auto oldType : entryFuncType.getInputs()) {
      inputTypes.push_back(mapToABIType(oldType));
    }
    SmallVector<Type> resultTypes;
    for (auto oldType : entryFuncType.getResults()) {
      resultTypes.push_back(mapToABIType(oldType));
    }
    auto wrapperFuncType =
        FunctionType::get(entryFuncOp.getContext(), inputTypes, resultTypes);

    auto wrapperFuncOp = FuncOp::create(entryFuncOp.getLoc(),
                                        entryFuncOp.getName(), wrapperFuncType);

    SmallVector<DictionaryAttr, 4> argAttrDict;
    entryFuncOp.getAllArgAttrs(argAttrDict);
    wrapperFuncOp.setAllArgAttrs(argAttrDict);
    SmallVector<DictionaryAttr, 4> resultAttrDict;
    entryFuncOp.getAllResultAttrs(resultAttrDict);
    wrapperFuncOp.setAllResultAttrs(resultAttrDict);

    populateReflectionAttrs(entryFuncOp, wrapperFuncOp);

    auto *entryBlock = wrapperFuncOp.addEntryBlock();
    auto entryBuilder = OpBuilder::atBlockBegin(entryBlock);

    // Marshal arguments.
    SmallVector<Value> arguments;
    for (auto arg : llvm::enumerate(entryBlock->getArguments())) {
      auto oldType = entryFuncType.getInput(arg.index());
      if (oldType.isa<TensorType>()) {
        arguments.push_back(entryBuilder.create<IREE::HAL::TensorCastOp>(
            entryFuncOp.getLoc(), oldType, arg.value()));
      } else {
        arguments.push_back(arg.value());
      }
    }

    // Make the call with the original types.
    auto callOp = entryBuilder.create<CallOp>(entryFuncOp.getLoc(), entryFuncOp,
                                              arguments);

    // Marshal results.
    SmallVector<Value> results;
    for (auto result : llvm::enumerate(callOp.getResults())) {
      auto oldType = entryFuncType.getResult(result.index());
      auto newType = wrapperFuncType.getResult(result.index());
      if (oldType.isa<TensorType>()) {
        results.push_back(entryBuilder.createOrFold<IREE::HAL::TensorCastOp>(
            entryFuncOp.getLoc(), newType, result.value()));
      } else {
        results.push_back(result.value());
      }
    }
    entryBuilder.create<ReturnOp>(entryFuncOp.getLoc(), results);

    return wrapperFuncOp;
  }

  // Populates attributes on |wrapperFuncOp| to support runtime reflection.
  void populateReflectionAttrs(FuncOp entryFuncOp, FuncOp wrapperFuncOp) {
    SmallVector<NamedAttribute, 4> attrs;
    auto abiAttr = entryFuncOp->getAttr("iree.abi");
    if (abiAttr) {
      attrs.emplace_back(StringAttr::get(entryFuncOp.getContext(), "iree.abi"),
                         abiAttr);
    }
    if (!attrs.empty()) {
      auto reflectionAttr = DictionaryAttr::get(&getContext(), attrs);
      wrapperFuncOp->setAttr("iree.reflection", reflectionAttr);
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createWrapEntryPointsPass() {
  return std::make_unique<WrapEntryPointsPass>();
}

static PassRegistration<WrapEntryPointsPass> pass;

}  // namespace ABI
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
