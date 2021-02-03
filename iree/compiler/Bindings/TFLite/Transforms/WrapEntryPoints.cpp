// Copyright 2021 Google LLC
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

#include "llvm/ADT/STLExtras.h"
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
namespace TFLite {

// Wraps each model entry point in a "_tflite_xx" function that matches the
// expectations of the IREE TFLite C bindings.
class WrapEntryPointsPass
    : public PassWrapper<WrapEntryPointsPass, OperationPass<ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<StandardOpsDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    SmallVector<FuncOp, 4> entryFuncOps;
    for (auto funcOp : moduleOp.getOps<FuncOp>()) {
      if (funcOp.isPublic()) entryFuncOps.push_back(funcOp);
    }
    if (entryFuncOps.size() == 0) {
      moduleOp.emitError()
          << "no entry points found; the tflite bindings "
             "require exactly 1 entry point (function with public visibility)";
      signalPassFailure();
      return;
    } else if (entryFuncOps.size() > 1) {
      moduleOp.emitError()
          << "multiple entry points found; the tflite bindings require exactly "
             "1 entry point (function with public visibility)";
      signalPassFailure();
      return;
    }

    // Create a wrapper function for the entry point.
    auto entryFuncOp = entryFuncOps.front();
    auto wrapperFuncOp = createWrapperFunc(entryFuncOp);
    moduleOp.insert(Block::iterator(entryFuncOp), wrapperFuncOp);

    // TODO(#3968): the story around what we export is pretty messy. We really
    // need to switch this to being an op (iree.entry_point or iree.export)
    // instead of being based on visibility or iree.module.export.
    entryFuncOp.removeAttr("iree.module.export");
    entryFuncOp.setPrivate();
  }

 private:
  // Creates the corresponding wrapper function for the given entry point.
  // The wrapper function will contain the reflection metadata required at
  // runtime to get input/output tensor names, quantization parameters, etc.
  //
  // We do this by creating a new function just for the bindings and calling the
  // existing entry point. This allows us to support multiple binding schemes as
  // transforms from other bindings can also perform their own equivalent
  // wrapping.
  //
  // NOTE: today we only support a single entry point; with minor tweaks we
  // could fix this up to support multiple if we wanted.
  FuncOp createWrapperFunc(FuncOp entryFuncOp) {
    // NOTE: this is where we could change our signature to provide additional
    // values from the runtime bindings as may be required - like semaphores for
    // async behavior or cancellation.
    auto entryFuncType = entryFuncOp.getType();
    auto wrapperFuncType = entryFuncType;

    auto wrapperFuncOp =
        FuncOp::create(entryFuncOp.getLoc(), "_tflite_main", wrapperFuncType);
    wrapperFuncOp.setPublic();
    wrapperFuncOp.getOperation()->setAttr("iree.module.export",
                                          UnitAttr::get(&getContext()));
    wrapperFuncOp.getOperation()->setAttr("iree.abi.stub",
                                          UnitAttr::get(&getContext()));

    SmallVector<DictionaryAttr, 4> argAttrDict;
    entryFuncOp.getAllArgAttrs(argAttrDict);
    wrapperFuncOp.setAllArgAttrs(argAttrDict);
    SmallVector<DictionaryAttr, 4> resultAttrDict;
    entryFuncOp.getAllResultAttrs(resultAttrDict);
    wrapperFuncOp.setAllResultAttrs(resultAttrDict);

    populateReflectionAttrs(entryFuncOp, wrapperFuncOp);

    // Just call the entryFuncOp and return the results.
    // If we wanted to perform additional work here to invalidate cached shapes
    // from the shape support functions or validate the inputs we'd do that
    // here. Format conversion/decomposition (interleaved complex ->
    // deinterleaved, float <-> quantized conversions, etc) can also be inserted
    // such that other bindings that don't need such things aren't impacted.
    auto *entryBlock = wrapperFuncOp.addEntryBlock();
    auto entryBuilder = OpBuilder::atBlockBegin(entryBlock);
    auto results = entryBuilder.create<CallOp>(
        entryFuncOp.getLoc(), entryFuncOp,
        llvm::to_vector<4>(llvm::map_range(
            entryBlock->getArguments(),
            [](BlockArgument arg) { return static_cast<Value>(arg); })));
    entryBuilder.create<ReturnOp>(entryFuncOp.getLoc(), results.getResults());

    return wrapperFuncOp;
  }

  // Populates attributes on |wrapperFuncOp| to support runtime reflection like
  // IO tensor names and quantization information.
  void populateReflectionAttrs(FuncOp entryFuncOp, FuncOp wrapperFuncOp) {
    SmallVector<NamedAttribute, 4> attrs;
    attrs.push_back(buildIONamesAttr(entryFuncOp));
    // TODO(#3972): tfl.io.quant: quantization information.
    // TODO(#3978): tfl.io.types: tensor types (complex/strings/etc).
    auto reflectionAttr = DictionaryAttr::get(attrs, &getContext());
    wrapperFuncOp->setAttr("iree.reflection", reflectionAttr);
  }

  // Constructs an attribute containing all of the input and output identifiers:
  //   tfl.io.names=arg0;arg1;ret0;ret1
  //
  // Default names will be used if no iree.identifiers are set on the function.
  NamedAttribute buildIONamesAttr(FuncOp entryFuncOp) {
    SmallVector<std::string, 4> pieces;
    for (int i = 0; i < entryFuncOp.getNumArguments(); ++i) {
      StringRef identifier =
          entryFuncOp.getArgAttrOfType<StringAttr>(i, "iree.identifier")
              .getValue();
      if (identifier.empty()) {
        pieces.push_back("arg" + std::to_string(i));
      } else {
        pieces.push_back(identifier.str());
      }
    }
    for (int i = 0; i < entryFuncOp.getNumResults(); ++i) {
      StringRef identifier =
          entryFuncOp.getResultAttrOfType<StringAttr>(i, "iree.identifier")
              .getValue();
      if (identifier.empty()) {
        pieces.push_back("ret" + std::to_string(i));
      } else {
        pieces.push_back(identifier.str());
      }
    }
    return NamedAttribute{
        Identifier::get("tfl.io.names", &getContext()),
        StringAttr::get(llvm::join(pieces, ";"), &getContext())};
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createWrapEntryPointsPass() {
  return std::make_unique<WrapEntryPointsPass>();
}

static PassRegistration<WrapEntryPointsPass> pass(
    "iree-tflite-wrap-entry-points",
    "Wraps model entry points in functions compatible with the tflite "
    "bindings");

}  // namespace TFLite
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
