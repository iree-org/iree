// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Bindings/Native/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace ABI {

// Maps a source type to the native ABI type.
static Type mapToABIType(Type type) {
  if (type.isa<TensorType>()) {
    return IREE::HAL::BufferViewType::get(type.getContext());
  }
  return type;
}

// Creates the corresponding wrapper function for the given import function.
static func::FuncOp createImportWrapperFunc(func::FuncOp importOp,
                                            FunctionType oldImportType,
                                            FunctionType newImportType,
                                            StringRef privateName) {
  // Create the internal wrapper function with the original import signature.
  auto wrapperOp =
      func::FuncOp::create(importOp.getLoc(), privateName, oldImportType);
  wrapperOp.setPrivate();

  // Copy arg/result attrs from the import op to the wrapper function.
  // We may want to remove them from the import but would need to filter.
  SmallVector<DictionaryAttr, 4> argAttrDict;
  importOp.getAllArgAttrs(argAttrDict);
  wrapperOp.setAllArgAttrs(argAttrDict);
  SmallVector<DictionaryAttr, 4> resultAttrDict;
  importOp.getAllResultAttrs(resultAttrDict);
  wrapperOp.setAllResultAttrs(resultAttrDict);

  auto *entryBlock = wrapperOp.addEntryBlock();
  auto entryBuilder = OpBuilder::atBlockBegin(entryBlock);

  // Marshal arguments.
  SmallVector<Value> arguments;
  for (auto arg : llvm::enumerate(entryBlock->getArguments())) {
    auto oldType = oldImportType.getInput(arg.index());
    auto newType = newImportType.getInput(arg.index());
    if (oldType.isa<TensorType>()) {
      // This is where we could perform type casting or in-place storage binding
      // if the user had any attrs specifying it.
      auto argLoc = arg.value().getLoc();
      auto exportOp = entryBuilder.create<IREE::HAL::TensorExportOp>(
          argLoc, newType, arg.value());
      arguments.push_back(exportOp.getTarget());
    } else {
      arguments.push_back(arg.value());
    }
  }

  // Make the call with the updated types.
  auto callOp =
      entryBuilder.create<func::CallOp>(importOp.getLoc(), importOp, arguments);

  // Marshal results.
  SmallVector<Value> results;
  for (auto result : llvm::enumerate(callOp.getResults())) {
    auto oldType = oldImportType.getResult(result.index());
    if (oldType.isa<TensorType>()) {
      results.push_back(entryBuilder.create<IREE::HAL::TensorImportOp>(
          importOp.getLoc(), oldType, result.value()));
    } else {
      results.push_back(result.value());
    }
  }

  entryBuilder.create<func::ReturnOp>(importOp.getLoc(), results);
  return wrapperOp;
}

// Updates |importOp| to use the native ABI and creates a wrapper function that
// preserves the original behavior. All callers will be updated to point at the
// new wrapper function.
static LogicalResult wrapImportFunc(IREE::ABI::InvocationModel invocationModel,
                                    mlir::ModuleOp moduleOp,
                                    func::FuncOp importOp,
                                    SymbolTable &symbolTable) {
  // Replace all existing calls to the import to instead call the wrapper.
  auto publicName = importOp.getName().str();
  auto privateName = "_" + publicName;
  auto privateNameAttr =
      mlir::StringAttr::get(importOp.getContext(), privateName);
  if (failed(symbolTable.replaceAllSymbolUses(importOp, privateNameAttr,
                                              moduleOp))) {
    return importOp.emitError() << "unknown symbol table op encountered; "
                                   "cannot fix up symbol names";
  }

  // Convert import signature types to those required by the binding ABI.
  //
  // NOTE: this is where we could change our signature to provide additional
  // values from the runtime bindings as may be required - like semaphores for
  // async behavior or cancellation.
  auto oldImportType = importOp.getFunctionType();
  SmallVector<Type> inputTypes;
  for (auto oldType : oldImportType.getInputs()) {
    inputTypes.push_back(mapToABIType(oldType));
  }
  SmallVector<Type> resultTypes;
  for (auto oldType : oldImportType.getResults()) {
    resultTypes.push_back(mapToABIType(oldType));
  }
  auto newImportType =
      FunctionType::get(importOp.getContext(), inputTypes, resultTypes);

  // Update the import to the new type and mark it as being converted so we
  // don't try to convert it again.
  importOp.setType(newImportType);
  importOp->setAttr("iree.abi.stub", UnitAttr::get(importOp.getContext()));

  // Create the wrapper function that matches the original internal types but
  // calls out to the updated import using ABI types.
  auto wrapperOp = createImportWrapperFunc(importOp, oldImportType,
                                           newImportType, privateName);
  if (!wrapperOp) return failure();
  moduleOp.insert(++Block::iterator(importOp), wrapperOp);

  return success();
}

// Populates attributes on |wrapperOp| to support runtime reflection.
// These are attached to the exported function and can be queried at runtime
// with iree_vm_function_lookup_attr_by_name.
static void populateReflectionAttrs(IREE::ABI::InvocationModel invocationModel,
                                    func::FuncOp exportOp,
                                    func::FuncOp wrapperOp) {
  auto *context = exportOp.getContext();
  SmallVector<NamedAttribute, 4> attrs;

  if (auto abiAttr = exportOp->getAttr("iree.abi")) {
    attrs.emplace_back(StringAttr::get(context, "iree.abi"), abiAttr);
  }

  switch (invocationModel) {
    default:
    case IREE::ABI::InvocationModel::Sync:
      break;
    case IREE::ABI::InvocationModel::CoarseFences:
      attrs.emplace_back(StringAttr::get(context, "iree.abi.model"),
                         StringAttr::get(context, "coarse-fences"));
      break;
  }

  if (!attrs.empty()) {
    auto reflectionAttr = DictionaryAttr::get(context, attrs);
    wrapperOp->setAttr("iree.reflection", reflectionAttr);
  }
}

// Creates the corresponding wrapper function for the given export function.
static func::FuncOp createExportWrapperFunc(
    IREE::ABI::InvocationModel invocationModel, func::FuncOp exportOp,
    StringRef publicName) {
  // Copy arg/result attrs from the import op to the wrapper function.
  // We may want to remove them from the import but would need to filter.
  SmallVector<DictionaryAttr, 4> argAttrDict;
  exportOp.getAllArgAttrs(argAttrDict);
  SmallVector<DictionaryAttr, 4> resultAttrDict;
  exportOp.getAllResultAttrs(resultAttrDict);

  // Convert argument types to those required by the binding ABI.
  //
  // NOTE: this is where we could change our signature to provide additional
  // values from the runtime bindings as may be required - like semaphores for
  // async behavior or cancellation.
  auto oldExportType = exportOp.getFunctionType();
  SmallVector<Type> inputTypes;
  auto fenceType = IREE::HAL::FenceType::get(exportOp.getContext());
  for (auto oldType : oldExportType.getInputs()) {
    inputTypes.push_back(mapToABIType(oldType));
  }
  switch (invocationModel) {
    default:
    case IREE::ABI::InvocationModel::Sync:
      break;
    case IREE::ABI::InvocationModel::CoarseFences:
      inputTypes.push_back(fenceType);  // wait
      inputTypes.push_back(fenceType);  // signal
      argAttrDict.push_back(nullptr);   // wait
      argAttrDict.push_back(nullptr);   // signal
      break;
  }
  SmallVector<Type> resultTypes;
  for (auto oldType : oldExportType.getResults()) {
    resultTypes.push_back(mapToABIType(oldType));
  }
  auto newExportType =
      FunctionType::get(exportOp.getContext(), inputTypes, resultTypes);

  // Update the import to the new type and mark it as being converted so we
  // don't try to convert it again.
  auto wrapperOp =
      func::FuncOp::create(exportOp.getLoc(), publicName, newExportType);
  wrapperOp.setPublic();
  wrapperOp->setAttr("iree.abi.stub", UnitAttr::get(exportOp.getContext()));
  wrapperOp.setAllArgAttrs(argAttrDict);
  wrapperOp.setAllResultAttrs(resultAttrDict);

  // Populate the reflection attrs based on the original types.
  populateReflectionAttrs(invocationModel, exportOp, wrapperOp);

  auto *entryBlock = wrapperOp.addEntryBlock();
  auto entryBuilder = OpBuilder::atBlockBegin(entryBlock);

  // Build a map of result value to the argument that has its backing storage.
  SmallVector<Value> resultStorages;
  resultStorages.resize(resultTypes.size());
  for (unsigned i = 0; i < exportOp.getNumArguments(); ++i) {
    auto outputAttr =
        exportOp.getArgAttrOfType<IntegerAttr>(i, "iree.abi.output");
    if (!outputAttr) continue;
    // Today all outputs need to be a !hal.buffer - we could change this
    // in the future to be something more generalized.
    auto storageArg = entryBlock->getArgument(i);
    if (!storageArg.getType().isa<IREE::HAL::BufferType>()) {
      exportOp.emitError() << "storage argument " << i
                           << " has an invalid type " << storageArg.getType()
                           << "; must be a !hal.buffer";
      return {};
    }
    resultStorages[outputAttr.getInt()] = storageArg;
  }

  // Build a map of each I/O argument to the fence that covers them.
  // TODO(benvanik): actually support a map; for now we just handle the 1:M
  // coarse mode where all inputs are covered by a single wait fence and all
  // outputs are covered by a single signal fence.
  Value waitFence;
  Value signalFence;
  switch (invocationModel) {
    default:
    case IREE::ABI::InvocationModel::Sync:
      break;
    case IREE::ABI::InvocationModel::CoarseFences:
      waitFence = entryBlock->getArgument(entryBlock->getNumArguments() - 2);
      signalFence = entryBlock->getArgument(entryBlock->getNumArguments() - 1);
      break;
  }

  // Marshal arguments.
  SmallVector<Value> arguments;
  for (auto arg : llvm::enumerate(
           entryBlock->getArguments().slice(0, oldExportType.getNumInputs()))) {
    auto oldType = oldExportType.getInput(arg.index());
    if (oldType.isa<TensorType>()) {
      auto argLoc = arg.value().getLoc();
      auto importOp = entryBuilder.create<IREE::HAL::TensorImportOp>(
          argLoc, oldType, arg.value(), waitFence);
      arguments.push_back(importOp.getTarget());
    } else {
      arguments.push_back(arg.value());
    }
  }

  // Make the call with the original types.
  auto callOp =
      entryBuilder.create<func::CallOp>(exportOp.getLoc(), exportOp, arguments);
  auto asyncResults = llvm::to_vector(callOp.getResults());

  // Insert a barrier if requested - all tensors will be calculated and the
  // fence will be signaled. Note that even if there are no tensor results we
  // need to signal the fence.
  if (signalFence) {
    SmallVector<Value> asyncTensors;
    for (auto result : asyncResults) {
      if (result.getType().isa<TensorType>()) asyncTensors.push_back(result);
    }
    if (asyncTensors.empty()) {
      // TODO(benvanik): maybe use a global timeline? global stores may not
      // have completed by now in cases where the user wants to loop back.
      entryBuilder.create<IREE::HAL::FenceSignalOp>(exportOp.getLoc(),
                                                    signalFence);
    } else {
      auto barrierOp = entryBuilder.create<IREE::HAL::TensorBarrierOp>(
          exportOp.getLoc(), asyncTensors, signalFence);
      asyncResults = llvm::to_vector(barrierOp.getResults());
    }
  }

  // Marshal results.
  SmallVector<Value> results;
  for (auto [resultIndex, result] : llvm::enumerate(asyncResults)) {
    auto oldType = oldExportType.getResult(resultIndex);
    auto newType = newExportType.getResult(resultIndex);
    if (oldType.isa<TensorType>()) {
      auto dynamicDims = IREE::Util::buildDynamicDimsForValue(
          result.getLoc(), result, entryBuilder);
      results.push_back(entryBuilder.create<IREE::HAL::TensorExportOp>(
          result.getLoc(), newType, result, TypeAttr::get(result.getType()),
          dynamicDims, resultStorages[resultIndex]));
    } else {
      results.push_back(result);
    }
  }

  entryBuilder.create<func::ReturnOp>(exportOp.getLoc(), results);
  return wrapperOp;
}

// Replaces |exportOp| with a wrapper function that exports the native ABI.
// The original function will be made private and be renamed.
// This allows us to support multiple binding schemes as transforms from other
// bindings can also perform their own equivalent wrapping.
static LogicalResult wrapExportFunc(IREE::ABI::InvocationModel invocationModel,
                                    mlir::ModuleOp moduleOp,
                                    func::FuncOp exportOp,
                                    SymbolTable &symbolTable) {
  // Rename the original function so that our wrapper can use the original
  // name in its public definition.
  auto publicName = exportOp.getName().str();
  auto privateName = "_" + publicName;
  auto privateNameAttr =
      mlir::StringAttr::get(exportOp.getContext(), privateName);
  if (failed(symbolTable.replaceAllSymbolUses(exportOp, privateNameAttr,
                                              moduleOp))) {
    return exportOp.emitError() << "unknown symbol table op encountered; "
                                   "cannot fix up symbol names";
  }
  exportOp.setName(privateNameAttr);
  exportOp.setPrivate();

  // Create the wrapper function that conforms to the IREE native ABI and
  // marshals arguments/results to the original function.
  auto wrapperOp =
      createExportWrapperFunc(invocationModel, exportOp, publicName);
  if (!wrapperOp) return failure();
  moduleOp.insert(Block::iterator(exportOp), wrapperOp);

  return success();
}

// Wraps all entry points in a function that is compatible with the
// expected invocation semantics of bindings following the native IREE ABI.
// Imports are also handled as they are entry points in another module.
class WrapEntryPointsPass
    : public PassWrapper<WrapEntryPointsPass, OperationPass<ModuleOp>> {
 public:
  WrapEntryPointsPass() = default;
  WrapEntryPointsPass(const WrapEntryPointsPass &pass) {}
  WrapEntryPointsPass(IREE::ABI::InvocationModel invocationModel) {
    this->invocationModel = invocationModel;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, mlir::arith::ArithDialect,
                    mlir::tensor::TensorDialect, IREE::HAL::HALDialect>();
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

    // Gather functions that need wrapping.
    SmallVector<func::FuncOp> importOps;
    SmallVector<func::FuncOp> exportOps;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      // Ignore functions already marked as having their ABI goo handled.
      if (funcOp->hasAttr("iree.abi.stub")) continue;
      if (funcOp.isExternal()) {
        // Imported function.
        importOps.push_back(funcOp);
      } else if (funcOp.isPublic()) {
        // Exported function.
        exportOps.push_back(funcOp);
      }
    }

    SymbolTable symbolTable(moduleOp);

    // Create a wrapper function for each imported function.
    // This will preserve the internal types (tensors/etc) but change the import
    // to taking the ABI types and rewrite calls.
    for (auto importOp : importOps) {
      if (failed(wrapImportFunc(invocationModel, moduleOp, importOp,
                                symbolTable))) {
        return signalPassFailure();
      }
    }

    // Create a wrapper function for each exported function.
    // This will change the export to taking the ABI types and preserve the
    // internal types.
    for (auto exportOp : exportOps) {
      if (failed(wrapExportFunc(invocationModel, moduleOp, exportOp,
                                symbolTable))) {
        return signalPassFailure();
      }
    }
  }

 private:
  Option<InvocationModel> invocationModel{
      *this,
      "invocation-model",
      llvm::cl::desc("Specifies the execution model used for invocations."),
      llvm::cl::init(IREE::ABI::InvocationModel::Sync),
      llvm::cl::values(
          clEnumValN(IREE::ABI::InvocationModel::Sync, "sync",
                     "Fully synchronous behavior with no fences."),
          clEnumValN(IREE::ABI::InvocationModel::CoarseFences, "coarse-fences",
                     "Exposes one wait fence for all inputs and one signal "
                     "fence for all outputs.")),
  };
};

std::unique_ptr<OperationPass<ModuleOp>> createWrapEntryPointsPass(
    IREE::ABI::InvocationModel invocationModel) {
  return std::make_unique<WrapEntryPointsPass>(invocationModel);
}

static PassRegistration<WrapEntryPointsPass> pass;

}  // namespace ABI
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
