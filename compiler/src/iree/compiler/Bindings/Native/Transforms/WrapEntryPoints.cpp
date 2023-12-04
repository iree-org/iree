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

// Returns the invocation model specified on |op| or the |defaultModel|.
static IREE::ABI::InvocationModel
getInvocationModel(Operation *op, IREE::ABI::InvocationModel defaultModel) {
  auto modelAttr = op->getAttrOfType<StringAttr>("iree.abi.model");
  if (!modelAttr)
    return defaultModel;
  if (modelAttr == "coarse-fences") {
    return IREE::ABI::InvocationModel::CoarseFences;
  } else {
    return IREE::ABI::InvocationModel::Sync;
  }
}

// Maps a source type to the native ABI type.
static Type mapToABIType(Type type) {
  if (llvm::isa<TensorType>(type)) {
    return IREE::HAL::BufferViewType::get(type.getContext());
  }
  return type;
}

// Removes all ABI attrs handled by this pass from all dictionaries.
static void stripABIAttrs(SmallVectorImpl<DictionaryAttr> &allAttrs) {
  for (auto &attrDict : allAttrs) {
    SmallVector<NamedAttribute> attrs;
    attrs.reserve(attrDict.size());
    for (auto attr : attrDict) {
      // TODO(benvanik): faster lookup.
      if (attr.getName() != "iree.abi.output" &&
          attr.getName() != "iree.abi.encoding") {
        attrs.push_back(attr);
      }
    }
    attrDict = DictionaryAttr::get(attrDict.getContext(), attrs);
  }
}

// Creates the corresponding wrapper function for the given import function.
static func::FuncOp
createImportWrapperFunc(IREE::ABI::InvocationModel invocationModel,
                        func::FuncOp importOp, FunctionType oldImportType,
                        FunctionType newImportType, StringRef privateName) {
  // Create the internal wrapper function with the original import signature.
  auto wrapperOp =
      func::FuncOp::create(importOp.getLoc(), privateName, oldImportType);
  wrapperOp.setPrivate();

  // Copy arg/result attrs from the import op to the wrapper function.
  // We may want to remove them from the import but would need to filter.
  SmallVector<DictionaryAttr> argAttrDict;
  importOp.getAllArgAttrs(argAttrDict);
  stripABIAttrs(argAttrDict);
  wrapperOp.setAllArgAttrs(argAttrDict);
  SmallVector<DictionaryAttr> resultAttrDict;
  importOp.getAllResultAttrs(resultAttrDict);
  stripABIAttrs(resultAttrDict);
  wrapperOp.setAllResultAttrs(resultAttrDict);
  switch (invocationModel) {
  default:
  case IREE::ABI::InvocationModel::Sync:
    break;
  case IREE::ABI::InvocationModel::CoarseFences:
    argAttrDict.push_back(nullptr); // wait
    argAttrDict.push_back(nullptr); // signal
    break;
  }

  // Update the import type and propagate back the attributes we may have
  // modified above.
  importOp.setType(newImportType);
  importOp.setAllArgAttrs(argAttrDict);
  importOp.setAllResultAttrs(resultAttrDict);

  auto *entryBlock = wrapperOp.addEntryBlock();
  auto entryBuilder = OpBuilder::atBlockBegin(entryBlock);

  // Gather tensor arguments we may need to handle specially.
  SmallVector<Value> entryArgs = {entryBlock->getArguments().begin(),
                                  entryBlock->getArguments().end()};
  SmallVector<size_t> tensorArgIndices;
  SmallVector<Value> tensorArgs;
  for (auto [argIndex, arg] : llvm::enumerate(entryArgs)) {
    auto oldType = oldImportType.getInput(argIndex);
    if (llvm::isa<TensorType>(oldType)) {
      tensorArgIndices.push_back(argIndex);
      tensorArgs.push_back(arg);
    }
  }

  // Side-effecting imports need to have a host-wait inserted on them today.
  // We could add more configuration options here but for now we require that
  // users mark their functions 'nosideeffects' to avoid the host wait.
  const bool hasSideEffects = !importOp->hasAttr("nosideeffects");

  // When running async we insert a barrier on tensor arguments and attach that
  // to the fence we pass to the import for waiting. We'll also allocate the
  // signal fence that the import must signal when the returned tensors are
  // ready.
  Value waitFence;
  Value signalFence;
  switch (invocationModel) {
  default:
  case IREE::ABI::InvocationModel::Sync:
    // No fences.
    break;
  case IREE::ABI::InvocationModel::CoarseFences: {
    // HACK: this is relying on the fact that there's only one HAL device.
    // We should instead have a way of creating fences on the device that
    // is used to produce the tensors we're wrapping.
    auto device =
        entryBuilder.create<IREE::HAL::ExSharedDeviceOp>(importOp.getLoc());

    // When exporting a fence we need to put a barrier between the rest of the
    // program and the tensors consumed by the import.
    if (tensorArgs.empty()) {
      // No tensors passed to the import - pass in an immediate signal.
      waitFence = entryBuilder.create<IREE::Util::NullOp>(
          importOp.getLoc(), entryBuilder.getType<IREE::HAL::FenceType>());
    } else {
      waitFence = entryBuilder.create<IREE::HAL::FenceCreateOp>(
          importOp.getLoc(), entryBuilder.getType<IREE::HAL::FenceType>(),
          device, IREE::HAL::FenceFlagBitfield::None);
      auto barrierOp = entryBuilder.create<IREE::HAL::TensorBarrierOp>(
          importOp.getLoc(), tensorArgs, waitFence);
      for (auto [argIndex, readyArg] :
           llvm::zip_equal(tensorArgIndices, barrierOp.getResults())) {
        entryArgs[argIndex] = readyArg;
      }
    }

    // When the import produces resources we need to pass in a fence it can
    // signal when execution is ready.
    // TODO(benvanik): always pass in a signal fence? could be useful if we
    // want to allow for async work using fences that's not device-related.
    const bool haveTensorResults =
        llvm::any_of(oldImportType.getResults(),
                     [](Type type) { return llvm::isa<TensorType>(type); });
    if (!haveTensorResults && !hasSideEffects) {
      // No tensors returned from import - pass in an immediate signal.
      signalFence = entryBuilder.create<IREE::Util::NullOp>(
          importOp.getLoc(), entryBuilder.getType<IREE::HAL::FenceType>());
    } else {
      signalFence = entryBuilder.create<IREE::HAL::FenceCreateOp>(
          importOp.getLoc(), entryBuilder.getType<IREE::HAL::FenceType>(),
          device, IREE::HAL::FenceFlagBitfield::None);
    }
    break;
  }
  }

  // Marshal arguments.
  SmallVector<Value> arguments;
  for (auto [argIndex, arg] : llvm::enumerate(entryArgs)) {
    auto oldType = oldImportType.getInput(argIndex);
    auto newType = newImportType.getInput(argIndex);
    if (llvm::isa<TensorType>(oldType)) {
      // This is where we could perform type casting or in-place storage binding
      // if the user had any attrs specifying it.
      // NOTE: we insert a barrier on this above if needed so that the wait
      // fence will be signaled when the tensor is ready for consumption by the
      // import.
      auto encoding =
          importOp.getArgAttrOfType<TypeAttr>(argIndex, "iree.abi.encoding");
      auto exportOp = entryBuilder.create<IREE::HAL::TensorExportOp>(
          arg.getLoc(), newType, arg,
          encoding ? encoding : TypeAttr::get(oldType), /*name=*/nullptr);
      arguments.push_back(exportOp.getTarget());
    } else {
      arguments.push_back(arg);
    }
  }
  if (waitFence)
    arguments.push_back(waitFence);
  if (signalFence)
    arguments.push_back(signalFence);

  // Make the call with the updated types.
  auto callOp =
      entryBuilder.create<func::CallOp>(importOp.getLoc(), importOp, arguments);

  // If the call has side-effects then we need to wait on its signal fence on
  // the host. This is because they may have launched a thread of their own to
  // perform work that we can't track.
  if (hasSideEffects && signalFence) {
    auto timeoutMillis =
        entryBuilder.create<arith::ConstantIntOp>(importOp.getLoc(), -1, 32);
    entryBuilder.create<IREE::HAL::FenceAwaitOp>(importOp.getLoc(),
                                                 entryBuilder.getI32Type(),
                                                 timeoutMillis, signalFence);
  }

  // Marshal results.
  SmallVector<Value> results;
  for (auto [resultIndex, result] : llvm::enumerate(callOp.getResults())) {
    auto oldType = oldImportType.getResult(resultIndex);
    if (llvm::isa<TensorType>(oldType)) {
      // NOTE: we set the import pending on the signal fence from the import
      // indicating when the returned tensor is ready for consumption by the
      // program.
      auto encoding = importOp.getResultAttrOfType<TypeAttr>(
          resultIndex, "iree.abi.encoding");
      results.push_back(entryBuilder.create<IREE::HAL::TensorImportOp>(
          importOp.getLoc(), oldType, result,
          encoding ? encoding : TypeAttr::get(oldType), signalFence,
          /*name=*/nullptr));
    } else {
      results.push_back(result);
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
  auto oldImportType = importOp.getFunctionType();
  SmallVector<Type> inputTypes;
  for (auto oldType : oldImportType.getInputs()) {
    inputTypes.push_back(mapToABIType(oldType));
  }
  auto fenceType = IREE::HAL::FenceType::get(importOp.getContext());
  switch (invocationModel) {
  default:
  case IREE::ABI::InvocationModel::Sync:
    break;
  case IREE::ABI::InvocationModel::CoarseFences:
    inputTypes.push_back(fenceType); // wait
    inputTypes.push_back(fenceType); // signal
    break;
  }
  SmallVector<Type> resultTypes;
  for (auto oldType : oldImportType.getResults()) {
    resultTypes.push_back(mapToABIType(oldType));
  }
  auto newImportType =
      FunctionType::get(importOp.getContext(), inputTypes, resultTypes);

  // Create the wrapper function that matches the original internal types but
  // calls out to the updated import using ABI types.
  auto wrapperOp = createImportWrapperFunc(
      invocationModel, importOp, oldImportType, newImportType, privateName);
  if (!wrapperOp)
    return failure();
  moduleOp.insert(++Block::iterator(importOp), wrapperOp);

  // Update the import to the new type and mark it as being converted so we
  // don't try to convert it again.
  importOp->setAttr("iree.abi.stub", UnitAttr::get(importOp.getContext()));

  return success();
}

static StringAttr getNameFromDictAttr(DictionaryAttr attr) {
  return attr ? attr.getAs<StringAttr>("iree.abi.name") : nullptr;
}

static StringAttr inferArgumentName(MLIRContext *context, int index,
                                    DictionaryAttr attrs) {
  if (auto attrName = getNameFromDictAttr(attrs))
    return attrName;
  return StringAttr::get(context, "input" + std::to_string(index));
}

static StringAttr inferResultName(MLIRContext *context, int index,
                                  DictionaryAttr attrs) {
  if (auto attrName = getNameFromDictAttr(attrs))
    return attrName;
  return StringAttr::get(context, "output" + std::to_string(index));
}

static DictionaryAttr getIOAttr(ArrayAttr allAttrs, unsigned i) {
  if (!allAttrs)
    return nullptr;
  return cast_or_null<DictionaryAttr>(allAttrs.getValue()[i]);
}

static void formatIOAttr(DictionaryAttr attrs, llvm::raw_ostream &os) {
  if (!attrs || attrs.empty())
    return;
  auto shouldIncludeAttr = [](const NamedAttribute &attr) {
    return attr.getName().getValue() != "iree.abi.name";
  };
  if (!llvm::any_of(attrs, shouldIncludeAttr))
    return;
  os << " {";
  llvm::interleaveComma(llvm::make_filter_range(attrs, shouldIncludeAttr), os,
                        [&](auto argAttr) {
                          os << argAttr.getName().getValue();
                          os << " = ";
                          os << argAttr.getValue();
                        });
  os << "}";
}

// Returns a string representing the |exportOp| in a human-friendly way.
// This doesn't have to match the exact MLIR (though could) as the intent is for
// it to be helpful instead of something a user could compile. This means we
// want to bake away argument/result attributes if we can do something
// meaningful with them (like names).
static StringAttr
formatSourceDeclaration(IREE::ABI::InvocationModel invocationModel,
                        func::FuncOp exportOp, StringRef publicName,
                        ArrayAttr allArgAttrs, ArrayAttr allResultAttrs) {
  std::string decl;
  llvm::raw_string_ostream os(decl);
  switch (invocationModel) {
  default:
    assert(false && "unhandled invocation model");
    break;
  case IREE::ABI::InvocationModel::Sync:
    os << "sync ";
    break;
  case IREE::ABI::InvocationModel::CoarseFences:
    os << "async ";
    break;
  }
  os << "func @" << publicName;
  os << "(";
  for (auto arg : exportOp.getArguments()) {
    if (arg.getArgNumber() > 0)
      os << ", ";
    os << "%";
    os << inferArgumentName(exportOp.getContext(), arg.getArgNumber(),
                            getIOAttr(allArgAttrs, arg.getArgNumber()))
              .getValue();
    os << ": " << arg.getType();
    if (auto argAttrs = getIOAttr(allArgAttrs, arg.getArgNumber())) {
      formatIOAttr(argAttrs, os);
    }
  }
  os << ") -> (";
  for (auto [resultNumber, resultType] :
       llvm::enumerate(exportOp.getResultTypes())) {
    if (resultNumber > 0)
      os << ", ";
    os << "%";
    os << inferResultName(exportOp.getContext(), resultNumber,
                          getIOAttr(allResultAttrs, resultNumber))
              .getValue();
    os << ": " << resultType;
    if (auto resultAttrs = getIOAttr(allResultAttrs, resultNumber)) {
      formatIOAttr(resultAttrs, os);
    }
  }
  os << ")";
  return StringAttr::get(exportOp.getContext(), decl);
}

// Populates attributes on |wrapperOp| to support runtime reflection.
// These are attached to the exported function and can be queried at runtime
// with iree_vm_function_lookup_attr_by_name.
static void populateReflectionAttrs(IREE::ABI::InvocationModel invocationModel,
                                    func::FuncOp exportOp,
                                    func::FuncOp wrapperOp) {
  auto *context = exportOp.getContext();
  SmallVector<NamedAttribute> attrs;

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

  // If not provided by the user add the source declaration as the MLIR type.
  // Users in source frontends can override this with something more natural
  // (python/whatever).
  if (auto declAttr = exportOp->getAttr("iree.abi.declaration")) {
    attrs.emplace_back(StringAttr::get(context, "iree.abi.declaration"),
                       declAttr);
  } else {
    attrs.emplace_back(StringAttr::get(context, "iree.abi.declaration"),
                       formatSourceDeclaration(invocationModel, exportOp,
                                               wrapperOp.getName(),
                                               exportOp.getAllArgAttrs(),
                                               exportOp.getAllResultAttrs()));
  }

  if (!attrs.empty()) {
    auto reflectionAttr = DictionaryAttr::get(context, attrs);
    wrapperOp->setAttr("iree.reflection", reflectionAttr);
  }
}

// Creates the corresponding wrapper function for the given export function.
static func::FuncOp
createExportWrapperFunc(IREE::ABI::InvocationModel invocationModel,
                        func::FuncOp exportOp, StringRef publicName) {
  // Copy arg/result attrs from the export op to the wrapper function.
  // We may want to remove them from the export but would need to filter.
  SmallVector<DictionaryAttr> argAttrDict;
  exportOp.getAllArgAttrs(argAttrDict);
  stripABIAttrs(argAttrDict);
  SmallVector<DictionaryAttr> resultAttrDict;
  exportOp.getAllResultAttrs(resultAttrDict);
  stripABIAttrs(resultAttrDict);

  // Convert argument types to those required by the binding ABI.
  //
  // NOTE: this is where we could change our signature to provide additional
  // values from the runtime bindings as may be required - like semaphores for
  // async behavior or cancellation.
  auto oldExportType = exportOp.getFunctionType();
  SmallVector<Type> inputTypes;
  for (auto oldType : oldExportType.getInputs()) {
    inputTypes.push_back(mapToABIType(oldType));
  }
  auto fenceType = IREE::HAL::FenceType::get(exportOp.getContext());
  switch (invocationModel) {
  default:
  case IREE::ABI::InvocationModel::Sync:
    break;
  case IREE::ABI::InvocationModel::CoarseFences:
    inputTypes.push_back(fenceType); // wait
    inputTypes.push_back(fenceType); // signal
    argAttrDict.push_back(nullptr);  // wait
    argAttrDict.push_back(nullptr);  // signal
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
    if (!outputAttr)
      continue;
    // Today all outputs need to be a !hal.buffer - we could change this
    // in the future to be something more generalized.
    auto storageArg = entryBlock->getArgument(i);
    if (!llvm::isa<IREE::HAL::BufferType>(storageArg.getType()) &&
        !llvm::isa<IREE::HAL::BufferViewType>(storageArg.getType())) {
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
  for (auto [argIndex, arg] : llvm::enumerate(
           entryBlock->getArguments().slice(0, oldExportType.getNumInputs()))) {
    auto oldType = oldExportType.getInput(argIndex);
    if (llvm::isa<TensorType>(oldType)) {
      auto encoding =
          exportOp.getArgAttrOfType<TypeAttr>(argIndex, "iree.abi.encoding");
      auto importOp = entryBuilder.create<IREE::HAL::TensorImportOp>(
          arg.getLoc(), oldType, arg,
          encoding ? encoding : TypeAttr::get(oldType), waitFence,
          inferArgumentName(entryBuilder.getContext(), argIndex,
                            exportOp.getArgAttrDict(argIndex)));
      arguments.push_back(importOp.getTarget());
    } else {
      arguments.push_back(arg);
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
      if (llvm::isa<TensorType>(result.getType()))
        asyncTensors.push_back(result);
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
    if (llvm::isa<TensorType>(oldType)) {
      auto encoding = exportOp.getResultAttrOfType<TypeAttr>(
          resultIndex, "iree.abi.encoding");
      auto dynamicDims = IREE::Util::buildDynamicDimsForValue(
          result.getLoc(), result, entryBuilder);
      auto resultStorage = resultStorages[resultIndex];
      results.push_back(entryBuilder.create<IREE::HAL::TensorExportOp>(
          result.getLoc(), newType, result,
          encoding ? encoding : TypeAttr::get(result.getType()), dynamicDims,
          resultStorage,
          inferResultName(entryBuilder.getContext(), resultIndex,
                          exportOp.getResultAttrDict(resultIndex))));
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
  if (!wrapperOp)
    return failure();
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
      if (funcOp->hasAttr("iree.abi.stub"))
        continue;
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
      if (failed(wrapImportFunc(getInvocationModel(importOp, invocationModel),
                                moduleOp, importOp, symbolTable))) {
        return signalPassFailure();
      }
    }

    // Create a wrapper function for each exported function.
    // This will change the export to taking the ABI types and preserve the
    // internal types.
    for (auto exportOp : exportOps) {
      if (failed(wrapExportFunc(getInvocationModel(exportOp, invocationModel),
                                moduleOp, exportOp, symbolTable))) {
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

std::unique_ptr<OperationPass<ModuleOp>>
createWrapEntryPointsPass(IREE::ABI::InvocationModel invocationModel) {
  return std::make_unique<WrapEntryPointsPass>(invocationModel);
}

static PassRegistration<WrapEntryPointsPass> pass;

} // namespace ABI
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
