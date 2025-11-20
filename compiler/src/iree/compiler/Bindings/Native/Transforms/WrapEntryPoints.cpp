// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Bindings/Native/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::ABI {

#define GEN_PASS_DEF_WRAPENTRYPOINTSPASS
#include "iree/compiler/Bindings/Native/Transforms/Passes.h.inc"

// Returns the invocation model specified on |op| or the |defaultModel|.
static IREE::ABI::InvocationModel
getInvocationModel(Operation *op, IREE::ABI::InvocationModel defaultModel) {
  auto modelAttr = op->getAttrOfType<StringAttr>("iree.abi.model");
  if (!modelAttr) {
    return defaultModel;
  } else if (modelAttr == "coarse-fences") {
    return IREE::ABI::InvocationModel::CoarseFences;
  } else {
    return IREE::ABI::InvocationModel::Sync;
  }
}

// Maps a source type to the native ABI type.
static Type mapToABIType(Type type) {
  if (isa<TensorType>(type)) {
    return IREE::HAL::BufferViewType::get(type.getContext());
  }
  return type;
}

// Returns true if the given |attr| is a known ABI attribute that is only used
// by this pass.
static bool isABIAttr(NamedAttribute attr) {
  return attr.getName() == "iree.abi.affinity" ||
         attr.getName() == "iree.abi.encoding" ||
         attr.getName() == "iree.abi.model" ||
         attr.getName() == "iree.abi.output" ||
         attr.getName() == "iree.abi.transients";
}

// Removes all ABI attrs handled by this pass from all dictionaries.
static void stripABIAttrs(SmallVectorImpl<DictionaryAttr> &allAttrs) {
  for (auto &attrDict : allAttrs) {
    SmallVector<NamedAttribute> attrs;
    attrs.reserve(attrDict.size());
    for (auto attr : attrDict) {
      if (!isABIAttr(attr)) {
        attrs.push_back(attr);
      }
    }
    attrDict = DictionaryAttr::get(attrDict.getContext(), attrs);
  }
}

// Removes all ABI attrs from the |op| and its args/results.
static void stripABIAttrs(FunctionOpInterface op) {
  NamedAttrList attrs;
  for (auto attr : op->getAttrs()) {
    if (!isABIAttr(attr)) {
      attrs.push_back(attr);
    }
  }
  op->setAttrs(attrs);

  SmallVector<DictionaryAttr> argAttrs;
  op.getAllArgAttrs(argAttrs);
  stripABIAttrs(argAttrs);
  op.setAllArgAttrs(argAttrs);
  SmallVector<DictionaryAttr> resultAttrs;
  op.getAllResultAttrs(resultAttrs);
  stripABIAttrs(resultAttrs);
  op.setAllResultAttrs(resultAttrs);
}

template <typename T>
static T fallback(T optionalValue, T defaultValue) {
  return optionalValue ? optionalValue : defaultValue;
}

// Creates the corresponding wrapper function for the given import function.
static IREE::Util::FuncOp
createImportWrapperFunc(IREE::ABI::InvocationModel invocationModel,
                        FunctionOpInterface importOp,
                        FunctionType oldImportType, FunctionType newImportType,
                        StringRef privateName) {
  // Create the internal wrapper function with the original import signature.
  auto wrapperOp =
      IREE::Util::FuncOp::create(importOp.getLoc(), privateName, oldImportType);
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
  importOp.setType(newImportType);

  auto *entryBlock = wrapperOp.addEntryBlock();
  auto entryBuilder = OpBuilder::atBlockBegin(entryBlock);

  // Gather tensor arguments we may need to handle specially.
  SmallVector<Value> entryArgs = {entryBlock->getArguments().begin(),
                                  entryBlock->getArguments().end()};
  SmallVector<size_t> tensorArgIndices;
  SmallVector<Value> tensorArgs;
  for (auto [argIndex, arg] : llvm::enumerate(entryArgs)) {
    auto oldType = oldImportType.getInput(argIndex);
    if (isa<TensorType>(oldType)) {
      tensorArgIndices.push_back(argIndex);
      tensorArgs.push_back(arg);
    }
  }

  // Side-effecting imports need to have a host-wait inserted on them today.
  // We could add more configuration options here but for now we require that
  // users mark their functions 'nosideeffects' to avoid the host wait.
  const bool hasSideEffects = !importOp->hasAttr("nosideeffects");

  // Fetch and normalize any explicitly assigned affinity.
  auto defaultAffinityAttr = importOp->getAttr("iree.abi.affinity");
  if (defaultAffinityAttr) {
    importOp->setAttr("stream.affinity", defaultAffinityAttr);
  }

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
    Value device;
    // TODO(benvanik): support other affinity types.
    if (auto deviceAffinityAttr =
            dyn_cast_if_present<IREE::HAL::DeviceAffinityAttr>(
                defaultAffinityAttr)) {
      device =
          IREE::HAL::DeviceResolveOp::create(
              entryBuilder, importOp.getLoc(),
              entryBuilder.getType<IREE::HAL::DeviceType>(), deviceAffinityAttr)
              .getResult(0);
    } else {
      // HACK: if no devices are available we get the first one available at
      // runtime. This is suboptimal but we expect most usage to have affinities
      // assigned prior to ABI conversion.
      device =
          IREE::HAL::DeviceType::resolveAny(importOp.getLoc(), entryBuilder);
    }

    // When exporting a fence we need to put a barrier between the rest of the
    // program and the tensors consumed by the import.
    if (tensorArgs.empty()) {
      // No tensors passed to the import - pass in an immediate signal.
      waitFence = IREE::Util::NullOp::create(
          entryBuilder, importOp.getLoc(),
          entryBuilder.getType<IREE::HAL::FenceType>());
    } else {
      waitFence = IREE::HAL::FenceCreateOp::create(
          entryBuilder, importOp.getLoc(),
          entryBuilder.getType<IREE::HAL::FenceType>(), device,
          IREE::HAL::FenceFlagBitfield::None);
      auto barrierOp = IREE::HAL::TensorBarrierOp::create(
          entryBuilder, importOp.getLoc(), tensorArgs, waitFence);
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
        llvm::any_of(oldImportType.getResults(), llvm::IsaPred<TensorType>);
    if (!haveTensorResults && !hasSideEffects) {
      // No tensors returned from import - pass in an immediate signal.
      signalFence = IREE::Util::NullOp::create(
          entryBuilder, importOp.getLoc(),
          entryBuilder.getType<IREE::HAL::FenceType>());
    } else {
      signalFence = IREE::HAL::FenceCreateOp::create(
          entryBuilder, importOp.getLoc(),
          entryBuilder.getType<IREE::HAL::FenceType>(), device,
          IREE::HAL::FenceFlagBitfield::None);
    }
    break;
  }
  }

  // Marshal arguments.
  SmallVector<Value> arguments;
  for (auto [argIndex, arg] : llvm::enumerate(entryArgs)) {
    auto oldType = oldImportType.getInput(argIndex);
    auto newType = newImportType.getInput(argIndex);
    if (isa<TensorType>(oldType)) {
      // This is where we could perform type casting or in-place storage binding
      // if the user had any attrs specifying it.
      // NOTE: we insert a barrier on this above if needed so that the wait
      // fence will be signaled when the tensor is ready for consumption by the
      // import.
      auto encodingAttr =
          importOp.getArgAttrOfType<TypeAttr>(argIndex, "iree.abi.encoding");
      auto tensorExportOp = IREE::HAL::TensorExportOp::create(
          entryBuilder, arg.getLoc(), newType, arg,
          fallback(encodingAttr, TypeAttr::get(oldType)),
          /*name=*/nullptr,
          fallback(importOp.getArgAttr(argIndex, "iree.abi.affinity"),
                   defaultAffinityAttr));
      arguments.push_back(tensorExportOp.getTarget());
    } else {
      arguments.push_back(arg);
    }
  }
  if (waitFence) {
    arguments.push_back(waitFence);
  }
  if (signalFence) {
    arguments.push_back(signalFence);
  }

  // Make the call with the updated types.
  auto callOp = IREE::Util::CallOp::create(entryBuilder, importOp.getLoc(),
                                           importOp, arguments);

  // Determine if we need to await the signal fence on the host.
  // We only need to await if:
  // 1. The import has side-effects (may launch threads we can't track)
  // 2. AND there are no tensor results to propagate the fence through
  // If there are tensor results the fence propagates via hal.tensor.import
  // and the caller is responsible for synchronization.
  const bool haveTensorResults =
      llvm::any_of(oldImportType.getResults(), llvm::IsaPred<TensorType>);
  const bool mustAwait = hasSideEffects && signalFence && !haveTensorResults;
  if (mustAwait) {
    auto timeoutMillis =
        arith::ConstantIntOp::create(entryBuilder, importOp.getLoc(), -1, 32);
    IREE::HAL::FenceAwaitOp::create(
        entryBuilder, importOp.getLoc(), entryBuilder.getI32Type(),
        timeoutMillis, IREE::HAL::WaitFlagBitfield::None, signalFence);
  }

  // Marshal results.
  SmallVector<Value> results;
  for (auto [resultIndex, result] : llvm::enumerate(callOp.getResults())) {
    auto oldType = oldImportType.getResult(resultIndex);
    if (isa<TensorType>(oldType)) {
      // NOTE: we set the import pending on the signal fence from the import
      // indicating when the returned tensor is ready for consumption by the
      // program.
      auto encodingAttr = importOp.getResultAttrOfType<TypeAttr>(
          resultIndex, "iree.abi.encoding");
      auto consumeAttr = importOp.getResultAttrOfType<UnitAttr>(
          resultIndex, "iree.abi.consume");
      auto affinityAttr =
          importOp.getResultAttr(resultIndex, "iree.abi.affinity");
      auto tensorImportOp = IREE::HAL::TensorImportOp::create(
          entryBuilder, importOp.getLoc(), oldType, result,
          fallback(encodingAttr, TypeAttr::get(oldType)),
          consumeAttr ? true : false, signalFence,
          /*name=*/nullptr, fallback(affinityAttr, defaultAffinityAttr));
      results.push_back(tensorImportOp);
    } else {
      results.push_back(result);
    }
  }

  IREE::Util::ReturnOp::create(entryBuilder, importOp.getLoc(), results);

  stripABIAttrs(importOp);

  // Set the HAL ABI convention attribute on the import wrapper.
  // If we awaited then the wrapper is synchronous regardless of invocation
  // model. Otherwise, follow the module's invocation model.
  IREE::HAL::ExecutionModel executionModel;
  if (mustAwait) {
    executionModel = IREE::HAL::ExecutionModel::Synchronous;
  } else {
    executionModel = invocationModel == IREE::ABI::InvocationModel::CoarseFences
                         ? IREE::HAL::ExecutionModel::CoarseFences
                         : IREE::HAL::ExecutionModel::Synchronous;
  }
  IREE::HAL::ABIConventionAttr::setExecutionModel(wrapperOp, executionModel);

  return wrapperOp;
}

// Updates |importOp| to use the native ABI and creates a wrapper function that
// preserves the original behavior. All callers will be updated to point at the
// new wrapper function.
static LogicalResult wrapImportFunc(IREE::ABI::InvocationModel invocationModel,
                                    mlir::ModuleOp moduleOp,
                                    FunctionOpInterface importOp,
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
  SmallVector<Type> inputTypes;
  for (auto oldType : importOp.getArgumentTypes()) {
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
  for (auto oldType : importOp.getResultTypes()) {
    resultTypes.push_back(mapToABIType(oldType));
  }
  auto newImportType =
      FunctionType::get(importOp.getContext(), inputTypes, resultTypes);

  // Create the wrapper function that matches the original internal types but
  // calls out to the updated import using ABI types.
  auto wrapperOp = createImportWrapperFunc(
      invocationModel, importOp, cast<FunctionType>(importOp.getFunctionType()),
      newImportType, privateName);
  if (!wrapperOp) {
    return failure();
  }
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
  if (auto attrName = getNameFromDictAttr(attrs)) {
    return attrName;
  }
  return StringAttr::get(context, "input" + std::to_string(index));
}

static StringAttr inferResultName(MLIRContext *context, int index,
                                  DictionaryAttr attrs) {
  if (auto attrName = getNameFromDictAttr(attrs)) {
    return attrName;
  }
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
  if (!llvm::any_of(attrs, shouldIncludeAttr)) {
    return;
  }
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
                        FunctionOpInterface exportOp, StringRef publicName,
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
    if (arg.getArgNumber() > 0) {
      os << ", ";
    }
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
    if (resultNumber > 0) {
      os << ", ";
    }
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
                                    FunctionOpInterface exportOp,
                                    IREE::Util::FuncOp wrapperOp) {
  auto *context = exportOp.getContext();
  SmallVector<NamedAttribute> attrs;

  if (auto reflectionAttr =
          exportOp->getAttrOfType<DictionaryAttr>("iree.reflection")) {
    llvm::append_range(attrs, reflectionAttr.getValue());
  }

  if (auto abiAttr = exportOp->getAttr("iree.abi")) {
    attrs.emplace_back("iree.abi", abiAttr);
  }

  switch (invocationModel) {
  default:
  case IREE::ABI::InvocationModel::Sync:
    break;
  case IREE::ABI::InvocationModel::CoarseFences:
    attrs.emplace_back("iree.abi.model",
                       StringAttr::get(context, "coarse-fences"));
    break;
  }

  // If not provided by the user add the source declaration as the MLIR type.
  // Users in source frontends can override this with something more natural
  // (python/whatever).
  if (auto declAttr = exportOp->getAttr("iree.abi.declaration")) {
    attrs.emplace_back("iree.abi.declaration", declAttr);
  } else {
    attrs.emplace_back("iree.abi.declaration",
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
static IREE::Util::FuncOp
createExportWrapperFunc(IREE::ABI::InvocationModel invocationModel,
                        FunctionOpInterface exportOp, StringRef publicName) {
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
  SmallVector<Type> inputTypes;
  for (auto oldType : exportOp.getArgumentTypes()) {
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
  for (auto oldType : exportOp.getResultTypes()) {
    resultTypes.push_back(mapToABIType(oldType));
  }
  auto newExportType =
      FunctionType::get(exportOp.getContext(), inputTypes, resultTypes);

  // Update the import to the new type and mark it as being converted so we
  // don't try to convert it again.
  auto wrapperOp =
      IREE::Util::FuncOp::create(exportOp.getLoc(), publicName, newExportType);
  wrapperOp.setPublic();
  wrapperOp->setAttr("iree.abi.stub", UnitAttr::get(exportOp.getContext()));
  wrapperOp.setAllArgAttrs(argAttrDict);
  wrapperOp.setAllResultAttrs(resultAttrDict);

  // Populate the reflection attrs based on the original types.
  populateReflectionAttrs(invocationModel, exportOp, wrapperOp);
  exportOp->removeAttr("iree.reflection");

  // Fetch and normalize any explicitly assigned affinity.
  auto defaultAffinityAttr = exportOp->getAttr("iree.abi.affinity");
  if (defaultAffinityAttr) {
    exportOp->setAttr("stream.affinity", defaultAffinityAttr);
  }

  auto *entryBlock = wrapperOp.addEntryBlock();
  auto entryBuilder = OpBuilder::atBlockBegin(entryBlock);

  // Build a map of result value to the argument that has its backing storage.
  SmallVector<Value> resultStorages;
  resultStorages.resize(resultTypes.size());
  for (unsigned i = 0; i < exportOp.getNumArguments(); ++i) {
    auto outputAttr =
        exportOp.getArgAttrOfType<IntegerAttr>(i, "iree.abi.output");
    if (!outputAttr) {
      continue;
    }
    // Today all outputs need to be a !hal.buffer - we could change this
    // in the future to be something more generalized.
    auto storageArg = entryBlock->getArgument(i);
    if (!isa<IREE::HAL::BufferType>(storageArg.getType()) &&
        !isa<IREE::HAL::BufferViewType>(storageArg.getType())) {
      exportOp.emitError() << "storage argument " << i
                           << " has an invalid type " << storageArg.getType()
                           << "; must be a !hal.buffer";
      return {};
    }
    resultStorages[outputAttr.getInt()] = storageArg;
  }

  // Find the transient storage buffer if provided.
  Value transientStorage;
  for (unsigned i = 0; i < exportOp.getNumArguments(); ++i) {
    if (exportOp.getArgAttrOfType<UnitAttr>(i, "iree.abi.transients")) {
      auto storageArg = entryBlock->getArgument(i);
      if (!llvm::isa<IREE::HAL::BufferType>(storageArg.getType()) &&
          !llvm::isa<IREE::HAL::BufferViewType>(storageArg.getType())) {
        exportOp.emitError() << "transient storage argument " << i
                             << " has an invalid type " << storageArg.getType()
                             << "; must be a !hal.buffer or !hal.buffer_view";
        return {};
      }
      if (transientStorage) {
        exportOp.emitError()
            << "only one argument may have the iree.abi.transients attribute";
        return {};
      }
      transientStorage = storageArg;
    }
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
  auto oldExportType = cast<FunctionType>(exportOp.getFunctionType());
  SmallVector<Value> arguments;
  for (auto [argIndex, arg] : llvm::enumerate(
           entryBlock->getArguments().slice(0, oldExportType.getNumInputs()))) {
    auto oldType = oldExportType.getInput(argIndex);
    if (isa<TensorType>(oldType)) {
      auto encodingAttr =
          exportOp.getArgAttrOfType<TypeAttr>(argIndex, "iree.abi.encoding");
      auto consumeAttr =
          exportOp.getArgAttrOfType<UnitAttr>(argIndex, "iree.abi.consume");
      auto affinityAttr = exportOp.getArgAttr(argIndex, "iree.abi.affinity");
      auto argName = inferArgumentName(entryBuilder.getContext(), argIndex,
                                       exportOp.getArgAttrDict(argIndex));
      auto tensorImportOp = IREE::HAL::TensorImportOp::create(
          entryBuilder, arg.getLoc(), oldType, arg,
          fallback(encodingAttr, TypeAttr::get(oldType)),
          /*consume=*/consumeAttr ? true : false, waitFence, argName,
          fallback(affinityAttr, defaultAffinityAttr));
      arguments.push_back(tensorImportOp.getTarget());
    } else {
      arguments.push_back(arg);
    }
  }

  // Make the call with the original types.
  auto callOp = IREE::Util::CallOp::create(entryBuilder, exportOp.getLoc(),
                                           exportOp, arguments);
  auto asyncResults = llvm::to_vector(callOp.getResults());

  // Alias results to storage buffers if provided.
  for (unsigned resultIndex = 0; resultIndex < asyncResults.size();
       ++resultIndex) {
    if (!resultStorages[resultIndex]) {
      continue;
    }
    auto source = asyncResults[resultIndex];
    auto sourceDims = IREE::Util::buildDynamicDimsForValue(
        exportOp.getLoc(), source, entryBuilder);
    auto aliasOp = IREE::HAL::TensorAliasOp::create(
        entryBuilder, exportOp.getLoc(), source.getType(), source, sourceDims,
        resultStorages[resultIndex], waitFence,
        fallback(exportOp.getResultAttr(resultIndex, "iree.abi.affinity"),
                 defaultAffinityAttr));
    asyncResults[resultIndex] = cast<OpResult>(aliasOp.getResult());
  }

  // Annotate transient storage on produced results, if provided.
  // The op indicates "any transient memory used for producing this value should
  // be allocated from this storage."
  if (transientStorage) {
    for (auto [resultIndex, result] : llvm::enumerate(asyncResults)) {
      if (llvm::isa<TensorType>(result.getType())) {
        auto sourceDims = IREE::Util::buildDynamicDimsForValue(
            exportOp.getLoc(), result, entryBuilder);
        auto transientsOp = IREE::HAL::TensorTransientsOp::create(
            entryBuilder, exportOp.getLoc(), result.getType(), result,
            sourceDims, transientStorage,
            fallback(exportOp.getResultAttr(resultIndex, "iree.abi.affinity"),
                     defaultAffinityAttr));
        asyncResults[resultIndex] = cast<OpResult>(transientsOp.getResult());
      }
    }
  }

  // Insert a barrier if requested - all tensors will be calculated and the
  // fence will be signaled. Note that even if there are no tensor results we
  // need to signal the fence.
  if (signalFence) {
    SmallVector<Value> asyncTensors;
    for (auto result : asyncResults) {
      if (isa<TensorType>(result.getType())) {
        asyncTensors.push_back(result);
      }
    }
    if (asyncTensors.empty()) {
      // TODO(benvanik): maybe use a global timeline? global stores may not
      // have completed by now in cases where the user wants to loop back.
      IREE::HAL::FenceSignalOp::create(entryBuilder, exportOp.getLoc(),
                                       signalFence);
    } else {
      auto barrierOp = IREE::HAL::TensorBarrierOp::create(
          entryBuilder, exportOp.getLoc(), asyncTensors, signalFence);
      asyncResults = llvm::to_vector(barrierOp.getResults());
    }
  }

  // Marshal results.
  SmallVector<Value> results;
  for (auto [resultIndex, result] : llvm::enumerate(asyncResults)) {
    auto oldType = oldExportType.getResult(resultIndex);
    auto newType = newExportType.getResult(resultIndex);
    if (isa<TensorType>(oldType)) {
      auto encodingAttr = exportOp.getResultAttrOfType<TypeAttr>(
          resultIndex, "iree.abi.encoding");
      auto affinityAttr =
          exportOp.getResultAttr(resultIndex, "iree.abi.affinity");
      auto resultName =
          inferResultName(entryBuilder.getContext(), resultIndex,
                          exportOp.getResultAttrDict(resultIndex));
      auto dynamicDims = IREE::Util::buildDynamicDimsForValue(
          result.getLoc(), result, entryBuilder);
      auto tensorExportOp = IREE::HAL::TensorExportOp::create(
          entryBuilder, result.getLoc(), newType, result,
          fallback(encodingAttr, TypeAttr::get(result.getType())), dynamicDims,
          resultName, fallback(affinityAttr, defaultAffinityAttr));
      results.push_back(tensorExportOp);
    } else {
      results.push_back(result);
    }
  }

  stripABIAttrs(exportOp);

  // Set the HAL ABI convention attribute on the export wrapper.
  IREE::HAL::ABIConventionAttr::setExecutionModel(
      exportOp, invocationModel == IREE::ABI::InvocationModel::CoarseFences
                    ? IREE::HAL::ExecutionModel::CoarseFences
                    : IREE::HAL::ExecutionModel::Synchronous);

  IREE::Util::ReturnOp::create(entryBuilder, exportOp.getLoc(), results);
  return wrapperOp;
}

// Replaces |exportOp| with a wrapper function that exports the native ABI.
// The original function will be made private and be renamed.
// This allows us to support multiple binding schemes as transforms from other
// bindings can also perform their own equivalent wrapping.
static LogicalResult wrapExportFunc(IREE::ABI::InvocationModel invocationModel,
                                    mlir::ModuleOp moduleOp,
                                    FunctionOpInterface exportOp,
                                    SymbolTable &symbolTable) {
  // Rename the original function so that our wrapper can use the original
  // name in its public definition.
  auto publicName = exportOp.getName().str();
  auto privateName = "_" + publicName;
  if (failed(symbolTable.rename(exportOp, privateName))) {
    return exportOp.emitError() << "unknown symbol table op encountered; "
                                   "cannot fix up symbol names";
  }
  exportOp.setPrivate();

  // Create the wrapper function that conforms to the IREE native ABI and
  // marshals arguments/results to the original function.
  auto wrapperOp =
      createExportWrapperFunc(invocationModel, exportOp, publicName);
  if (!wrapperOp) {
    return failure();
  }
  symbolTable.insert(wrapperOp, Block::iterator(exportOp));

  return success();
}

// Wraps all entry points in a function that is compatible with the
// expected invocation semantics of bindings following the native IREE ABI.
// Imports are also handled as they are entry points in another module.
class WrapEntryPointsPass
    : public impl::WrapEntryPointsPassBase<WrapEntryPointsPass> {
public:
  using Base::Base;

  WrapEntryPointsPass(IREE::ABI::InvocationModel invocationModel) {
    this->invocationModel = invocationModel;
  }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    // Gather functions that need wrapping.
    SmallVector<FunctionOpInterface> importOps;
    SmallVector<FunctionOpInterface> exportOps;
    for (auto funcOp : moduleOp.getOps<IREE::Util::FuncOp>()) {
      // Ignore functions already marked as having their ABI goo handled.
      if (funcOp->hasAttr("iree.abi.stub")) {
        continue;
      } else if (funcOp.isExternal()) {
        // Imported function.
        importOps.push_back(funcOp);
      } else if (funcOp.isPublic()) {
        // Exported function.
        exportOps.push_back(funcOp);
      }
    }
    if (importOps.empty() && exportOps.empty()) {
      return; // no-op
    }

    SymbolTable symbolTable(moduleOp);

    // Create a wrapper function for each imported function.
    // This will preserve the internal types (tensors/etc) but change the
    // import to taking the ABI types and rewrite calls.
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
};

} // namespace mlir::iree_compiler::IREE::ABI
