// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TensorFlow exported functions support a structured calling convention
// consisting of fixed-arity lists and dicts flattened onto linear arguments
// and results. Metadata attributes are attached per argument and result
// indicating the "index path" into this nested structure (i.e. mixture of
// integral and string indices to descend into the hierachy).
//
// This pass unflattens the metadata, recreating the actual hierarchy and then
// creates a wrapper function conformant with the IREE ABI that is responsible
// which presents a nested view of the arguments and results. It then emits
// reflection metadata with full type mapping describing this situation and
// makes the original TF exported functions private.

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree_tf_compiler/TF/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace json = llvm::json;
namespace IREE = mlir::iree_compiler::IREE;

namespace mlir {
namespace iree_integrations {
namespace TF {

namespace {

enum class LevelType {
  None,

  // Leaf value.
  Value,

  // Structured level.
  List,
  Dict,
  Tuple,
};

json::Value mapTypeToJsonTypeRecord(Type type) {
  // All ShapedTypes are treated as buffer_views by the ABI.
  if (auto shapedType = type.dyn_cast<ShapedType>()) {
    json::Array record({
        json::Value("ndarray"),
        mapTypeToJsonTypeRecord(shapedType.getElementType()),
        shapedType.hasRank() ? json::Value(shapedType.getRank())
                             : json::Value(nullptr),
    });
    if (shapedType.hasRank()) {
      for (auto dim : shapedType.getShape()) {
        record.push_back(dim == ShapedType::kDynamicSize ? json::Value(nullptr)
                                                         : json::Value(dim));
      }
    }
    return record;
  }

  // Primitives.
  if (auto integerType = type.dyn_cast<IntegerType>()) {
    std::string name = (Twine("i") + Twine(integerType.getWidth())).str();
    return json::Value(std::move(name));
  }
  if (auto floatType = type.dyn_cast<FloatType>()) {
    if (floatType == FloatType::getBF16(floatType.getContext())) {
      // Why Google?
      return json::Value("bf16");
    }
    std::string name = (Twine("f") + Twine(floatType.getWidth())).str();
    return json::Value(std::move(name));
  }

  return json::Value("unknown");
}

struct StructureLevel {
  LevelType type = LevelType::None;

  // For Value level types, this is the index of the value (func argument
  // index for arguments, return index for returns).
  int valueIndex = 0;
  Type valueType;

  // For child levels, the key in the parent container, either a string or int
  // value.
  std::string skey;
  int ikey = 0;

  // Children (must be heap allocated due to recursion).
  std::vector<StructureLevel> children;

  // The root argument level is still just a list but has a special dict
  // child in which to shove kwargs. Once everything is constructed, if the
  // kwargs dict is not empty, it is added as the last child.
  bool isRootArgs = false;
  std::unique_ptr<StructureLevel> kwargs;

  // If this is the kwargs dict level, it will have this bit set.
  bool isKwargs = false;

  static StructureLevel leafValue(int valueIndex) {
    return StructureLevel{LevelType::Value, valueIndex};
  }

  static StructureLevel createRootArgsList() {
    StructureLevel ret = StructureLevel{LevelType::List};
    ret.isRootArgs = true;
    ret.kwargs =
        std::unique_ptr<StructureLevel>(new StructureLevel{LevelType::Dict});
    ret.kwargs->isKwargs = true;
    return ret;
  }

  Type getIrType(Builder builder) {
    auto variantType = IREE::VariantType::get(builder.getContext());
    if (type == LevelType::Value) {
      if (valueType.isa<TensorType>()) {
        return IREE::HAL::BufferViewType::get(builder.getContext());
      }
      return valueType;
    } else if (type == LevelType::List || type == LevelType::Tuple) {
      return IREE::ListType::get(variantType);
    } else if (type == LevelType::Dict) {
      return IREE::ListType::get(variantType);
    }

    llvm_unreachable("Unknown LevelType");
  }

  // For List/Dict/Tuple levels, returns the size of the list that is needed
  // to store all entries.
  int getNeededListSize() {
    if (type == LevelType::List || type == LevelType::Tuple) {
      int maxIkey = 0;
      for (auto &child : children) {
        maxIkey = std::max(maxIkey, child.ikey);
      }
      return maxIkey + 1;
    } else if (type == LevelType::Dict) {
      return children.size();
    }

    llvm_unreachable("Unsupported LevelType for getNeededListSize");
  }

  // Creates a JSON reflection type record describing this entity.
  json::Value createReflectionType() {
    switch (type) {
      case LevelType::Value:
        return mapTypeToJsonTypeRecord(valueType);
      case LevelType::List:
      case LevelType::Tuple: {
        json::Array typeRecord;
        typeRecord.push_back(
            json::Value(type == LevelType::List ? "slist" : "stuple"));
        int lastIndex = 0;
        for (auto &child : children) {
          for (int j = children.size(); j < child.ikey; ++j) {
            typeRecord.push_back(json::Value(nullptr));
          }
          lastIndex = child.valueIndex;
          typeRecord.push_back(child.createReflectionType());
        }
        return json::Value(std::move(typeRecord));
      }
      case LevelType::Dict: {
        json::Array typeRecord;
        typeRecord.push_back(isKwargs ? json::Value("sdict_kwargs")
                                      : json::Value("sdict"));
        for (auto &child : children) {
          json::Array nvRecord;
          nvRecord.push_back(child.skey);
          nvRecord.push_back(child.createReflectionType());
          typeRecord.push_back(json::Value(std::move(nvRecord)));
        }
        return json::Value(std::move(typeRecord));
      }
      default:
        llvm_unreachable("Unsupported LevelType");
    }

    return json::Value(nullptr);
  }

  // Recursively emits argument loads by processing all children and
  // populating callArgs with the Values of leaves.
  void emitDereferenceArgs(Location loc, OpBuilder &builder, Value thisValue,
                           SmallVector<Value> &callArgs) {
    // Terminal.
    if (type == LevelType::Value) {
      assert(valueIndex < callArgs.size() && "mismatched number of call args");
      assert(!callArgs[valueIndex] && "duplicate argument bindings");
      auto value = thisValue;
      if (value.getType().isa<IREE::HAL::BufferViewType>()) {
        value = builder.createOrFold<IREE::HAL::TensorCastOp>(loc, valueType,
                                                              thisValue);
      }
      callArgs[valueIndex] = value;
      return;
    }

    // Recurse into sequence (index can be sparse on child ikey).
    if (type == LevelType::List || type == LevelType::Tuple) {
      for (StructureLevel &child : children) {
        Value childValue =
            child.emitGetFromList(loc, builder, thisValue, child.ikey);
        child.emitDereferenceArgs(loc, builder, childValue, callArgs);
      }
      return;
    }

    // Recurse into dict (modeled as a dense tuple of children).
    if (type == LevelType::Dict) {
      for (auto it : llvm::enumerate(children)) {
        StructureLevel &child = it.value();
        Value childValue =
            child.emitGetFromList(loc, builder, thisValue, it.index());
        child.emitDereferenceArgs(loc, builder, childValue, callArgs);
      }
      return;
    }
    llvm_unreachable("unhandled StructureLevel type");
  }

  // Emits operations to recursively create this structure from the given
  // ValueRange of flattened values.
  Value emitCreateReturns(Location loc, OpBuilder &builder,
                          ResultRange &callReturns) {
    // Terminal.
    if (type == LevelType::Value) {
      assert(valueIndex < callReturns.size() &&
             "mismatched number of call returns");
      Value value = callReturns[valueIndex];
      if (valueType.isa<TensorType>()) {
        value = builder.createOrFold<IREE::HAL::TensorCastOp>(
            loc, getIrType(builder), value);
      }
      return value;
    }
    // Recurse into sequence (index can be sparse on child ikey).
    if (type == LevelType::List || type == LevelType::Tuple) {
      Value listSizeValue =
          builder.create<ConstantOp>(loc, builder.getIndexType(),
                                     builder.getIndexAttr(getNeededListSize()));
      Value listValue = builder.create<IREE::ListCreateOp>(
          loc, getIrType(builder), listSizeValue);
      builder.create<IREE::ListResizeOp>(loc, listValue, listSizeValue);
      for (StructureLevel &child : children) {
        Value childValue = child.emitCreateReturns(loc, builder, callReturns);
        Value indexValue = builder.create<ConstantOp>(
            loc, builder.getIndexType(), builder.getIndexAttr(child.ikey));
        builder.create<IREE::ListSetOp>(loc, listValue, indexValue, childValue);
      }
      return listValue;
    }

    // Recurse into dict (modeled as a dense tuple of children).
    if (type == LevelType::Dict) {
      Value listSizeValue =
          builder.create<ConstantOp>(loc, builder.getIndexType(),
                                     builder.getIndexAttr(getNeededListSize()));
      Value listValue = builder.create<IREE::ListCreateOp>(
          loc, getIrType(builder), listSizeValue);
      builder.create<IREE::ListResizeOp>(loc, listValue, listSizeValue);
      for (auto it : llvm::enumerate(children)) {
        StructureLevel &child = it.value();
        Value childValue = child.emitCreateReturns(loc, builder, callReturns);
        Value indexValue = builder.create<ConstantOp>(
            loc, builder.getIndexType(), builder.getIndexAttr(it.index()));
        builder.create<IREE::ListSetOp>(loc, listValue, indexValue, childValue);
      }
      return listValue;
    }
    llvm_unreachable("unhandled StructureLevel type");
  }

  // Emits operations to load this instance from a parent list value at the
  // given index.
  Value emitGetFromList(Location loc, OpBuilder &builder, Value parentList,
                        int index) {
    Value indexValue = builder.create<ConstantOp>(loc, builder.getIndexType(),
                                                  builder.getIndexAttr(index));
    Value itemValue = builder.create<IREE::ListGetOp>(loc, getIrType(builder),
                                                      parentList, indexValue);
    // TODO: Null check, etc. How does that work if returning a tensor? Need
    // to box somehow?
    if (itemValue.getType().isa<IREE::HAL::BufferViewType>()) {
      itemValue = builder.createOrFold<IREE::HAL::TensorCastOp>(loc, valueType,
                                                                itemValue);
    }
    return itemValue;
  }

  void normalize() {
    // Handle root kwargs.
    if (isRootArgs && !kwargs->children.empty()) {
      kwargs->ikey = children.size();
      children.push_back(std::move(*kwargs));
    }

    // Sort by key.
    if (type == LevelType::List || type == LevelType::Tuple) {
      std::sort(
          children.begin(), children.end(),
          [](StructureLevel &a, StructureLevel &b) { return a.ikey < b.ikey; });
    } else if (type == LevelType::Dict) {
      std::sort(
          children.begin(), children.end(),
          [](StructureLevel &a, StructureLevel &b) { return a.skey < b.skey; });
    }

    for (auto &child : children) child.normalize();
  }

  StructureLevel *bindValue(Location loc, int newValueIndex, Type valueType,
                            ArrayAttr indexPathAttr, bool bindTuple = false) {
    StructureLevel *current = this;
    // Move forward through non terminal path segments.
    for (Attribute indexAttr : indexPathAttr) {
      if (auto stringAttr = indexAttr.dyn_cast<StringAttr>()) {
        auto childKey = stringAttr.getValue();
        current = current->allocateChild(loc, childKey);
        if (!current) return nullptr;
      } else if (auto intAttr = indexAttr.dyn_cast<IntegerAttr>()) {
        int childIndex = intAttr.getInt();
        current =
            current->allocateChild(loc, childIndex, /*asTuple=*/bindTuple);
        if (!current) return nullptr;
      } else {
        emitError(loc)
            << "each index path component must be a string or integer";
        return nullptr;
      }
    }

    // If the root is not yet assigned, then it must be None.
    if (current->type != LevelType::None) {
      emitError(loc) << "duplicate assignment to structure path "
                     << indexPathAttr;
      return nullptr;
    }
    current->type = LevelType::Value;
    current->valueIndex = newValueIndex;
    current->valueType = valueType;
    return current;
  }

  StructureLevel *allocateChild(Location loc, StringRef childKey) {
    if (type == LevelType::None) type = LevelType::Dict;
    if (type != LevelType::Dict) {
      // Special case for root-args: shove it into the kwargs.
      if (isRootArgs) {
        return kwargs->allocateChild(loc, childKey);
      } else {
        emitError(loc) << "structure path mismatch: dereference a non-dict "
                       << "with a dict key '" << childKey << "'";
        return nullptr;
      }
    }
    for (auto &child : children) {
      if (child.skey == childKey) return &child;
    }

    // Not found: Create.
    children.push_back({});
    children.back().skey = childKey.str();
    return &children.back();
  }

  StructureLevel *allocateChild(Location loc, int childIndex,
                                bool asTuple = false) {
    if (type == LevelType::None)
      type = asTuple ? LevelType::Tuple : LevelType::List;
    if (type != LevelType::List && type != LevelType::Tuple) {
      emitError(loc) << "structure path mismatch: dereference a non-sequence "
                     << "with a sequence key " << childIndex;
      return nullptr;
    }
    for (auto &child : children) {
      if (child.ikey == childIndex) return &child;
    }

    // Not found: Create.
    children.push_back({});
    children.back().ikey = childIndex;
    return &children.back();
  }
};

LogicalResult materializeABIWrapper(ModuleOp module, FuncOp internalFunc,
                                    StringRef exportedName) {
  Location loc = internalFunc.getLoc();
  OpBuilder builder(internalFunc);
  const Identifier savedModelIndexPathIdent =
      builder.getIdentifier("tf_saved_model.index_path");
  FunctionType internalFuncType = internalFunc.getType();
  json::Array refArgs;
  json::Array refReturns;

  // Process each flattened argument into the argsRoot.
  StructureLevel argsRoot = StructureLevel::createRootArgsList();
  SmallVector<StructureLevel *> flattenedArgLevels;
  for (int i = 0, e = internalFunc.getNumArguments(); i < e; i++) {
    auto indexPathAttr = internalFunc.getArgAttrOfType<mlir::ArrayAttr>(
        i, savedModelIndexPathIdent);
    if (!indexPathAttr) {
      return internalFunc.emitError()
             << "Missing argument attribute " << savedModelIndexPathIdent
             << " on argument " << i;
    }
    internalFunc.removeArgAttr(i, savedModelIndexPathIdent);
    flattenedArgLevels.push_back(argsRoot.bindValue(
        loc, i, internalFuncType.getInput(i), indexPathAttr));
    if (!flattenedArgLevels.back()) {
      return failure();
    }
  }
  argsRoot.normalize();

  // Process each flattened result into the resultsRoot.
  StructureLevel resultsRoot = StructureLevel{};
  for (int i = 0, e = internalFunc.getNumResults(); i < e; i++) {
    auto indexPathAttr = internalFunc.getResultAttrOfType<mlir::ArrayAttr>(
        i, savedModelIndexPathIdent);
    if (!indexPathAttr) {
      return internalFunc.emitError()
             << "Missing result attribute " << savedModelIndexPathIdent
             << " on result " << i;
    }
    internalFunc.removeResultAttr(i, savedModelIndexPathIdent);
    // TODO: The TensorFlow SavedModel attribute system does not distinguish
    // lists from tuples, but TensorFlow internally does. Until this is
    // plumbed through somehow, arbitrarily emit results as tuples as that
    // was determined by someone at some point to be more canonical.
    if (!resultsRoot.bindValue(loc, i, internalFuncType.getResult(i),
                               indexPathAttr, /*bindTuple=*/true)) {
      return failure();
    }
  }
  resultsRoot.normalize();
  // Special case: root return is ambiguous between tuple and list. Bias
  // towards multi-return safe by converting to tuple.
  // TODO: Investigate upstream whether there are additional signals to be
  // plumbed.
  bool isMultiResult = resultsRoot.type == LevelType::Tuple;

  // Build the wrapper function type.
  SmallVector<Type> wrapperArgTypes;
  SmallVector<Type> wrapperResultTypes;
  for (StructureLevel &topLevelArg : argsRoot.children) {
    wrapperArgTypes.push_back(topLevelArg.getIrType(builder));
  }
  if (resultsRoot.type == LevelType::None) {
    // No returns.
  } else if (isMultiResult) {
    // Multi result for each child of the root.
    for (auto &child : resultsRoot.children) {
      wrapperResultTypes.push_back(child.getIrType(builder));
    }
  } else {
    // Single result (just the root).
    wrapperResultTypes.push_back(resultsRoot.getIrType(builder));
  }

  // Create the wrapper function.
  FunctionType wrapperFuncType =
      builder.getFunctionType(wrapperArgTypes, wrapperResultTypes);
  auto wrapperFunc = builder.create<FuncOp>(loc, exportedName, wrapperFuncType);
  SymbolTable::setSymbolVisibility(wrapperFunc,
                                   SymbolTable::Visibility::Public);
  Block *entryBlock = wrapperFunc.addEntryBlock();
  builder.setInsertionPointToEnd(entryBlock);

  // Flatten the arguments.
  // For each argument of the wrapper function, associate with a
  // StructureLevel and recursively emit dereferencing ops until reaching a
  // leaf.
  SmallVector<Value> callArgs;
  callArgs.resize(internalFunc.getNumArguments());
  for (auto it : llvm::enumerate(argsRoot.children)) {
    BlockArgument wrapperArgValue = entryBlock->getArgument(it.index());
    it.value().emitDereferenceArgs(loc, builder, wrapperArgValue, callArgs);
    refArgs.push_back(it.value().createReflectionType());
  }
  assert(llvm::all_of(callArgs, [](Value v) { return v != nullptr; }) &&
         "not all call arguments mapped");

  // Emit the call to the internal func.
  ResultRange internalResults =
      builder
          .create<CallOp>(loc, internalFunc.getType().getResults(),
                          internalFunc.getName(), callArgs)
          .getResults();

  // And then unflatten the results for return from the wrapper.
  SmallVector<Value> wrapperReturns;
  wrapperReturns.resize(wrapperResultTypes.size());
  if (resultsRoot.type == LevelType::None) {
    assert(wrapperReturns.empty() && "mismatched none return");
  } else if (isMultiResult) {
    // Multiple return.
    assert(resultsRoot.children.size() == wrapperReturns.size() &&
           "mismatched multiple result arity");
    for (int i = 0, e = resultsRoot.children.size(); i < e; ++i) {
      wrapperReturns[i] = resultsRoot.children[i].emitCreateReturns(
          loc, builder, internalResults);
      refReturns.push_back(resultsRoot.children[i].createReflectionType());
    }
  } else {
    // Single return.
    assert(wrapperReturns.size() == 1 &&
           "mismatched return arity for unary func");
    wrapperReturns[0] =
        resultsRoot.emitCreateReturns(loc, builder, internalResults);
    refReturns.push_back(resultsRoot.createReflectionType());
  }

  assert(llvm::all_of(wrapperReturns, [](Value v) { return v != nullptr; }) &&
         "not all call returns mapped");
  builder.create<ReturnOp>(loc, wrapperReturns);

  // Add ABI attribute.
  {
    std::string refStr;
    json::Object refDict;
    refDict["v"] = json::Value(1);
    refDict["a"] = json::Value(std::move(refArgs));
    refDict["r"] = json::Value(std::move(refReturns));
    json::Value refDictValue(std::move(refDict));
    llvm::raw_string_ostream refOut(refStr);
    refOut << refDictValue;
    refOut.flush();
    wrapperFunc->setAttr("iree.abi", builder.getStringAttr(refStr));
  }

  return success();
}

}  // namespace

class SavedModelToIREEABIPass
    : public PassWrapper<SavedModelToIREEABIPass, OperationPass<ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, iree_compiler::IREEDialect,
                    IREE::HAL::HALDialect,
                    mlir::tf_saved_model::TensorFlowSavedModelDialect>();
  }

  void runOnOperation() override {
    if (failed(run())) {
      signalPassFailure();
    }
  }

  LogicalResult run() {
    mlir::Builder builder(getOperation());
    const Identifier savedModelIndexPathIdent =
        builder.getIdentifier("tf_saved_model.index_path");
    (void)savedModelIndexPathIdent;

    // Handle saved model exported functions.
    for (auto func : getOperation().getOps<FuncOp>()) {
      // Transfer exported names to IREE.
      auto exportedNames = mlir::tf_saved_model::GetExportedNames(func);
      if (exportedNames.empty()) continue;
      if (exportedNames.size() > 1) {
        return func.emitError() << "Multiple exported names not supported yet";
      }

      StringRef exportedName = exportedNames.front();
      StringRef internalName = func.getName();
      if (internalName == exportedName) {
        // Normally, the actual IR function name is some mangled form only
        // relevant to some long departed TensorFlow devs. But there is nothing
        // saying it has to be, so if there is a collision, be nice and move
        // it out of the way.
        std::string rename = internalName.str();
        rename.append("__ireesm");
        SymbolTable::setSymbolName(func, rename);
      }
      SymbolTable::setSymbolVisibility(func, SymbolTable::Visibility::Private);
      if (failed(materializeABIWrapper(getOperation(), func, exportedName))) {
        return failure();
      }

      // Remove its designation as a saved model export.
      func->removeAttr("tf_saved_model.exported_names");
    }

    // We should have now removed anything requiring saved model semantics.
    getOperation()->removeAttr("tf_saved_model.semantics");
    return success();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createSavedModelToIREEABIPass() {
  return std::make_unique<SavedModelToIREEABIPass>();
}

static PassRegistration<SavedModelToIREEABIPass> pass(
    "tf-saved-model-to-iree-abi",
    "Creates IREE ABI entrypoints for saved model exports");

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir
