// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"

#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Parser/Parser.h"

namespace mlir::iree_compiler {

LogicalResult ImportTable::build(Operation *rootOp,
                                 const TypeConverter &typeConverter) {
  for (auto funcOp : rootOp->getRegion(0).getOps<FunctionOpInterface>()) {
    if (!funcOp.isExternal()) {
      continue; // only external functions are imports
    }

    ImportTable::Import import;
    import.name = funcOp.getNameAttr();
    import.fallback = funcOp->getAttrOfType<SymbolRefAttr>("vm.fallback");

    // Try to use an assigned signature or fall back to converting the input.
    if (auto importOp = dyn_cast<IREE::VM::ImportOp>(funcOp.getOperation())) {
      // Import ops have their signature used directly.
      import.signature = importOp.getFunctionType();
    } else if (auto signatureAttr =
                   funcOp->getAttrOfType<TypeAttr>("vm.signature")) {
      // Directly use the specified signature.
      import.signature =
          dyn_cast_if_present<FunctionType>(signatureAttr.getValue());
    }
    if (!import.signature) {
      // Convert the signature using the type converter.
      SmallVector<Type> argumentTypes;
      if (failed(typeConverter.convertTypes(funcOp.getArgumentTypes(),
                                            argumentTypes))) {
        return funcOp.emitError() << "unable to convert import argument types";
      }
      SmallVector<Type> resultTypes;
      if (failed(typeConverter.convertTypes(funcOp.getResultTypes(),
                                            resultTypes))) {
        return funcOp.emitError() << "unable to convert import result types";
      }
      import.signature =
          FunctionType::get(rootOp->getContext(), argumentTypes, resultTypes);
    }

    symbols[import.name.getValue()] = std::move(import);
  }

  return success();
}

std::optional<ImportTable::Import> ImportTable::find(StringRef symbolName) {
  auto it = symbols.find(symbolName);
  if (it == symbols.end())
    return std::nullopt;
  return it->second;
}

// TODO(benvanik): replace with iree/compiler/Utils/ModuleUtils.h.
// There may be some special insertion order arrangement required based on the
// nested vm.module here.

LogicalResult appendImportModule(IREE::VM::ModuleOp importModuleOp,
                                 ModuleOp targetModuleOp) {
  SymbolTable symbolTable(targetModuleOp);
  OpBuilder targetBuilder(targetModuleOp);
  targetBuilder.setInsertionPoint(&targetModuleOp.getBody()->back());
  importModuleOp.walk([&](IREE::VM::ImportOp importOp) {
    std::string fullName =
        (importModuleOp.getName() + "." + importOp.getName()).str();
    if (auto *existingOp = symbolTable.lookup(fullName)) {
      existingOp->erase();
    }
    auto clonedOp = cast<IREE::VM::ImportOp>(targetBuilder.clone(*importOp));
    mlir::StringAttr fullNameAttr =
        mlir::StringAttr::get(clonedOp.getContext(), fullName);
    clonedOp.setName(fullNameAttr);
    clonedOp.setPrivate();
  });
  return success();
}

LogicalResult appendImportModule(StringRef importModuleSrc,
                                 ModuleOp targetModuleOp) {
  auto importModuleRef = mlir::parseSourceString<mlir::ModuleOp>(
      importModuleSrc, targetModuleOp.getContext());
  if (!importModuleRef) {
    return targetModuleOp.emitError()
           << "unable to append import module; import module failed to parse";
  }
  for (auto importModuleOp : importModuleRef->getOps<IREE::VM::ModuleOp>()) {
    if (failed(appendImportModule(importModuleOp, targetModuleOp))) {
      importModuleOp.emitError() << "failed to import module";
    }
  }
  return success();
}

Value castToImportType(Value value, Type targetType, OpBuilder &builder) {
  auto sourceType = value.getType();
  if (sourceType == targetType)
    return value;
  bool sourceIsInteger = llvm::isa<IntegerType>(sourceType);

  // Allow bitcast between same width float/int types. This is used for
  // marshalling to "untyped" VM interfaces, which will have an integer type.
  if (llvm::isa<FloatType>(sourceType) && llvm::isa<IntegerType>(targetType) &&
      sourceType.getIntOrFloatBitWidth() ==
          targetType.getIntOrFloatBitWidth()) {
    return builder.create<mlir::arith::BitcastOp>(value.getLoc(), targetType,
                                                  value);
  } else if (sourceIsInteger &&
             (targetType.isSignedInteger() || targetType.isSignlessInteger())) {
    if (targetType.getIntOrFloatBitWidth() >
        sourceType.getIntOrFloatBitWidth()) {
      return builder.create<mlir::arith::ExtSIOp>(value.getLoc(), targetType,
                                                  value);
    } else {
      return builder.create<mlir::arith::TruncIOp>(value.getLoc(), targetType,
                                                   value);
    }
  } else if (sourceIsInteger && targetType.isUnsignedInteger()) {
    if (targetType.getIntOrFloatBitWidth() >
        sourceType.getIntOrFloatBitWidth()) {
      return builder.create<mlir::arith::ExtUIOp>(value.getLoc(), targetType,
                                                  value);
    } else {
      return builder.create<mlir::arith::TruncIOp>(value.getLoc(), targetType,
                                                   value);
    }
  } else {
    return value;
  }
}

Value castFromImportType(Value value, Type targetType, OpBuilder &builder) {
  // Right now the to-import and from-import types are the same.
  return castToImportType(value, targetType, builder);
}

void copyImportAttrs(IREE::VM::ImportOp importOp, Operation *callOp) {
  if (importOp->hasAttr("nosideeffects")) {
    callOp->setAttr("nosideeffects", UnitAttr::get(importOp.getContext()));
  }
}

namespace detail {

size_t getSegmentSpanSize(Type spanType) {
  if (auto tupleType = llvm::dyn_cast<TupleType>(spanType)) {
    return tupleType.size();
  } else {
    return 1;
  }
}

std::optional<SmallVector<Value>> rewriteAttrToOperands(Location loc,
                                                        Attribute attrValue,
                                                        Type inputType,
                                                        OpBuilder &builder) {
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attrValue)) {
    // NOTE: we intentionally go to std.constant ops so that the standard
    // conversions can do their job. If we want to remove the dependency
    // from standard ops in the future we could instead go directly to
    // one of the vm constant ops.
    auto constValue = builder.create<mlir::arith::ConstantOp>(
        loc, inputType,
        IntegerAttr::get(inputType, APInt(inputType.getIntOrFloatBitWidth(),
                                          intAttr.getValue().getSExtValue())));
    return {{constValue}};
  } else if (auto floatAttr = llvm::dyn_cast<FloatAttr>(attrValue)) {
    bool lossy = false;
    APFloat value = floatAttr.getValue();
    value.convert(llvm::cast<FloatType>(inputType).getFloatSemantics(),
                  llvm::RoundingMode::NearestTiesToEven, &lossy);
    auto constValue = builder.create<mlir::arith::ConstantOp>(
        loc, inputType, FloatAttr::get(inputType, value));
    return {{constValue}};
  } else if (auto elementsAttr =
                 llvm::dyn_cast<DenseIntElementsAttr>(attrValue)) {
    SmallVector<Value> elementValues;
    elementValues.reserve(elementsAttr.getNumElements());
    for (auto intAttr : elementsAttr.getValues<Attribute>()) {
      elementValues.push_back(builder.create<mlir::arith::ConstantOp>(
          loc, elementsAttr.getType().getElementType(),
          cast<TypedAttr>(intAttr)));
    }
    return elementValues;
  } else if (auto arrayAttr = llvm::dyn_cast<ArrayAttr>(attrValue)) {
    SmallVector<Value> allValues;
    for (auto elementAttr : arrayAttr) {
      auto flattenedValues =
          rewriteAttrToOperands(loc, elementAttr, inputType, builder);
      if (!flattenedValues)
        return std::nullopt;
      allValues.append(flattenedValues->begin(), flattenedValues->end());
    }
    return allValues;
  } else if (auto strAttr = llvm::dyn_cast<StringAttr>(attrValue)) {
    return {{builder.create<IREE::VM::RodataInlineOp>(loc, strAttr)}};
  }

  // This may be a custom dialect type. As we can't trivially access the storage
  // of these we need to ask the dialect to do it for us.
  auto *conversionInterface =
      attrValue.getDialect()
          .getRegisteredInterface<VMConversionDialectInterface>();
  if (conversionInterface) {
    bool anyFailed = false;
    SmallVector<Value> allValues;
    if (auto tupleType = llvm::dyn_cast<TupleType>(inputType)) {
      // Custom dialect type maps into a tuple; we expect 1:1 tuple elements to
      // attribute storage elements.
      auto tupleTypes = llvm::to_vector(tupleType.getTypes());
      int ordinal = 0;
      LogicalResult walkStatus = conversionInterface->walkAttributeStorage(
          attrValue, [&](Attribute elementAttr) {
            if (anyFailed)
              return;
            auto elementType = tupleTypes[ordinal++];
            auto flattenedValues =
                rewriteAttrToOperands(loc, elementAttr, elementType, builder);
            if (!flattenedValues) {
              anyFailed = true;
              return;
            }
            allValues.append(flattenedValues->begin(), flattenedValues->end());
          });
      if (failed(walkStatus))
        return std::nullopt;
    } else {
      // Custom dialect type maps into zero or more input types (ala arrays).
      LogicalResult walkStatus = conversionInterface->walkAttributeStorage(
          attrValue, [&](Attribute elementAttr) {
            if (anyFailed)
              return;
            auto flattenedValues =
                rewriteAttrToOperands(loc, elementAttr, inputType, builder);
            if (!flattenedValues) {
              anyFailed = true;
              return;
            }
            allValues.append(flattenedValues->begin(), flattenedValues->end());
          });
      if (failed(walkStatus))
        return std::nullopt;
    }
    if (anyFailed)
      return std::nullopt;
    return allValues;
  }

  emitError(loc) << "unsupported attribute encoding: " << attrValue;
  return std::nullopt;
}

} // namespace detail

} // namespace mlir::iree_compiler
