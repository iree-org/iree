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

Value castToImportType(Value value, Type targetType,
                       ConversionPatternRewriter &rewriter) {
  auto sourceType = value.getType();
  if (sourceType == targetType)
    return value;
  bool sourceIsInteger = llvm::isa<IntegerType>(sourceType);

  // Allow bitcast between same width float/int types. This is used for
  // marshalling to "untyped" VM interfaces, which will have an integer type.
  if (llvm::isa<FloatType>(sourceType) && llvm::isa<IntegerType>(targetType) &&
      sourceType.getIntOrFloatBitWidth() ==
          targetType.getIntOrFloatBitWidth()) {
    return rewriter.create<mlir::arith::BitcastOp>(value.getLoc(), targetType,
                                                   value);
  } else if (sourceIsInteger &&
             (targetType.isSignedInteger() || targetType.isSignlessInteger())) {
    if (targetType.getIntOrFloatBitWidth() >
        sourceType.getIntOrFloatBitWidth()) {
      return rewriter.create<mlir::arith::ExtSIOp>(value.getLoc(), targetType,
                                                   value);
    } else {
      return rewriter.create<mlir::arith::TruncIOp>(value.getLoc(), targetType,
                                                    value);
    }
  } else if (sourceIsInteger && targetType.isUnsignedInteger()) {
    if (targetType.getIntOrFloatBitWidth() >
        sourceType.getIntOrFloatBitWidth()) {
      return rewriter.create<mlir::arith::ExtUIOp>(value.getLoc(), targetType,
                                                   value);
    } else {
      return rewriter.create<mlir::arith::TruncIOp>(value.getLoc(), targetType,
                                                    value);
    }
  } else {
    return value;
  }
}

Value castFromImportType(Value value, Type targetType,
                         ConversionPatternRewriter &rewriter) {
  // Right now the to-import and from-import types are the same.
  return castToImportType(value, targetType, rewriter);
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

std::optional<SmallVector<Value>>
rewriteAttrToOperands(Location loc, Attribute attrValue, Type inputType,
                      ConversionPatternRewriter &rewriter) {
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attrValue)) {
    // NOTE: we intentionally go to std.constant ops so that the standard
    // conversions can do their job. If we want to remove the dependency
    // from standard ops in the future we could instead go directly to
    // one of the vm constant ops.
    auto constValue = rewriter.createOrFold<mlir::arith::ConstantOp>(
        loc, inputType,
        IntegerAttr::get(inputType,
                         APInt(32, static_cast<int32_t>(intAttr.getInt()))));
    return {{constValue}};
  }
  if (auto elementsAttr = llvm::dyn_cast<DenseIntElementsAttr>(attrValue)) {
    SmallVector<Value> elementValues;
    elementValues.reserve(elementsAttr.getNumElements());
    for (auto intAttr : elementsAttr.getValues<Attribute>()) {
      elementValues.push_back(rewriter.createOrFold<mlir::arith::ConstantOp>(
          loc, elementsAttr.getType().getElementType(),
          cast<TypedAttr>(intAttr)));
    }
    return elementValues;
  }
  if (auto arrayAttr = llvm::dyn_cast<ArrayAttr>(attrValue)) {
    SmallVector<Value> allValues;
    for (auto elementAttr : arrayAttr) {
      auto flattenedValues =
          rewriteAttrToOperands(loc, elementAttr, inputType, rewriter);
      if (!flattenedValues)
        return std::nullopt;
      allValues.append(flattenedValues->begin(), flattenedValues->end());
    }
    return allValues;
  }
  if (auto strAttr = llvm::dyn_cast<StringAttr>(attrValue)) {
    return {{rewriter.create<IREE::VM::RodataInlineOp>(loc, strAttr)}};
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
                rewriteAttrToOperands(loc, elementAttr, elementType, rewriter);
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
                rewriteAttrToOperands(loc, elementAttr, inputType, rewriter);
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
