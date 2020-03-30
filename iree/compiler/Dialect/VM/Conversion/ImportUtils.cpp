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

#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"

#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"

namespace mlir {
namespace iree_compiler {

LogicalResult appendImportModule(IREE::VM::ModuleOp importModuleOp,
                                 ModuleOp targetModuleOp) {
  SymbolTable symbolTable(targetModuleOp);
  OpBuilder targetBuilder(targetModuleOp);
  targetBuilder.setInsertionPoint(&targetModuleOp.getBody()->back());
  importModuleOp.walk([&](IREE::VM::ImportOp importOp) {
    std::string fullName =
        (importModuleOp.getName() + "." + importOp.getName()).str();
    auto *existingOp = symbolTable.lookup(fullName);
    // TODO(benvanik): verify that the imports match.
    if (!existingOp) {
      auto clonedOp = cast<IREE::VM::ImportOp>(targetBuilder.clone(*importOp));
      clonedOp.setName(fullName);
      SymbolTable::setSymbolVisibility(clonedOp,
                                       SymbolTable::Visibility::Private);
    }
  });
  return success();
}

LogicalResult appendImportModule(StringRef importModuleSrc,
                                 ModuleOp targetModuleOp) {
  auto importModuleRef =
      mlir::parseSourceString(importModuleSrc, targetModuleOp.getContext());
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

namespace detail {

// Makes a human-readable symbol name for the given string value.
// This is not uniqued and may need uniquing before being added to the symbol
// table.
//
// For example:
//   'Some string!' -> '_utf8_some_string'
//   'I'm a really long'... -> '_utf8_im_a_really_long'
static std::string makeSafeIdentifier(StringRef unsafeIdentifier) {
  std::string result = "_utf8_";
  llvm::raw_string_ostream os(result);
  bool lastUnderscore = true;
  for (char c : unsafeIdentifier) {
    if (!llvm::isPrint(c)) continue;
    if (llvm::isAlnum(c)) {
      os << llvm::toLower(c);
      lastUnderscore = false;
    } else if (!lastUnderscore) {
      os << "_";
      lastUnderscore = true;
    }
  }
  std::string prefix = os.str().substr(0, 32);
  if (!StringRef(prefix).endswith("_")) {
    prefix += "_";
  }
  return prefix + llvm::utohexstr(static_cast<uint64_t>(
                      llvm::hash_value(unsafeIdentifier)));
}

// Creates a module-level vm.rodata containing the string contents and returns
// the dereferenced byte buffer.
static Value createStringTableValue(Location loc, StringAttr attrValue,
                                    Type inputType,
                                    ConversionPatternRewriter &rewriter) {
  auto stringValue = attrValue.getValue();

  // Make an identifier-friendly version of the string so that the value is
  // more readable in IR (so "I'm some string" becomes "im_some_string", etc).
  auto safeIdentifier = makeSafeIdentifier(stringValue);

  // Encode the string value bytes into an elements attr as UTF-8 bytes.
  SmallVector<APInt, 16> stringBytes(stringValue.size());
  for (int i = 0; i < stringValue.size(); ++i) {
    stringBytes[i] = APInt(8, stringValue[i]);
  }
  auto utf8Bytes = DenseIntElementsAttr::get(
      VectorType::get({static_cast<int64_t>(stringBytes.size())},
                      rewriter.getIntegerType(8)),
      stringBytes);

  // Create vm.rodata holding the data.
  auto funcOp = dyn_cast_or_null<IREE::VM::FuncOp>(
      rewriter.getInsertionBlock()->getParentOp());
  assert(funcOp && "value access not in a function");
  auto insertPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(funcOp);
  auto rodataOp =
      rewriter.create<IREE::VM::RodataOp>(loc, safeIdentifier, utf8Bytes);
  SymbolTable::setSymbolVisibility(rodataOp, SymbolTable::Visibility::Private);
  rewriter.restoreInsertionPoint(insertPoint);

  // Load the UTF-8 bytes to pass as a value.
  return rewriter.create<IREE::VM::ConstRefRodataOp>(loc, rodataOp);
}

Optional<SmallVector<Value, 4>> rewriteAttrToOperands(
    Location loc, Attribute attrValue, Type inputType,
    ConversionPatternRewriter &rewriter) {
  if (auto intAttr = attrValue.dyn_cast<IntegerAttr>()) {
    // NOTE: we intentionally go to std.constant ops so that the standard
    // conversions can do their job. If we want to remove the dependency
    // from standard ops in the future we could instead go directly to
    // one of the vm constant ops.
    auto constValue = rewriter.createOrFold<mlir::ConstantOp>(
        loc, inputType,
        IntegerAttr::get(inputType,
                         APInt(32, static_cast<int32_t>(intAttr.getInt()))));
    return {{constValue}};
  } else if (auto elementsAttr = attrValue.dyn_cast<DenseIntElementsAttr>()) {
    SmallVector<Value, 4> elementValues;
    elementValues.reserve(elementsAttr.getNumElements());
    for (auto intAttr : elementsAttr.getAttributeValues()) {
      elementValues.push_back(rewriter.createOrFold<mlir::ConstantOp>(
          loc, elementsAttr.getType().getElementType(), intAttr));
    }
    return elementValues;
  } else if (auto arrayAttr = attrValue.dyn_cast<ArrayAttr>()) {
    SmallVector<Value, 4> allValues;
    for (auto elementAttr : arrayAttr) {
      auto flattenedValues =
          rewriteAttrToOperands(loc, elementAttr, inputType, rewriter);
      if (!flattenedValues) return llvm::None;
      allValues.append(flattenedValues->begin(), flattenedValues->end());
    }
    return allValues;
  } else if (auto strAttr = attrValue.dyn_cast<StringAttr>()) {
    return {{createStringTableValue(loc, strAttr, inputType, rewriter)}};
  }

  // This may be a custom dialect type. As we can't trivially access the storage
  // of these we need to ask the dialect to do it for us.
  auto *conversionInterface =
      attrValue.getDialect()
          .getRegisteredInterface<VMConversionDialectInterface>();
  if (conversionInterface) {
    bool anyFailed = false;
    SmallVector<Value, 4> allValues;
    if (auto tupleType = inputType.dyn_cast<TupleType>()) {
      // Custom dialect type maps into a tuple; we expect 1:1 tuple elements to
      // attribute storage elements.
      auto tupleTypes = llvm::to_vector<4>(tupleType.getTypes());
      int ordinal = 0;
      conversionInterface->walkAttributeStorage(
          attrValue, [&](Attribute elementAttr) {
            if (anyFailed) return;
            auto elementType = tupleTypes[ordinal++];
            auto flattenedValues =
                rewriteAttrToOperands(loc, elementAttr, elementType, rewriter);
            if (!flattenedValues) {
              anyFailed = true;
              return;
            }
            allValues.append(flattenedValues->begin(), flattenedValues->end());
          });
    } else {
      // Custom dialect type maps into zero or more input types (ala arrays).
      conversionInterface->walkAttributeStorage(
          attrValue, [&](Attribute elementAttr) {
            if (anyFailed) return;
            auto flattenedValues =
                rewriteAttrToOperands(loc, elementAttr, inputType, rewriter);
            if (!flattenedValues) {
              anyFailed = true;
              return;
            }
            allValues.append(flattenedValues->begin(), flattenedValues->end());
          });
    }
    if (anyFailed) return llvm::None;
    return allValues;
  }

  emitError(loc) << "unsupported attribute encoding: " << attrValue.getType();
  return llvm::None;
}

}  // namespace detail

}  // namespace iree_compiler
}  // namespace mlir
