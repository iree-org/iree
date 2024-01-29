// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/IR/VMDialect.h"

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/IR/VMTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir::iree_compiler::IREE::VM {

#include "iree/compiler/Dialect/VM/IR/VMOpInterfaces.cpp.inc" // IWYU pragma: keep

// Fallback asm printer for ops that do not define their own. See op-specific
// printers in the op implementations.
struct VMDialect::VMOpAsmInterface
    : public OpAsmOpInterface::FallbackModel<VMOpAsmInterface> {
  VMOpAsmInterface() = default;

  static bool classof(Operation *op) { return true; }

  void getAsmResultNames(Operation *op, OpAsmSetValueNameFn setNameFn) const {
    if (op->getNumResults() == 0) {
      return;
    }

    SmallString<32> osBuffer;
    llvm::raw_svector_ostream os(osBuffer);

    if (llvm::isa<VectorType>(op->getResult(0).getType())) {
      os << "v";
    }
    if (auto refType =
            llvm::dyn_cast<IREE::VM::RefType>(op->getResult(0).getType())) {
      if (llvm::isa<BufferType>(refType.getObjectType())) {
        os << "buffer";
      } else if (llvm::isa<ListType>(refType.getObjectType())) {
        os << "list";
      } else {
        os << "ref";
      }
    }

    setNameFn(op->getResult(0), os.str());
  }

  void getAsmBlockArgumentNames(Operation *op, Region &region,
                                OpAsmSetValueNameFn setNameFn) const {}
  void getAsmBlockNames(Operation *op, OpAsmSetBlockNameFn setNameFn) const {}

  static StringRef getDefaultDialect() { return ""; }
};

namespace {

// Used to control inlining behavior.
struct VMInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    // Check the inlining policy specified on the callable first.
    if (auto inliningPolicy =
            callable->getAttrOfType<IREE::Util::InliningPolicyAttrInterface>(
                "inlining_policy")) {
      if (!inliningPolicy.isLegalToInline(call, callable))
        return false;
    }
    // Sure!
    return true;
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    // Sure!
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    // Sure!
    return true;
  }

  void handleTerminator(Operation *op, Block *newDest) const final {
    auto returnOp = dyn_cast<VM::ReturnOp>(op);
    if (!returnOp) {
      return;
    }

    // Replace the return with a branch to the dest.
    OpBuilder builder(op);
    builder.create<VM::BranchOp>(op->getLoc(), newDest, returnOp.getOperands());
    op->erase();
  }

  void handleTerminator(Operation *op, ValueRange valuesToReplace) const final {
    // TODO(benvanik): handle other terminators (break/etc).
    auto returnOp = dyn_cast<VM::ReturnOp>(op);
    if (!returnOp) {
      return;
    }

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToReplace.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands())) {
      valuesToReplace[it.index()].replaceAllUsesWith(it.value());
    }
  }

  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const override {
    return nullptr;
  }
};

struct VMFolderInterface : public DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  bool shouldMaterializeInto(Region *region) const override {
    // TODO(benvanik): redirect to scope.
    return false;
  }
};

} // namespace

VMDialect::VMDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<VMDialect>()),
      fallbackOpAsmInterface(new VMOpAsmInterface) {
  context->loadDialect<IREE::Util::UtilDialect>();

  registerAttributes();
  registerTypes();
  addInterfaces<VMInlinerInterface, VMFolderInterface>();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/VM/IR/VMOps.cpp.inc" // IWYU pragma: keep
      >();
}

VMDialect::~VMDialect() { delete fallbackOpAsmInterface; }

// Provides a hook for op interface.
void *VMDialect::getRegisteredInterfaceForOp(mlir::TypeID interface,
                                             mlir::OperationName opName) {
  if (interface == TypeID::get<mlir::OpAsmOpInterface>()) {
    return fallbackOpAsmInterface;
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Type printing and parsing
//===----------------------------------------------------------------------===//

Type VMDialect::parseType(DialectAsmParser &parser) const {
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  llvm::StringRef spec = parser.getFullSymbolSpec();
  if (spec.consume_front("buffer")) {
    return IREE::VM::RefType::getChecked(
        IREE::VM::BufferType::get(loc.getContext()), loc);
  } else if (spec.consume_front("list")) {
    if (!spec.consume_front("<") || !spec.consume_back(">")) {
      parser.emitError(parser.getCurrentLocation())
          << "malformed list type '" << parser.getFullSymbolSpec() << "'";
      return Type();
    }
    Type elementType;
    if (spec == "?") {
      elementType = OpaqueType::get(getContext());
    } else {
      // Make sure to pass a null-terminated string to the type parser.
      elementType = mlir::parseType(spec.str(), getContext());
    }
    if (!elementType) {
      parser.emitError(parser.getCurrentLocation())
          << "invalid list element type specification: '"
          << parser.getFullSymbolSpec() << "'";
      return Type();
    }
    return IREE::VM::RefType::getChecked(
        IREE::VM::ListType::getChecked(elementType, loc), loc);
  } else if (spec.consume_front("ref")) {
    if (!spec.consume_front("<") || !spec.consume_back(">")) {
      parser.emitError(parser.getCurrentLocation())
          << "malformed ref type '" << parser.getFullSymbolSpec() << "'";
      return Type();
    }
    if (spec == "?") {
      // Special case for opaque types (where any type can match).
      return IREE::VM::RefType::getChecked(
          IREE::VM::OpaqueType::get(getContext()), loc);
    }
    // Make sure to pass a null-terminated string to the type parser.
    auto objectType = mlir::parseType(spec.str(), getContext());
    if (!objectType) {
      parser.emitError(parser.getCurrentLocation())
          << "invalid ref object type specification: '"
          << parser.getFullSymbolSpec() << "'";
      return Type();
    }
    return IREE::VM::RefType::getChecked(objectType, loc);
  } else if (spec == "opaque") {
    return IREE::VM::OpaqueType::get(getContext());
  }
  emitError(loc, "unknown VM type: ") << spec;
  return Type();
}

void VMDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (auto refType = llvm::dyn_cast<IREE::VM::RefType>(type)) {
    auto objectType = refType.getObjectType();
    if (auto bufferType = llvm::dyn_cast<IREE::VM::BufferType>(objectType)) {
      printType(bufferType, os);
    } else if (auto listType = llvm::dyn_cast<IREE::VM::ListType>(objectType)) {
      printType(listType, os);
    } else if (llvm::isa<IREE::VM::OpaqueType>(objectType)) {
      os << "ref<?>";
    } else {
      os << "ref<" << objectType << ">";
    }
  } else if (llvm::isa<IREE::VM::OpaqueType>(type)) {
    os << "opaque";
  } else if (llvm::isa<IREE::VM::BufferType>(type)) {
    os << "buffer";
  } else if (auto listType = llvm::dyn_cast<IREE::VM::ListType>(type)) {
    os << "list<";
    if (llvm::isa<OpaqueType>(listType.getElementType())) {
      os << "?";
    } else {
      os << listType.getElementType();
    }
    os << ">";
  } else {
    assert(false && "unhandled VM type");
  }
}

//===----------------------------------------------------------------------===//
// Dialect hooks
//===----------------------------------------------------------------------===//

Operation *VMDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                          Type type, Location loc) {
  auto typedValue = dyn_cast<TypedAttr>(value);
  if (!typedValue)
    return nullptr;

  if (ConstI32Op::isBuildableWith(typedValue, type)) {
    auto convertedValue = ConstI32Op::convertConstValue(typedValue);
    if (llvm::cast<IntegerAttr>(convertedValue).getValue() == 0) {
      return builder.create<VM::ConstI32ZeroOp>(loc);
    }
    return builder.create<VM::ConstI32Op>(loc, convertedValue);
  } else if (ConstI64Op::isBuildableWith(typedValue, type)) {
    auto convertedValue = ConstI64Op::convertConstValue(typedValue);
    if (llvm::cast<IntegerAttr>(convertedValue).getValue() == 0) {
      return builder.create<VM::ConstI64ZeroOp>(loc);
    }
    return builder.create<VM::ConstI64Op>(loc, convertedValue);
  } else if (ConstF32Op::isBuildableWith(typedValue, type)) {
    auto convertedValue = ConstF32Op::convertConstValue(typedValue);
    if (llvm::cast<FloatAttr>(convertedValue).getValue().isZero()) {
      return builder.create<VM::ConstF32ZeroOp>(loc);
    }
    return builder.create<VM::ConstF32Op>(loc, convertedValue);
  } else if (ConstF64Op::isBuildableWith(typedValue, type)) {
    auto convertedValue = ConstF64Op::convertConstValue(typedValue);
    if (llvm::cast<FloatAttr>(convertedValue).getValue().isZero()) {
      return builder.create<VM::ConstF64ZeroOp>(loc);
    }
    return builder.create<VM::ConstF64Op>(loc, convertedValue);
  } else if (llvm::isa<IREE::VM::RefType>(type)) {
    // The only constant type we support for refs is null so we can just
    // emit that here.
    // TODO(benvanik): relace unit attr with a proper null ref attr.
    return builder.create<VM::ConstRefZeroOp>(loc, type);
  }
  // TODO(benvanik): handle other constant value types.
  return nullptr;
}

} // namespace mlir::iree_compiler::IREE::VM
