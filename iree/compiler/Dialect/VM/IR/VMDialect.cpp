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

#include "iree/compiler/Dialect/VM/IR/VMDialect.h"

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/IR/VMTypes.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

#include "iree/compiler/Dialect/VM/IR/VMOpInterface.cpp.inc"

static DialectRegistration<VMDialect> vm_dialect;

namespace {

// Used for custom printing support.
struct VMOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  void getAsmResultNames(Operation *op,
                         OpAsmSetValueNameFn setNameFn) const final {
    SmallString<32> osBuffer;
    llvm::raw_svector_ostream os(osBuffer);

    // TODO(b/143187291): tablegen this by adding a value name prefix field.
    if (op->getResult(0).getType().isa<VectorType>()) {
      os << "v";
    }
    if (auto globalLoadOp = dyn_cast<GlobalLoadI32Op>(op)) {
      os << globalLoadOp.global();
    } else if (auto globalLoadOp = dyn_cast<GlobalLoadRefOp>(op)) {
      os << globalLoadOp.global();
    } else if (isa<ConstRefZeroOp>(op)) {
      os << "null";
    } else if (isa<ConstI32ZeroOp>(op)) {
      os << "zero";
    } else if (auto constOp = dyn_cast<ConstI32Op>(op)) {
      if (auto intAttr = constOp.value().dyn_cast<IntegerAttr>()) {
        if (intAttr.getValue() == 0) {
          os << "zero";
        } else {
          os << 'c' << intAttr.getValue();
        }
      } else {
        os << 'c';
      }
    } else if (auto rodataOp = dyn_cast<ConstRefRodataOp>(op)) {
      os << rodataOp.rodata();
    } else if (auto refType =
                   op->getResult(0).getType().dyn_cast<IREE::VM::RefType>()) {
      if (refType.getObjectType().isa<ListType>()) {
        os << "list";
      } else {
        os << "ref";
      }
    } else if (isa<CmpEQI32Op>(op)) {
      os << "eq";
    } else if (isa<CmpNEI32Op>(op)) {
      os << "ne";
    } else if (isa<CmpLTI32SOp>(op)) {
      os << "slt";
    } else if (isa<CmpLTI32UOp>(op)) {
      os << "ult";
    } else if (isa<CmpLTEI32SOp>(op)) {
      os << "slte";
    } else if (isa<CmpLTEI32UOp>(op)) {
      os << "ulte";
    } else if (isa<CmpGTI32SOp>(op)) {
      os << "sgt";
    } else if (isa<CmpGTI32UOp>(op)) {
      os << "ugt";
    } else if (isa<CmpGTEI32SOp>(op)) {
      os << "sgte";
    } else if (isa<CmpGTEI32UOp>(op)) {
      os << "ugte";
    } else if (isa<CmpEQRefOp>(op)) {
      os << "req";
    } else if (isa<CmpNERefOp>(op)) {
      os << "rne";
    } else if (isa<CmpNZRefOp>(op)) {
      os << "rnz";
    }

    setNameFn(op->getResult(0), os.str());
  }
};

// Used to control inlining behavior.
struct VMInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Region *dest, Region *src,
                       BlockAndValueMapping &valueMapping) const final {
    // TODO(benvanik): disallow inlining across async calls.

    // Don't inline functions with the 'noinline' attribute.
    // Useful primarily for benchmarking.
    if (auto funcOp = dyn_cast<VM::FuncOp>(src->getParentOp())) {
      if (funcOp.noinline()) {
        return false;
      }
    }

    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest,
                       BlockAndValueMapping &valueMapping) const final {
    // TODO(benvanik): disallow inlining across async calls.
    return true;
  }

  void handleTerminator(Operation *op, Block *newDest) const final {
    // TODO(benvanik): handle other terminators (break/etc).
    auto returnOp = dyn_cast<VM::ReturnOp>(op);
    if (!returnOp) {
      return;
    }

    // Replace the return with a branch to the dest.
    OpBuilder builder(op);
    builder.create<VM::BranchOp>(op->getLoc(), newDest, returnOp.getOperands());
    op->erase();
  }

  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToReplace) const final {
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

struct VMFolderInterface : public OpFolderDialectInterface {
  using OpFolderDialectInterface::OpFolderDialectInterface;

  bool shouldMaterializeInto(Region *region) const override {
    // TODO(benvanik): redirect to scope.
    return false;
  }
};

}  // namespace

VMDialect::VMDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<IREE::VM::ListType, IREE::VM::OpaqueType, IREE::VM::RefType>();
  addInterfaces<VMInlinerInterface, VMOpAsmInterface, VMFolderInterface>();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/VM/IR/VMOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Type printing and parsing
//===----------------------------------------------------------------------===//

Type VMDialect::parseType(DialectAsmParser &parser) const {
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  llvm::StringRef spec = parser.getFullSymbolSpec();
  if (spec.consume_front("list")) {
    if (!spec.consume_front("<") || !spec.consume_back(">")) {
      parser.emitError(parser.getCurrentLocation())
          << "malformed list type '" << parser.getFullSymbolSpec() << "'";
      return Type();
    }
    auto elementType = mlir::parseType(spec, getContext());
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
          << "malformed ref_ptr type '" << parser.getFullSymbolSpec() << "'";
      return Type();
    }
    if (spec == "?") {
      // Special case for opaque types (where any type can match).
      return IREE::VM::RefType::getChecked(
          IREE::VM::OpaqueType::get(getContext()), loc);
    }
    auto objectType = mlir::parseType(spec, getContext());
    if (!objectType) {
      parser.emitError(parser.getCurrentLocation())
          << "invalid ref_ptr object type specification: '"
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
  switch (type.getKind()) {
    case IREE::VM::TypeKind::Ref: {
      auto objectType = type.cast<IREE::VM::RefType>().getObjectType();
      if (auto listType = objectType.dyn_cast<IREE::VM::ListType>()) {
        os << "list<" << listType.getElementType() << ">";
      } else if (objectType.isa<IREE::VM::OpaqueType>()) {
        os << "ref<?>";
      } else {
        os << "ref<" << objectType << ">";
      }
      break;
    }
    case IREE::VM::TypeKind::Opaque:
      os << "opaque";
      break;
    default:
      llvm_unreachable("unhandled VM type");
  }
}

//===----------------------------------------------------------------------===//
// Dialect hooks
//===----------------------------------------------------------------------===//

Operation *VMDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                          Type type, Location loc) {
  if (ConstI32Op::isBuildableWith(value, type)) {
    auto convertedValue = ConstI32Op::convertConstValue(value);
    if (convertedValue.cast<IntegerAttr>().getValue() == 0) {
      return builder.create<VM::ConstI32ZeroOp>(loc);
    }
    return builder.create<VM::ConstI32Op>(loc, convertedValue);
  } else if (type.isa<IREE::VM::RefType>()) {
    // The only constant type we support for ref_ptrs is null so we can just
    // emit that here.
    // TODO(b/144027097): relace unit attr with a proper null ref_ptr attr.
    return builder.create<VM::ConstRefZeroOp>(loc, type);
  }
  // TODO(benvanik): handle other constant value types.
  return nullptr;
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
