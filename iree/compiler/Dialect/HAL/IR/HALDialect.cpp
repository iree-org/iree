// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"

#include "iree/compiler/Dialect/HAL/Conversion/HALToHAL/ConvertHALToHAL.h"
#include "iree/compiler/Dialect/HAL/Conversion/HALToVM/ConvertHALToVM.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/IR/LoweringConfig.h"
#include "iree/compiler/Dialect/HAL/hal.imports.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

namespace {

// Used for custom printing support.
struct HALOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  /// Hooks for getting an alias identifier alias for a given symbol, that is
  /// not necessarily a part of this dialect. The identifier is used in place of
  /// the symbol when printing textual IR. These aliases must not contain `.` or
  /// end with a numeric digit([0-9]+). Returns success if an alias was
  /// provided, failure otherwise.
  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (auto targetAttr = attr.dyn_cast<DeviceTargetAttr>()) {
      os << "device_target_" << targetAttr.getSymbolNameFragment();
      return AliasResult::OverridableAlias;
    } else if (auto targetAttr = attr.dyn_cast<ExecutableTargetAttr>()) {
      os << "executable_target_" << targetAttr.getSymbolNameFragment();
      return AliasResult::OverridableAlias;
    } else if (attr.isa<LoweringConfig>()) {
      os << "config";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};

// Used to control inlining behavior.
struct HALInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    // Sure!
    return true;
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    // Sure!
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    // Sure!
    return true;
  }
};

class HALToVMConversionInterface : public VMConversionDialectInterface {
 public:
  using VMConversionDialectInterface::VMConversionDialectInterface;

  OwningModuleRef parseVMImportModule() const override {
    return mlir::parseSourceString(StringRef(iree_hal_imports_create()->data,
                                             iree_hal_imports_create()->size),
                                   getDialect()->getContext());
  }

  void populateVMConversionPatterns(
      SymbolTable &importSymbols, OwningRewritePatternList &patterns,
      TypeConverter &typeConverter) const override {
    populateHALToHALPatterns(getDialect()->getContext(), patterns,
                             typeConverter);
    populateHALToVMPatterns(getDialect()->getContext(), importSymbols, patterns,
                            typeConverter);
  }

  void walkAttributeStorage(
      Attribute attr,
      const function_ref<void(Attribute elementAttr)> &fn) const override {
    if (auto structAttr = attr.dyn_cast<DescriptorSetLayoutBindingAttr>()) {
      structAttr.walkStorage(fn);
    }
  }
};

}  // namespace

HALDialect::HALDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<HALDialect>()) {
  context->loadDialect<IREE::Util::UtilDialect>();

  registerAttributes();
  registerTypes();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/HAL/IR/HALOps.cpp.inc"
      >();
  addInterfaces<HALInlinerInterface, HALOpAsmInterface,
                HALToVMConversionInterface>();
}

//===----------------------------------------------------------------------===//
// Dialect hooks
//===----------------------------------------------------------------------===//

Operation *HALDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  if (type.isa<IndexType>()) {
    // Some folders materialize raw index types, which just become std
    // constants.
    return builder.create<mlir::ConstantIndexOp>(
        loc, value.cast<IntegerAttr>().getValue().getSExtValue());
  }
  return nullptr;
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
