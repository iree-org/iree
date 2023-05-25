// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"

#include "iree/compiler/Dialect/HAL/Conversion/HALToVM/Patterns.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/hal.imports.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"
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
    if (auto targetAttr = llvm::dyn_cast<DeviceTargetAttr>(attr)) {
      os << "device_target_" << targetAttr.getSymbolNameFragment();
      return AliasResult::OverridableAlias;
    } else if (auto targetAttr = llvm::dyn_cast<ExecutableTargetAttr>(attr)) {
      os << "executable_target_" << targetAttr.getSymbolNameFragment();
      return AliasResult::OverridableAlias;
    } else if (auto layoutAttr = llvm::dyn_cast<PipelineLayoutAttr>(attr)) {
      os << "pipeline_layout";
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
                       IRMapping &valueMapping) const final {
    // Sure!
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    // Sure!
    return true;
  }
};

class HALToVMConversionInterface : public VMConversionDialectInterface {
 public:
  using VMConversionDialectInterface::VMConversionDialectInterface;

  OwningOpRef<mlir::ModuleOp> parseVMImportModule() const override {
    return mlir::parseSourceString<mlir::ModuleOp>(
        StringRef(iree_hal_imports_create()->data,
                  iree_hal_imports_create()->size),
        getDialect()->getContext());
  }

  void populateVMConversionPatterns(
      SymbolTable &importSymbols, RewritePatternSet &patterns,
      ConversionTarget &conversionTarget,
      TypeConverter &typeConverter) const override {
    populateHALToVMPatterns(getDialect()->getContext(), importSymbols, patterns,
                            typeConverter);
  }

  LogicalResult walkAttributeStorage(
      Attribute attr,
      const function_ref<void(Attribute elementAttr)> &fn) const override {
    MLIRContext *context = attr.getContext();
    // TODO(benvanik): remove this interface or make it an attr interface.
    if (auto bindingAttr =
            llvm::dyn_cast<IREE::HAL::DescriptorSetBindingAttr>(attr)) {
      fn(IntegerAttr::get(IndexType::get(context),
                          APInt(64, bindingAttr.getOrdinal())));
      fn(IREE::HAL::DescriptorTypeAttr::get(context, bindingAttr.getType()));
      fn(IREE::HAL::DescriptorFlagsAttr::get(
          context,
          bindingAttr.getFlags().value_or(IREE::HAL::DescriptorFlags::None)));
      return success();
    }
    if (auto dtAttr = llvm::dyn_cast<IREE::HAL::DescriptorTypeAttr>(attr)) {
      // Repack as a normal integer attribute.
      fn(IntegerAttr::get(IntegerType::get(context, 32),
                          APInt(32, static_cast<uint32_t>(dtAttr.getValue()))));
      return success();
    }
    return failure();
  }
};

}  // namespace

HALDialect::HALDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<HALDialect>()) {
  context->loadDialect<mlir::cf::ControlFlowDialect>();
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
  if (llvm::isa<IndexType>(type)) {
    // Some folders materialize raw index types, which just become std
    // constants.
    return builder.create<mlir::arith::ConstantIndexOp>(
        loc, llvm::cast<IntegerAttr>(value).getValue().getSExtValue());
  }
  return nullptr;
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
