// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp.inc"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir::iree_compiler::IREE::Codegen {

struct IREECodegenDialectOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (llvm::isa<TranslationInfoAttr>(attr)) {
      os << "translation";
      return AliasResult::OverridableAlias;
    } else if (llvm::isa<CompilationInfoAttr>(attr)) {
      os << "compilation";
      return AliasResult::OverridableAlias;
    } else if (llvm::isa<LoweringConfigAttr>(attr)) {
      os << "config";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};

void IREECodegenDialect::initialize() {
  initializeCodegenAttrs();
  addInterfaces<IREECodegenDialectOpAsmInterface>();

  addOperations<
#define GET_OP_LIST
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.cpp.inc"
      >();
}

LogicalResult
IREECodegenDialect::verifyOperationAttribute(Operation *op,
                                             NamedAttribute attribute) {
  StringRef symbol = attribute.getName().strref();
  Attribute attr = attribute.getValue();

  if (symbol == kTuningSpecEntrypointAttrName) {
    if (!llvm::isa<UnitAttr>(attr)) {
      return op->emitError("'") << symbol << "' attribute must be a UnitAttr";
    }
  } else {
    return op->emitError("found unsupported '")
           << symbol << "' attribute on operation";
  }
  return success();
}

LogicalResult IREECodegenDialect::verifyRegionArgAttribute(
    Operation *op, unsigned regionIndex, unsigned argIndex,
    NamedAttribute attribute) {
  StringRef symbol = attribute.getName().strref();
  llvm::outs() << "symbol Arg " << symbol << "\n";
  return success();
}

LogicalResult IREECodegenDialect::verifyRegionResultAttribute(
    Operation *op, unsigned /*regionIndex*/, unsigned /*resultIndex*/,
    NamedAttribute attribute) {
  StringRef symbol = attribute.getName().strref();
  llvm::outs() << "symbol Res " << symbol << "\n";
  return success();
}

} // namespace mlir::iree_compiler::IREE::Codegen
