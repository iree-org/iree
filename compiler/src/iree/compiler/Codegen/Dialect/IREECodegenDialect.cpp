// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"

#include "iree/compiler/Codegen/Dialect/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.cpp.inc"
#include "iree/compiler/Codegen/Dialect/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/UKernelOps.h"
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
#include "iree/compiler/Codegen/Dialect/IREECodegenOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "iree/compiler/Codegen/Dialect/UKernelOps.cpp.inc"
      >();
}

} // namespace mlir::iree_compiler::IREE::Codegen
