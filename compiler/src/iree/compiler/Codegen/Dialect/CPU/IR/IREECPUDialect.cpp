// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.cpp.inc"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"

namespace mlir::iree_compiler::IREE::CPU {

struct IREECPUDialectOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (isa<IREE::CPU::LoweringConfigAttr>(attr)) {
      os << "config";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};

void IREECPUDialect::initialize() {
  registerAttributes();
  getContext()->getOrLoadDialect<IREE::Codegen::IREECodegenDialect>();
  addInterfaces<IREECPUDialectOpAsmInterface>();
}

} // namespace mlir::iree_compiler::IREE::CPU
