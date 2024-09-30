// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.cpp.inc"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"

namespace mlir::iree_compiler::IREE::GPU {

struct IREEGPUDialectOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (llvm::isa<LoweringConfigAttr>(attr)) {
      os << "config";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};

void IREEGPUDialect::initialize() {
  registerAttributes();
  addInterfaces<IREEGPUDialectOpAsmInterface>();

  addOperations<
#define GET_OP_LIST
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.cpp.inc"
      >();
}

} // namespace mlir::iree_compiler::IREE::GPU
