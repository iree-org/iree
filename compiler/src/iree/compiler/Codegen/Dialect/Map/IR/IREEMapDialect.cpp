// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapDialect.h"

#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapAttrs.h"
#include "mlir/IR/DialectImplementation.h"

#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapDialect.cpp.inc"

namespace mlir::iree_compiler::IREE::Map {

struct IREEMapDialectOpAsmInterface final : OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (isa<PackMapAttr>(attr)) {
      os << "pack_map";
      return AliasResult::OverridableAlias;
    }
    if (isa<PackLayoutAttr>(attr)) {
      os << "pack_layout";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};

void IREEMapDialect::initialize() {
  addInterfaces<IREEMapDialectOpAsmInterface>();
  registerAttributes();
}

} // namespace mlir::iree_compiler::IREE::Map
