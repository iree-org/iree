// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include <numeric>
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.cpp.inc"

using namespace mlir::iree_compiler::IREE::VectorExt;

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtAttrInterfaces.cpp.inc"

namespace mlir::iree_compiler::IREE::VectorExt {

struct IREEVectorExtDialectOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (llvm::isa<LayoutAttr>(attr)) {
      os << "layout";
      return AliasResult::OverridableAlias;
    }
    if (llvm::isa<NestedLayoutAttr>(attr)) {
      os << "nested";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};

void IREEVectorExtDialect::initialize() {
  addInterfaces<IREEVectorExtDialectOpAsmInterface>();
  registerAttributes();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.cpp.inc"
      >();
}

} // namespace mlir::iree_compiler::IREE::VectorExt
