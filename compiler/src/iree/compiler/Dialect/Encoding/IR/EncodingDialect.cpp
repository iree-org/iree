// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/Encoding/IR/EncodingAttrs.cpp.inc"
#include "iree/compiler/Dialect/Encoding/IR/EncodingEnums.cpp.inc"
#undef GET_ATTRDEF_CLASSES

namespace mlir::iree_compiler::IREE::Encoding {

// Used to control inlining behavior.
struct IREEEncodingInlinerInterface : public DialectInlinerInterface {
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

void IREEEncodingDialect::initialize() {
  addInterfaces<IREEEncodingInlinerInterface>();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Dialect/Encoding/IR/EncodingAttrs.cpp.inc"
      >();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.cpp.inc"
      >();
}

} // namespace mlir::iree_compiler::IREE::Encoding

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.cpp.inc"
