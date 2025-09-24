// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::LinalgExt;

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtAttrs.cpp.inc" // IWYU pragma: keep

// Used to control inlining behavior.
namespace {
struct IREELinalgExtInlinerInterface : public DialectInlinerInterface {
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

struct IREELinalgExtDialectOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
};

} // namespace

void IREELinalgExtDialect::initialize() {
  addInterfaces<IREELinalgExtInlinerInterface>();

  addInterfaces<IREELinalgExtDialectOpAsmInterface>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtAttrs.cpp.inc"
      >();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp.inc"
      >();
}

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.cpp.inc"

//===---------------------------------------------------------------------===//
// iree_linalg_ext.split_reduction_mapping
//===---------------------------------------------------------------------===//

int64_t SplitReductionMappingAttr::getMappingId() const {
  return getDimension();
}

bool SplitReductionMappingAttr::isLinearMapping() const { return false; }

int64_t SplitReductionMappingAttr::getRelativeIndex() const {
  return getMappingId();
}

//===---------------------------------------------------------------------===//
// These attributes represent user-hints for certain optimizations to kick in
//===---------------------------------------------------------------------===//

/// Split reduction setter/getter methods.
static const char kSplitReductionAttribute[] =
    "iree_linalg_ext.split_reduction";

namespace mlir::iree_compiler::IREE::LinalgExt {

bool SplitReductionMappingAttr::operator<(
    const SplitReductionMappingAttr &rhs) const {
  return getDimension() < rhs.getDimension();
}

LogicalResult
SplitReductionMappingAttr::verifyAttrList(MLIRContext *context, Location loc,
                                          ArrayRef<Attribute> attrs,
                                          bool emitDiagnosticErrors) {
  if (attrs.empty()) {
    return success();
  }

  auto emitErrorFn = mlir::detail::getDefaultDiagnosticEmitFn(loc);
  auto emitError = [&](std::string message) -> LogicalResult {
    if (emitDiagnosticErrors) {
      return emitErrorFn() << message;
    }
    return failure();
  };
  SmallVector<SplitReductionMappingAttr> mappingAttrs;
  llvm::SmallDenseSet<SplitReductionMappingAttr> attrSet;
  for (auto attr : attrs) {
    auto typedAttr = dyn_cast_or_null<SplitReductionMappingAttr>(attr);
    if (!typedAttr) {
      return emitError("expected all the mapping attribute to be of "
                       "`SplitReductionMappingAttr` type");
    }
    if (attrSet.contains(typedAttr)) {
      return emitError("Illegal to repeat mapping specification");
    }
    attrSet.insert(typedAttr);
    mappingAttrs.push_back(typedAttr);
  }

  llvm::sort(mappingAttrs);

  // The elements need to start from 0 and be in ascending order without gaps.
  for (auto [index, attr] : llvm::enumerate(mappingAttrs)) {
    if (attr.getDimension() != index) {
      return emitError(llvm::formatv(
          "missing dimension {0} in mapping attribute list", index));
    }
  }

  return success();
}

void setSplitReductionAttribute(Operation *op, ArrayRef<int64_t> splitSize) {
  MLIRContext *context = op->getContext();
  auto indexType = IndexType::get(context);
  auto attrVec = llvm::map_to_vector(splitSize, [&](int64_t v) -> Attribute {
    return IntegerAttr::get(indexType, v);
  });
  op->setAttr(kSplitReductionAttribute, ArrayAttr::get(context, attrVec));
}

std::optional<SmallVector<int64_t>> getSplitReductionSizes(Operation *op) {
  auto arrayAttr = op->getAttrOfType<ArrayAttr>(kSplitReductionAttribute);
  if (!arrayAttr) {
    return std::nullopt;
  }
  SmallVector<int64_t> tileSizes;
  tileSizes.reserve(arrayAttr.size());
  for (auto attr : arrayAttr) {
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (!intAttr) {
      return std::nullopt;
    }
    tileSizes.push_back(intAttr.getInt());
  }
  return tileSizes;
}
} // namespace mlir::iree_compiler::IREE::LinalgExt
