// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::VectorExt;

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtEnums.cpp.inc" // IWYU pragma: keep

#define GET_ATTRDEF_CLASSES
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtAttrs.cpp.inc" // IWYU pragma: keep

void IREEVectorExtDialect::initialize() {

  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtAttrs.cpp.inc"
      >();

#define GET_OP_LIST
  addOperations<
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.cpp.inc"
      >();
}

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.cpp.inc"

// Parses an attribute with syntax
// <"BatchX"<"VecX", 2>, 4>
Attribute PerDimLayoutAttr::parse(AsmParser &parser, Type type) {
  SmallVector<std::string> dimNames;
  SmallVector<int64_t> dimShapes;
  std::string name;
  while (!(parser.parseOptionalLess() || parser.parseOptionalString(&name))) {
    dimNames.push_back(name);
  }
  int64_t dim;
  while (!(parser.parseOptionalComma() || parser.parseInteger(dim) ||
           parser.parseGreater())) {
    dimShapes.push_back(dim);
  }
  std::reverse(dimShapes.begin(), dimShapes.end());
  return PerDimLayoutAttr::get(parser.getContext(), dimNames, dimShapes);
}

void PerDimLayoutAttr::print(AsmPrinter &printer) const {
  for (auto label : getLabels())
    printer << "<" << label;
  for (auto shape : llvm::reverse(getShapes()))
    printer << ", " << shape << ">";
}

// Parses an attribute with syntax
// #layout<<"BatchX"<"VecX", 2>, 4>, <"BatchY"<"VecZ", 4>,2>>>
Attribute LayoutAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};
  SmallVector<PerDimLayoutAttr> layout;
  PerDimLayoutAttr perDimLayout;
  while (!(parser.parseAttribute<PerDimLayoutAttr>(perDimLayout, type))) {
    layout.push_back(perDimLayout);
    if (parser.parseOptionalComma())
      break;
  }
  if ((parser.parseGreater()))
    return {};
  return LayoutAttr::get(parser.getContext(), layout);
}

static void printArray(AsmPrinter &printer,
                       ArrayRef<PerDimLayoutAttr> layouts) {
  printer << "<";
  for (auto layout : llvm::enumerate(layouts)) {
    printer << layout.value();
    if (layout.index() < layouts.size() - 1)
      printer << ", ";
  }
  printer << ">";
}

void LayoutAttr::print(AsmPrinter &printer) const {
  printArray(printer, getLayouts());
}
