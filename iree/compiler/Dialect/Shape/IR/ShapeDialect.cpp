// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"

#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir {
namespace iree_compiler {

static DialectRegistration<ShapeDialect> base_dialect;

// Used to control inlining behavior.
struct ShapeInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation* op, Region* dest,
                       BlockAndValueMapping& valueMapping) const final {
    return true;
  }
};

// Used to control constant folding behavior as a fallback on the dialect when
// individual op folder does not match.
struct ShapeConstantFoldInterface : public DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  LogicalResult Fold(Operation* op, ArrayRef<Attribute> operands,
                     SmallVectorImpl<OpFoldResult>& results) const final {
    bool foundConstantRankedShape = false;
    for (Value result : op->getResults()) {
      auto rankedShape = result.getType().dyn_cast<Shape::RankedShapeType>();
      if (rankedShape && rankedShape.isFullyStatic()) {
        foundConstantRankedShape = true;
        results.push_back(TypeAttr::get(rankedShape));
      } else {
        results.push_back(nullptr);
      }
    }
    return success(foundConstantRankedShape);
  }
};

ShapeDialect::ShapeDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<ShapeDialect>()) {
  addTypes<Shape::RankedShapeType>();
  addInterfaces<ShapeConstantFoldInterface, ShapeInlinerInterface>();
#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.cpp.inc"
      >();
}

Operation* ShapeDialect::materializeConstant(OpBuilder& builder,
                                             Attribute value, Type type,
                                             Location loc) {
  if (auto typeAttr = value.dyn_cast<TypeAttr>()) {
    auto rankedShape = typeAttr.getValue().cast<Shape::RankedShapeType>();
    return builder.create<Shape::ConstRankedShapeOp>(loc, rankedShape);
  } else if (type.isa<IndexType>()) {
    // Some folders materialize raw index types, which just become std
    // constants.
    return builder.create<ConstantOp>(loc, type, value);
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Type parsing and printing
//===----------------------------------------------------------------------===//

static Type parseRankedShape(DialectAsmParser& parser) {
  llvm::SmallVector<int64_t, 7> dims;
  Type dimType;
  // Parse: ranked_shape<[
  if (failed(parser.parseLess()) || failed(parser.parseLSquare()))
    return nullptr;

  // Parse list of comma-separated dims, where each dim is an integer >= 0
  // or ?.
  for (bool first = true;; first = false) {
    if (!first) {
      if (failed(parser.parseOptionalComma())) break;
    }

    int64_t dim;
    OptionalParseResult optionalInteger = parser.parseOptionalInteger(dim);
    if (optionalInteger.hasValue()) {
      if (dim < 0) {
        parser.emitError(parser.getNameLoc(), "expected dim >= 0 or '?'");
        return nullptr;
      }
    } else if (succeeded(parser.parseOptionalQuestion())) {
      dim = -1;
    } else if (first) {
      // It is fine to not have a first dim.
      break;
    } else {
      parser.emitError(parser.getNameLoc(), "expected shape dim");
      return nullptr;
    }
    dims.push_back(dim);
  }
  if (failed(parser.parseRSquare())) return nullptr;

  // Parse optional: , type
  if (succeeded(parser.parseOptionalComma())) {
    if (failed(parser.parseType(dimType))) {
      return nullptr;
    }
  } else {
    dimType = parser.getBuilder().getIndexType();
  }
  if (failed(parser.parseGreater())) {
    parser.emitError(parser.getNameLoc(), "expected terminating '>'");
    return nullptr;
  }

  return Shape::RankedShapeType::getChecked(
      dims, parser.getEncodedSourceLoc(parser.getNameLoc()));
}

static void printRankedShape(Shape::RankedShapeType type,
                             DialectAsmPrinter& printer) {
  auto dims = type.getAllDims();
  printer << "ranked_shape<[";
  interleave(
      dims, printer,
      [&](int64_t dim) {
        if (dim < 0)
          printer << "?";
        else
          printer << dim;
      },
      ",");
  printer << "]>";
}

Type ShapeDialect::parseType(DialectAsmParser& parser) const {
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  llvm::StringRef spec = parser.getFullSymbolSpec();
  if (succeeded(parser.parseOptionalKeyword("ranked_shape"))) {
    return parseRankedShape(parser);
  }
  emitError(loc, "unknown Shape type: ") << spec;
  return Type();
}

void ShapeDialect::printType(Type type, DialectAsmPrinter& os) const {
  if (auto rankedShapeTy = type.dyn_cast<Shape::RankedShapeType>())
    return printRankedShape(type.cast<Shape::RankedShapeType>(), os);
  llvm_unreachable("unhandled Shape type");
}

}  // namespace iree_compiler
}  // namespace mlir
