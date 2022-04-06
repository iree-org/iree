// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TransformDialectExtensions.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/TransformOpInterface.h"
#include "iree-dialects/Transforms/Functional.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

// Note: with the recent TypeID changes, hiding these classes inside an
// anonymous namespace would require specific `MLIR_DECLARE_EXPLICIT_TYPE_ID`
// for each class.

// namespace {

/// Transform defined outside of the iree_linalg_transform dialect.
// TODO: Move to tablegen. Until this stabilizes upstream, simple C++ is enough.
class RewriteLinalgExtInParallelToHALOp
    : public Op<RewriteLinalgExtInParallelToHALOp,
                linalg::transform::TransformOpInterface::Trait> {
 public:
  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static constexpr llvm::StringLiteral getOperationName() {
    return llvm::StringLiteral(
        "iree_linalg_transform.rewrite_iree_linalg_ext_in_parallel_to_hal");
  }

  Value target() { return getOperation()->getOperand(0); }

  LogicalResult apply(linalg::transform::TransformResults &results,
                      linalg::transform::TransformState &state) {
    iree_compiler::IREE::LinalgExt::InParallelOpToHALRewriter pattern(
        this->getContext());
    ArrayRef<Operation *> targets = state.getPayloadOps(target());
    return functional::applyReturningPatternAt(
        pattern,
        cast<iree_compiler::IREE::LinalgExt::InParallelOp>(targets.front()));
  }

  // let assemblyFormat = "$target attr-dict";
  static ParseResult parse(OpAsmParser &parser, OperationState &state) {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand) ||
        parser.parseOptionalAttrDict(state.attributes) ||
        parser.resolveOperand(operand,
                              pdl::OperationType::get(parser.getContext()),
                              state.operands))
      return failure();
    return success();
  }

  // let assemblyFormat = "$target attr-dict";
  void print(OpAsmPrinter &printer) {
    printer << target() << ' ';
    printer.printOptionalAttrDict((*this)->getAttrs());
  }
};

/// Test extension of the Transform dialect. Registers additional ops and
/// declares PDL as dependent dialect since the additional ops are using PDL
/// types for operands and results.
class LinalgTransformDialectExtension
    : public mlir::linalg::transform::TransformDialectExtension<
          LinalgTransformDialectExtension> {
 public:
  LinalgTransformDialectExtension() {
    declareDependentDialect<pdl::PDLDialect>();
    registerTransformOp<RewriteLinalgExtInParallelToHALOp>();
    // TODO: hook up to Tablegen.
    //     registerTransformOps<
    // #define GET_OP_LIST
    // #include "LinalgTransformDialectExtension.cpp.inc"
    //         >();
  }
};

// } // namespace anonymous

void mlir::linalg::transform::registerLinalgTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<LinalgTransformDialectExtension>();
}