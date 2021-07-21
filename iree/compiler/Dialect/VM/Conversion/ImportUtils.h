// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_IMPORTUTILS_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_IMPORTUTILS_H_

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Represents a fixed single-value non-variadic segment in the variadic call
// segment_sizes array.
constexpr int kFixedSingleValue = -1;

// Appends a set of vm.import ops from a module to a target VM module.
// Imports will only be added if they are not already present in the target
// module.
LogicalResult appendImportModule(IREE::VM::ModuleOp importModuleOp,
                                 ModuleOp targetModuleOp);
LogicalResult appendImportModule(StringRef importModuleSrc,
                                 ModuleOp targetModuleOp);

namespace detail {
size_t getSegmentSpanSize(Type spanType);
Optional<SmallVector<Value, 4>> rewriteAttrToOperands(
    Location loc, Attribute attrValue, Type inputType,
    ConversionPatternRewriter &rewriter);
}  // namespace detail

// Rewrites the op T to a VM call to |importOp|.
// Automatically handles type conversion and special logic for variadic operands
// and special types (such as ranked shape).
template <typename T, typename Adaptor = typename T::Adaptor>
Optional<SmallVector<Value>> rewriteToCall(
    T op, Adaptor adaptor, IREE::VM::ImportOp importOp,
    TypeConverter &typeConverter, ConversionPatternRewriter &rewriter) {
  auto *operation = op.getOperation();
  bool isOpVariadic = importOp.isVariadic();
  OperationState state{
      op.getLoc(), isOpVariadic ? IREE::VM::CallVariadicOp::getOperationName()
                                : IREE::VM::CallOp::getOperationName()};
  state.addAttributes(llvm::to_vector<4>(operation->getDialectAttrs()));
  state.addAttribute("callee", rewriter.getSymbolRefAttr(importOp));

  auto importType = importOp.getType();
  for (auto resultType : operation->getResultTypes()) {
    if (failed(typeConverter.convertType(resultType, state.types))) {
      return None;
    }
  }

  SmallVector<uint16_t, 4> segmentSizes;
  int inputSetIndex = 0;
  for (auto input : llvm::enumerate(importType.getInputs())) {
    auto inputType = input.value();
    auto inputName = importOp.getFuncArgumentName(input.index());
    if (auto attrValue = op->getAttr(inputName)) {
      auto flattenedAttrs = detail::rewriteAttrToOperands(
          op.getLoc(), attrValue, inputType, rewriter);
      if (!flattenedAttrs) return None;
      state.addOperands(*flattenedAttrs);
      if (importOp.isFuncArgumentVariadic(input.index())) {
        segmentSizes.push_back(flattenedAttrs->size() /
                               detail::getSegmentSpanSize(inputType));
      } else {
        assert(flattenedAttrs->size() == 1 &&
               "expected non-variadic attribute to have a single value");
        segmentSizes.push_back(kFixedSingleValue);
      }
    } else {
      auto oldOperands = llvm::to_vector<4>(op.getODSOperands(inputSetIndex));
      auto newOperands =
          llvm::to_vector<4>(adaptor.getODSOperands(inputSetIndex));
      ++inputSetIndex;
      if (oldOperands.size() == 1 &&
          oldOperands[0].getType().template isa<Shape::RankedShapeType>()) {
        // Expand a ranked_shape into its dimensions.
        // We need to rematerialize the static dimensions and then pass through
        // the new dynamic dimensions that we have the SSA values for.
        auto rankedShapeType = oldOperands[0]
                                   .getType()
                                   .template dyn_cast<Shape::RankedShapeType>();
        for (int i = 0; i < rankedShapeType.getRank(); ++i) {
          auto dimOp = rewriter.createOrFold<Shape::RankedDimOp>(
              op.getLoc(), oldOperands[0], i);
          state.addOperands(dimOp);
        }
        segmentSizes.push_back(rankedShapeType.getRank());
      } else {
        state.addOperands(newOperands);
        if (importOp.isFuncArgumentVariadic(input.index())) {
          segmentSizes.push_back(newOperands.size());
        } else {
          segmentSizes.push_back(kFixedSingleValue);
        }
      }
    }
  }
  if (isOpVariadic) {
    state.addAttribute(
        "segment_sizes",
        DenseIntElementsAttr::get(
            VectorType::get({static_cast<int64_t>(segmentSizes.size())},
                            rewriter.getIntegerType(16)),
            segmentSizes));
    state.addAttribute("segment_types",
                       rewriter.getArrayAttr(llvm::to_vector<4>(llvm::map_range(
                           importType.getInputs(), [&](Type type) {
                             return TypeAttr::get(type).cast<Attribute>();
                           }))));
  }

  auto *callOp = rewriter.createOperation(state);
  return SmallVector<Value>(callOp->getResults());
}

// Utility for op to vm.call conversion.
template <typename T, typename Adaptor = typename T::Adaptor>
class VMImportOpConversion : public OpConversionPattern<T> {
 public:
  VMImportOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                       TypeConverter &typeConverter, StringRef importName)
      : OpConversionPattern<T>(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult matchAndRewrite(
      T op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto results = rewriteToCall(op, Adaptor{operands}, importOp,
                                 *this->getTypeConverter(), rewriter);
    if (!results.hasValue()) return failure();
    rewriter.replaceOp(op, results.getValue());
    return success();
  }

 protected:
  mutable IREE::VM::ImportOp importOp;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_IMPORTUTILS_H_
