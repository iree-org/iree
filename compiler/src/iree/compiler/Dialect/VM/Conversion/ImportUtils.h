// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_IMPORTUTILS_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_IMPORTUTILS_H_

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

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
std::optional<SmallVector<Value>>
rewriteAttrToOperands(Location loc, Attribute attrValue, Type inputType,
                      ConversionPatternRewriter &rewriter);
} // namespace detail

// Casts |value| to |targetType| ala static_cast for when the declared type
// differs from the type provided by the input dialect.
Value castToImportType(Value value, Type targetType,
                       ConversionPatternRewriter &rewriter);

// Casts |value| to |targetType| ala static_cast for when the declared return
// type of an import does not match the required output type.
Value castFromImportType(Value value, Type targetType,
                         ConversionPatternRewriter &rewriter);

// Copies known attributes from the |importOp| to the |callOp|.
// This allows for passes to quickly query the properties of the import such as
// nosideeffects.
void copyImportAttrs(IREE::VM::ImportOp importOp, Operation *callOp);

// Rewrites the op T to a VM call to |importOp|.
// Automatically handles type conversion and special logic for variadic operands
// and special types (such as ranked shape).
template <typename T, typename Adaptor = typename T::Adaptor>
std::optional<SmallVector<Value>>
rewriteToCall(T op, Adaptor adaptor, IREE::VM::ImportOp importOp,
              const TypeConverter &typeConverter,
              ConversionPatternRewriter &rewriter) {
  auto *operation = op.getOperation();
  bool isOpVariadic = importOp.isVariadic();
  OperationState state{
      op.getLoc(), isOpVariadic ? IREE::VM::CallVariadicOp::getOperationName()
                                : IREE::VM::CallOp::getOperationName()};
  state.addAttributes(llvm::to_vector(operation->getDialectAttrs()));
  state.addAttribute("callee", SymbolRefAttr::get(importOp));

  auto importType = importOp.getFunctionType();
  state.addTypes(importType.getResults());

  SmallVector<uint16_t> segmentSizes;
  int inputSetIndex = 0;
  for (auto input : llvm::enumerate(importType.getInputs())) {
    auto inputType = input.value();
    auto inputName = importOp.getFuncArgumentName(input.index());
    if (auto attrValue = op->getAttr(inputName)) {
      auto flattenedAttrs = detail::rewriteAttrToOperands(
          op.getLoc(), attrValue, inputType, rewriter);
      if (!flattenedAttrs)
        return std::nullopt;
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
      auto oldOperands = llvm::to_vector(op.getODSOperands(inputSetIndex));
      auto newOperands = llvm::to_vector(adaptor.getODSOperands(inputSetIndex));
      ++inputSetIndex;
      if (auto inputTupleType = inputType.template dyn_cast<TupleType>()) {
        // Unpack a tuple<...> from the variadic.
        // This only supports a single level of unpacking.
        if (inputTupleType.size() != newOperands.size()) {
          assert(false && "arity mismatch between tuple and variadic");
          return std::nullopt;
        }
        for (auto [newOperand, inputType] :
             llvm::zip_equal(newOperands, inputTupleType.getTypes())) {
          state.addOperands(castToImportType(newOperand, inputType, rewriter));
        }
      } else {
        for (auto &operand : newOperands) {
          state.addOperands(castToImportType(operand, inputType, rewriter));
        }
      }

      if (importOp.isFuncArgumentVariadic(input.index())) {
        segmentSizes.push_back(newOperands.size());
      } else {
        segmentSizes.push_back(kFixedSingleValue);
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
                       rewriter.getArrayAttr(llvm::map_to_vector(
                           importType.getInputs(), [&](Type type) {
                             return TypeAttr::get(type).cast<Attribute>();
                           })));
  }

  auto *callOp = rewriter.create(state);
  copyImportAttrs(importOp, callOp);

  SmallVector<Value> results;
  for (auto [result, targetType] :
       llvm::zip_equal(callOp->getResults(), operation->getResultTypes())) {
    targetType = typeConverter.convertType(targetType);
    if (!targetType)
      return std::nullopt;
    results.push_back(castFromImportType(result, targetType, rewriter));
  }
  return results;
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

  LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto results = rewriteToCall(op, adaptor, importOp,
                                 *this->getTypeConverter(), rewriter);
    if (!results.has_value())
      return failure();
    rewriter.replaceOp(op, results.value());
    return success();
  }

protected:
  mutable IREE::VM::ImportOp importOp;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_VM_CONVERSION_IMPORTUTILS_H_
