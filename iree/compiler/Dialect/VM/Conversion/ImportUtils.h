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

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_IMPORTUTILS_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_IMPORTUTILS_H_

#include "iree/compiler/Dialect/Types.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Appends a set of vm.import ops from a module to a target VM module.
// Imports will only be added if they are not already present in the target
// module.
LogicalResult appendImportModule(IREE::VM::ModuleOp importModuleOp,
                                 ModuleOp targetModuleOp);
LogicalResult appendImportModule(StringRef importModuleSrc,
                                 ModuleOp targetModuleOp);

// Utility for op to vm.call conversion.
template <typename T, typename Adaptor = typename T::OperandAdaptor>
class VMImportOpConversion : public OpConversionPattern<T> {
 public:
  VMImportOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                       TypeConverter &typeConverter, StringRef importName)
      : OpConversionPattern<T>(context), typeConverter(typeConverter) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  // Returns true if the import must be called with vm.call.variadic.
  bool isVariadic() const {
    for (int i = 0; i < importOp.getNumFuncArguments(); ++i) {
      if (importOp.isFuncArgumentVariadic(i)) return true;
    }
    return false;
  }

  PatternMatchResult matchAndRewrite(
      T op, llvm::ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (failed(rewriteToCall(op, Adaptor{operands}, rewriter))) {
      return OpConversionPattern<T>::matchFailure();
    }
    return OpConversionPattern<T>::matchSuccess();
  }

  virtual LogicalResult rewriteToCall(
      T op, Adaptor adaptor, ConversionPatternRewriter &rewriter) const {
    auto *operation = op.getOperation();
    bool isOpVariadic = isVariadic();
    OperationState state{
        op.getLoc(), isOpVariadic ? IREE::VM::CallVariadicOp::getOperationName()
                                  : IREE::VM::CallOp::getOperationName()};
    state.addAttributes(llvm::to_vector<4>(operation->getDialectAttrs()));
    state.addAttribute("callee", rewriter.getSymbolRefAttr(importOp));

    auto importType = importOp.getType();
    for (auto resultType : operation->getResultTypes()) {
      if (failed(typeConverter.convertType(resultType, state.types))) {
        return failure();
      }
    }

    SmallVector<uint8_t, 4> segmentSizes;
    int inputSetIndex = 0;
    for (auto input : llvm::enumerate(importType.getInputs())) {
      auto inputType = input.value();
      auto inputName = importOp.getFuncArgumentName(input.index());
      if (auto attrValue = op.getAttr(inputName)) {
        if (auto intAttr = attrValue.template dyn_cast<IntegerAttr>()) {
          // NOTE: we intentionally go to std.constant ops so that the standard
          // conversions can do their job. If we want to remove the dependency
          // from standard ops in the future we could instead go directly to
          // one of the vm constant ops.
          auto *constOp = rewriter.createOrFold<mlir::ConstantOp>(
              op.getLoc(), inputType, intAttr);
          state.operands.push_back(constOp);
        } else {
          op.emitOpError() << "unsupported attribute encoding: "
                           << attrValue.getType();
          return failure();
        }
        segmentSizes.push_back(-1);
      } else {
        auto operands =
            llvm::to_vector<4>(adaptor.getODSOperands(inputSetIndex++));
        state.addOperands(operands);
        if (importOp.isFuncArgumentVariadic(input.index())) {
          segmentSizes.push_back(operands.size());
        } else {
          segmentSizes.push_back(-1);
        }
      }
    }
    if (isOpVariadic) {
      state.addAttribute(
          "segment_sizes",
          DenseIntElementsAttr::get(
              VectorType::get({static_cast<int64_t>(segmentSizes.size())},
                              rewriter.getIntegerType(8)),
              segmentSizes));
      state.addAttribute(
          "segment_types",
          rewriter.getArrayAttr(llvm::to_vector<4>(
              llvm::map_range(importType.getInputs(), [&](Type type) {
                return TypeAttr::get(type).cast<Attribute>();
              }))));
    }

    auto *callOp = rewriter.createOperation(state);
    rewriter.replaceOp(op, llvm::to_vector<4>(callOp->getResults()));
    return success();
  }

 protected:
  mutable IREE::VM::ImportOp importOp;
  TypeConverter &typeConverter;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_IMPORTUTILS_H_
