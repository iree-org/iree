// Copyright 2021 Google LLC
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

#include "iree/compiler/Dialect/Modules/VMVX/Conversion/VMVXToVM/ConvertVMVXToVM.h"

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXOps.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

// Erases an op. This should only be used for ops that are legalized away
// as part of lowering (i.e. tagging or metadata ops that are unrepresentable
// in the VM dialect).
class EraseNonVMOp : public ConversionPattern {
 public:
  EraseNonVMOp(StringRef rootName, MLIRContext *ctx)
      : ConversionPattern(rootName, 0, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

// VMVX -> VM import conversion base for generic ops.
// Handles signatures with integers, VM types, or simple buffers.
template <typename T, typename Adaptor = typename T::Adaptor>
class VMVXImportOpConversion : public OpConversionPattern<T> {
 public:
  VMVXImportOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                         TypeConverter &typeConverter, StringRef importName)
      : OpConversionPattern<T>(context),
        importSymbols(importSymbols),
        typeConverter(typeConverter),
        importName(importName) {}

  LogicalResult matchAndRewrite(
      T op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    std::string importFqName = importName + getImportSuffix(op);
    auto importOp =
        importSymbols.template lookup<IREE::VM::ImportOp>(importFqName);
    if (!importOp) {
      op.emitError() << "failed to resolve VM function import for "
                     << importFqName;
      return failure();
    }
    return rewriteToCall(op, Adaptor{operands}, importOp, typeConverter,
                         rewriter);
  }

 protected:
  virtual std::string getImportSuffix(T op) const { return ""; }

  std::string getSizedTypeStr(Type elementType) const {
    int bitWidth = elementType.getIntOrFloatBitWidth();
    // Widen i1 -> i8 to match the VM type conversion.
    if (bitWidth == 1) {
      bitWidth = 8;
    }
    return "x" + std::to_string(bitWidth);
  }

  std::string getTypedTypeStr(Type type, bool forceUnsigned = false) const {
    Type elementType = type;
    auto shapedType = type.dyn_cast<ShapedType>();
    if (shapedType) {
      elementType = shapedType.getElementType();
    }

    std::string typePrefix = "x";
    if (elementType.isa<FloatType>()) {
      typePrefix = "f";
    } else if (elementType.isSignlessInteger()) {
      typePrefix = forceUnsigned ? "u" : "i";
    }

    int bitWidth = elementType.getIntOrFloatBitWidth();
    // Widen i1 -> i8 to match the VM type conversion.
    if (bitWidth == 1) {
      bitWidth = 8;
    }
    return typePrefix + std::to_string(bitWidth);
  }

 private:
  SymbolTable &importSymbols;
  TypeConverter &typeConverter;
  std::string importName;
};
#define VMVX_IMPORT_OP(op_type, op_mnemonic)        \
  patterns.insert<VMVXImportOpConversion<op_type>>( \
      context, importSymbols, typeConverter, op_mnemonic);

}  // namespace

void populateVMVXToVMPatterns(MLIRContext *context,
                              TypeConverter &typeConverter,
                              SymbolTable &importSymbols,
                              OwningRewritePatternList &patterns) {}

}  // namespace iree_compiler
}  // namespace mlir
