// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VMVX/Conversion/VMVXToVM/ConvertVMVXToVM.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXOps.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
// TODO: This is a big mess and doesn't support the generality we need. Redo
// it.
template <typename T>
class VMVXImportOpConversion : public OpConversionPattern<T> {
 public:
  VMVXImportOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                         TypeConverter &typeConverter)
      : OpConversionPattern<T>(context),
        importSymbols(importSymbols),
        typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(
      T op, typename T::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    std::string importFqName = getImportFqName(op);
    auto importOp =
        importSymbols.template lookup<IREE::VM::ImportOp>(importFqName);
    if (!importOp) {
      op.emitError() << "failed to resolve VM function import for "
                     << importFqName;
      return failure();
    }
    auto results = emitCall(op, adaptor, importOp, rewriter);
    if (!results.has_value()) return failure();
    rewriter.replaceOp(op, results.value());
    return success();
  }

 protected:
  virtual std::string getImportFqName(T op) const = 0;
  virtual std::optional<SmallVector<Value>> emitCall(
      T op, typename T::Adaptor adaptor, IREE::VM::ImportOp importOp,
      ConversionPatternRewriter &rewriter) const {
    return rewriteToCall(op, adaptor, importOp, typeConverter, rewriter);
  }

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
    auto shapedType = llvm::dyn_cast<ShapedType>(type);
    if (shapedType) {
      elementType = shapedType.getElementType();
    }

    std::string typePrefix = "x";
    if (llvm::isa<FloatType>(elementType)) {
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
};

class BinaryOpConversion : public VMVXImportOpConversion<IREE::VMVX::BinaryOp> {
 public:
  using VMVXImportOpConversion::VMVXImportOpConversion;

  std::string getImportFqName(IREE::VMVX::BinaryOp op) const override {
    int rank = op.getLhsStrides().size();
    std::string name("vmvx.");
    name.append(op.getOpcode().begin(), op.getOpcode().end());
    name.append(".");
    name.append(std::to_string(rank));
    name.append("d.");
    name.append(getTypedTypeStr(op.getElementType()));
    return name;
  }
};

// Converts the vmvx.copy op to an appropriate typed import.
class CopyOpConversion : public VMVXImportOpConversion<IREE::VMVX::CopyOp> {
 public:
  using VMVXImportOpConversion::VMVXImportOpConversion;

  std::string getImportFqName(IREE::VMVX::CopyOp op) const override {
    int rank = op.getInStrides().size();
    std::string name("vmvx.copy.");
    name.append(std::to_string(rank));
    name.append("d.");
    name.append(getSizedTypeStr(op.getElementType()));
    return name;
  }
};

// Converts the vmvx.fill2d op to an appropriate typed import.
class Fill2DOpConversion : public VMVXImportOpConversion<IREE::VMVX::Fill2DOp> {
 public:
  using VMVXImportOpConversion::VMVXImportOpConversion;

  std::string getImportFqName(IREE::VMVX::Fill2DOp op) const override {
    std::string name("vmvx.fill.2d.");
    name.append(getSizedTypeStr(op.getScalar().getType()));
    return name;
  }
};

// Converts the vmvx.query_tile_sizes op to its import.
class QueryTileSizesOpConversion
    : public VMVXImportOpConversion<IREE::VMVX::QueryTileSizesOp> {
 public:
  using VMVXImportOpConversion::VMVXImportOpConversion;

  std::string getImportFqName(IREE::VMVX::QueryTileSizesOp op) const override {
    return "vmvx.query_tile_sizes.2d";
  }
};

class UnaryOpConversion : public VMVXImportOpConversion<IREE::VMVX::UnaryOp> {
 public:
  using VMVXImportOpConversion::VMVXImportOpConversion;

  std::string getImportFqName(IREE::VMVX::UnaryOp op) const override {
    int rank = op.getInStrides().size();
    std::string name("vmvx.");
    name.append(op.getOpcode().begin(), op.getOpcode().end());
    name.append(".");
    name.append(std::to_string(rank));
    name.append("d.");
    name.append(getTypedTypeStr(op.getElementType()));
    return name;
  }
};

}  // namespace

void populateVMVXToVMPatterns(MLIRContext *context,
                              ConversionTarget &conversionTarget,
                              TypeConverter &typeConverter,
                              SymbolTable &importSymbols,
                              RewritePatternSet &patterns) {
  patterns.insert<BinaryOpConversion, CopyOpConversion, Fill2DOpConversion,
                  UnaryOpConversion, QueryTileSizesOpConversion>(
      context, importSymbols, typeConverter);
}

}  // namespace iree_compiler
}  // namespace mlir
