// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file implements a pass to emulate 16-bit brain float operations with
// 32-bit ones.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Utils/ConversionUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-emulate-bf16"

namespace mlir::iree_compiler {

namespace {

class Bf16EmulationConverter : public TypeConverter {
public:
  explicit Bf16EmulationConverter() {
    // Allow unknown types.
    addConversion([](Type ty) -> std::optional<Type> { return ty; });

    // Scalar case.
    addConversion([](FloatType ty) -> std::optional<Type> {
      if (ty.isBF16())
        return IntegerType::get(ty.getContext(), 16);
      return ty;
    });

    addConversion([this](ShapedType ty) -> std::optional<Type> {
      return ty.clone(convertType(ty.getElementType()));
    });

    addConversion([this](FunctionType ty) -> std::optional<Type> {
      SmallVector<Type> inputs;
      if (failed(convertTypes(ty.getInputs(), inputs)))
        return std::nullopt;

      SmallVector<Type> results;
      if (failed(convertTypes(ty.getResults(), results)))
        return std::nullopt;

      return FunctionType::get(ty.getContext(), inputs, results);
    });
  }
};

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//
struct ConvertHalInterfaceBindingSubspan final
    : OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newResultTy = getTypeConverter()->convertType(op.getType());
    if (!newResultTy)
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to legalize memref type: {0}", op.getType()));

    auto newOp =
        rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceBindingSubspanOp>(
            op, newResultTy, adaptor.getSet(), adaptor.getBinding(),
            adaptor.getDescriptorType(), adaptor.getByteOffset(),
            adaptor.getDynamicDims(), adaptor.getAlignmentAttr(),
            adaptor.getDescriptorFlagsAttr());
    LLVM_DEBUG(llvm::dbgs() << "Bf16Emulation: new op: " << newOp << "\n");
    (void)newOp;
    return success();
  }
};

struct ConvertMemRefAlloc final : OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy)
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to convert memref type: {0}", op.getType()));

    rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op, newTy, adaptor.getDynamicSizes(), adaptor.getSymbolOperands(),
        adaptor.getAlignmentAttr());
    return success();
  }
};

// Tries to completely convert a generic Operation.
// This will process attributes, result types, and nested regions.
struct GenericTypeConversionPattern : public ConversionPattern {
  GenericTypeConversionPattern(TypeConverter &typeConverter,
                               MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), 0, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert attributes only if this is a constant-like op.
    // This is because some ops use typed attributes for structural information
    // - like linalg ops using i64 for dimension indices - and if we converted
    // them all the ops would become invalid. This may still be too broad,
    // though, if some constant ops include attributes with both the type we
    // want to convert and structural information in the same type.
    llvm::SmallVector<NamedAttribute> newAttrs;
    if (op->hasTrait<OpTrait::ConstantLike>() ||
        isa<IREE::Util::GlobalOpInterface>(op)) {
      for (auto attr : op->getAttrs()) {
        auto oldAttr = attr.getValue();
        Attribute newAttr = oldAttr;
        if (auto floatAttr = dyn_cast<FloatAttr>(oldAttr)) {
          APInt apint = floatAttr.getValue().bitcastToAPInt();
          newAttr = rewriter.getI16IntegerAttr(apint.getZExtValue());
        } else if (auto denseAttr = dyn_cast<DenseFPElementsAttr>(oldAttr)) {
          newAttr =
              denseAttr.mapValues(rewriter.getI16Type(), [&](APFloat src) {
                return src.bitcastToAPInt();
              });
        }

        newAttrs.push_back(NamedAttribute(attr.getName(), newAttr));
      }
    } else {
      newAttrs.append(op->getAttrs().begin(), op->getAttrs().end());
    }

    llvm::SmallVector<Type> newResults;
    (void)getTypeConverter()->convertTypes(op->getResultTypes(), newResults);

    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         newResults, newAttrs, op->getSuccessors());

    for (Region &r : op->getRegions()) {
      Region *newRegion = state.addRegion();
      rewriter.inlineRegionBefore(r, *newRegion, newRegion->begin());
      TypeConverter::SignatureConversion result(newRegion->getNumArguments());

      if (failed(getTypeConverter()->convertSignatureArgs(
              newRegion->getArgumentTypes(), result))) {
        return rewriter.notifyMatchFailure(op,
                                           "argument type conversion failed");
      }

      rewriter.applySignatureConversion(newRegion, result);
    }

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct ConvertMemRefLoad final : OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newResTy = getTypeConverter()->convertType(op.getType());
    if (!newResTy)
      return rewriter.notifyMatchFailure(
          op->getLoc(), llvm::formatv("failed to convert memref type: {0}",
                                      op.getMemRefType()));

    rewriter.replaceOpWithNewOp<memref::LoadOp>(
        op, newResTy, adaptor.getMemref(), adaptor.getIndices(),
        op.getNontemporal());
    return success();
  }
};

struct ConvertMemRefStore final : OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getMemRefType());
    if (!newTy)
      return rewriter.notifyMatchFailure(
          op->getLoc(), llvm::formatv("failed to convert memref type: {0}",
                                      op.getMemRefType()));

    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, adaptor.getValue(), adaptor.getMemref(), adaptor.getIndices(),
        op.getNontemporal());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

std::optional<Value> materializeArithBitcast(OpBuilder &builder, Type resultTy,
                                             mlir::ValueRange inputs,
                                             mlir::Location loc) {
  return builder.create<arith::BitcastOp>(loc, resultTy, inputs);
}

static void populateIreeBf16EmulationPatterns(RewritePatternSet &patterns,
                                              TypeConverter &typeConverter) {
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
  patterns.add<GenericTypeConversionPattern, ConvertHalInterfaceBindingSubspan,
               ConvertMemRefAlloc, ConvertMemRefLoad, ConvertMemRefStore>(
      typeConverter, patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

struct ConvertBf16ToUInt16BuffersPass final
    : public ConvertBf16ToUInt16BuffersBase<ConvertBf16ToUInt16BuffersPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    ModuleOp op = getOperation();
    MLIRContext *ctx = &getContext();

    Bf16EmulationConverter typeConverter;
    typeConverter.addArgumentMaterialization(materializeArithBitcast);
    typeConverter.addTargetMaterialization(materializeArithBitcast);
    typeConverter.addSourceMaterialization(materializeArithBitcast);

    // Run the main emulation pass.
    {
      ConversionTarget target(*ctx);
      target.addDynamicallyLegalOp<func::FuncOp>([&typeConverter](
                                                     Operation *op) {
        return typeConverter.isLegal(cast<func::FuncOp>(op).getFunctionType());
      });
      target.addDynamicallyLegalDialect<arith::ArithDialect, func::FuncDialect,
                                        IREE::HAL::HALDialect,
                                        memref::MemRefDialect, scf::SCFDialect>(
          [&typeConverter](Operation *op) {
            bool legal = typeConverter.isLegal(op);
            LLVM_DEBUG(if (!legal) llvm::dbgs()
                       << "Bf16Emulation: illegal op: " << *op << "\n");
            return legal;
          });

      // Support the list of all vector operations that do not perform numerical
      // changes:
      target.addDynamicallyLegalOp<
          vector::BroadcastOp, vector::ShuffleOp, vector::ExtractElementOp,
          vector::ExtractOp, vector::InsertElementOp, vector::InsertOp,
          vector::ScalableInsertOp, vector::ScalableExtractOp,
          vector::InsertStridedSliceOp, vector::ReshapeOp,
          vector::ExtractStridedSliceOp, vector::TransferReadOp,
          vector::TransferWriteOp, vector::LoadOp, vector::StoreOp,
          vector::MaskedLoadOp, vector::MaskedStoreOp, vector::GatherOp,
          vector::ScatterOp, vector::ExpandLoadOp, vector::CompressStoreOp,
          vector::ShapeCastOp, vector::ConstantMaskOp, vector::CreateMaskOp,
          vector::MaskOp, vector::TransposeOp, vector::FlatTransposeOp,
          vector::SplatOp, vector::YieldOp>([&typeConverter](Operation *op) {
        bool legal = typeConverter.isLegal(op);
        LLVM_DEBUG(if (!legal) llvm::dbgs()
                   << "Bf16Emulation: illegal op: " << *op << "\n");
        return legal;
      });

      RewritePatternSet patterns(ctx);
      arith::populateExpandBFloat16Patterns(patterns);
      populateIreeBf16EmulationPatterns(patterns, typeConverter);

      if (failed(applyPartialConversion(op, target, std::move(patterns))))
        signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public interface
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>>
createConvertBf16ToUInt16BuffersPass() {
  return std::make_unique<ConvertBf16ToUInt16BuffersPass>();
}

} // namespace mlir::iree_compiler
