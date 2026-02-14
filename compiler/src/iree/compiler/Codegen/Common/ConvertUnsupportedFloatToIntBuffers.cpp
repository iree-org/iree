// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert unsupported float buffer types to
// integer types of the same bit width for backends that don't support them.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-codegen-convert-unsupported-float-to-int-buffers"

namespace mlir::iree_compiler {

#define GEN_PASS_DECL_CONVERTUNSUPPORTEDFLOATTOINTBUFFERSPASS
#define GEN_PASS_DEF_CONVERTUNSUPPORTEDFLOATTOINTBUFFERSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

using Options = ConvertUnsupportedFloatToIntBuffersPassOptions;

class UnsupportedFloatEmulationConverter : public TypeConverter {
public:
  explicit UnsupportedFloatEmulationConverter(const Options &options)
      : options(options) {
    // Allow unknown types.
    addConversion([](Type ty) -> std::optional<Type> { return ty; });

    // Scalar float case.
    addConversion([this](FloatType ty) -> std::optional<Type> {
      if (auto intTy = getIntTypeForFloat(ty)) {
        return intTy;
      }
      return ty;
    });

    addConversion([this](ShapedType ty) -> std::optional<Type> {
      return ty.clone(convertType(ty.getElementType()));
    });

    addConversion([this](FunctionType ty) -> std::optional<Type> {
      SmallVector<Type> inputs;
      if (failed(convertTypes(ty.getInputs(), inputs))) {
        return std::nullopt;
      }

      SmallVector<Type> results;
      if (failed(convertTypes(ty.getResults(), results))) {
        return std::nullopt;
      }

      return FunctionType::get(ty.getContext(), inputs, results);
    });
  }

  bool shouldConvertFloat(FloatType ty) const {
    return getIntTypeForFloat(ty) != nullptr;
  }

private:
  /// Returns the integer type to use for the given float type, or nullptr if
  /// the float type should not be converted.
  IntegerType getIntTypeForFloat(FloatType ty) const {
    if (options.includeBf16 && ty.isBF16()) {
      return IntegerType::get(ty.getContext(), 16);
    }
    IntegerType i8Type = IntegerType::get(ty.getContext(), 8);
    if (options.includeF8E5M2 && isa<Float8E5M2Type>(ty)) {
      return i8Type;
    }
    if (options.includeF8E4M3FN && isa<Float8E4M3FNType>(ty)) {
      return i8Type;
    }
    if (options.includeF8E5M2FNUZ && isa<Float8E5M2FNUZType>(ty)) {
      return i8Type;
    }
    if (options.includeF8E4M3FNUZ && isa<Float8E4M3FNUZType>(ty)) {
      return i8Type;
    }
    if (options.includeF8E8M0FNU && isa<Float8E8M0FNUType>(ty)) {
      return i8Type;
    }
    return nullptr;
  }
  ConvertUnsupportedFloatToIntBuffersPassOptions options;
};

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//
struct ConvertHalInterfaceBindingSubspan final
    : OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newResultTy = getTypeConverter()->convertType(op.getType());
    if (!newResultTy) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to legalize memref type: {}", op.getType()));
    }

    auto newOp =
        rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceBindingSubspanOp>(
            op, newResultTy, adaptor.getLayout(), adaptor.getBinding(),
            adaptor.getByteOffset(), adaptor.getDynamicDims(),
            adaptor.getAlignmentAttr(), adaptor.getDescriptorFlagsAttr());
    LLVM_DEBUG(llvm::dbgs()
               << "UnsupportedFloatEmulation: new op: " << newOp << "\n");
    (void)newOp;
    return success();
  }
};

struct ConvertMemRefAlloc final : OpConversionPattern<memref::AllocOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to convert memref type: {}", op.getType()));
    }

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
    auto *converter =
        static_cast<const UnsupportedFloatEmulationConverter *>(typeConverter);
    if (op->hasTrait<OpTrait::ConstantLike>() ||
        isa<IREE::Util::GlobalOpInterface>(op)) {
      for (auto attr : op->getAttrs()) {
        auto oldAttr = attr.getValue();
        Attribute newAttr = oldAttr;
        if (auto floatAttr = dyn_cast<FloatAttr>(oldAttr)) {
          auto floatTy = cast<FloatType>(floatAttr.getType());
          if (converter->shouldConvertFloat(floatTy)) {
            APInt apint = floatAttr.getValue().bitcastToAPInt();
            newAttr = rewriter.getIntegerAttr(
                rewriter.getIntegerType(apint.getBitWidth()), apint);
          }
        } else if (auto denseAttr = dyn_cast<DenseFPElementsAttr>(oldAttr)) {
          auto floatTy = cast<FloatType>(denseAttr.getType().getElementType());
          if (converter->shouldConvertFloat(floatTy)) {
            unsigned bitWidth = floatTy.getWidth();
            newAttr = denseAttr.mapValues(
                rewriter.getIntegerType(bitWidth),
                [&](const APFloat &src) { return src.bitcastToAPInt(); });
          }
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
    }
    Operation *newOp = rewriter.create(state);

    for (Region &newRegion : newOp->getRegions()) {
      TypeConverter::SignatureConversion result(newRegion.getNumArguments());

      if (failed(getTypeConverter()->convertSignatureArgs(
              newRegion.getArgumentTypes(), result))) {
        return rewriter.notifyMatchFailure(op,
                                           "argument type conversion failed");
      }

      rewriter.applySignatureConversion(&newRegion.front(), result,
                                        typeConverter);
    }

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct ConvertMemRefLoad final : OpConversionPattern<memref::LoadOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newResTy = getTypeConverter()->convertType(op.getType());
    if (!newResTy) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), llvm::formatv("failed to convert memref type: {}",
                                      op.getMemRefType()));
    }

    rewriter.replaceOpWithNewOp<memref::LoadOp>(
        op, newResTy, adaptor.getMemref(), adaptor.getIndices(),
        op.getNontemporal());
    return success();
  }
};

struct ConvertMemRefStore final : OpConversionPattern<memref::StoreOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getMemRefType());
    if (!newTy) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), llvm::formatv("failed to convert memref type: {}",
                                      op.getMemRefType()));
    }

    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, adaptor.getValue(), adaptor.getMemref(), adaptor.getIndices(),
        op.getNontemporal());
    return success();
  }
};

struct ConvertAmdgpuFatRawBufferCast final
    : OpConversionPattern<amdgpu::FatRawBufferCastOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(amdgpu::FatRawBufferCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to convert memref type: {}", op.getType()));
    }

    auto newOp = rewriter.replaceOpWithNewOp<amdgpu::FatRawBufferCastOp>(
        op, newTy, adaptor.getSource(), adaptor.getValidBytes(),
        adaptor.getCacheSwizzleStride(), adaptor.getBoundsCheck(),
        adaptor.getResetOffset());
    LLVM_DEBUG(llvm::dbgs()
               << "UnsupportedFloatEmulation: new op: " << newOp << "\n");
    (void)newOp;
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

Value materializeArithBitcast(OpBuilder &builder, Type resultTy,
                              mlir::ValueRange inputs, mlir::Location loc) {
  return arith::BitcastOp::create(builder, loc, resultTy, inputs);
}

static void
populateIreeUnsupportedFloatEmulationPatterns(RewritePatternSet &patterns,
                                              TypeConverter &typeConverter) {
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
  patterns.add<GenericTypeConversionPattern, ConvertHalInterfaceBindingSubspan,
               ConvertMemRefAlloc, ConvertMemRefLoad, ConvertMemRefStore,
               ConvertAmdgpuFatRawBufferCast>(typeConverter,
                                              patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

struct ConvertUnsupportedFloatToIntBuffersPass final
    : impl::ConvertUnsupportedFloatToIntBuffersPassBase<
          ConvertUnsupportedFloatToIntBuffersPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto op = getOperation();
    MLIRContext *ctx = &getContext();

    ConvertUnsupportedFloatToIntBuffersPassOptions opts;
    opts.includeBf16 = includeBf16;
    opts.includeF8E5M2 = includeF8E5M2;
    opts.includeF8E4M3FN = includeF8E4M3FN;
    opts.includeF8E5M2FNUZ = includeF8E5M2FNUZ;
    opts.includeF8E4M3FNUZ = includeF8E4M3FNUZ;
    opts.includeF8E8M0FNU = includeF8E8M0FNU;
    UnsupportedFloatEmulationConverter typeConverter(opts);
    typeConverter.addTargetMaterialization(materializeArithBitcast);
    typeConverter.addSourceMaterialization(materializeArithBitcast);

    // Run the main emulation pass.
    {
      ConversionTarget target(*ctx);
      target.addDynamicallyLegalOp<func::FuncOp>([&typeConverter](
                                                     Operation *op) {
        return typeConverter.isLegal(cast<func::FuncOp>(op).getFunctionType());
      });
      target.addLegalOp<arith::TruncFOp, arith::ExtFOp, ModuleOp>();
      target.addDynamicallyLegalDialect<
          arith::ArithDialect, func::FuncDialect, IREE::HAL::HALDialect,
          memref::MemRefDialect, scf::SCFDialect,
          IREE::Codegen::IREECodegenDialect>([&typeConverter](Operation *op) {
        bool legal = typeConverter.isLegal(op);
        LLVM_DEBUG(if (!legal) llvm::dbgs()
                   << "UnsupportedFloatEmulation: illegal op: " << *op << "\n");
        return legal;
      });

      // Support the list of all vector operations that do not perform numerical
      // changes. Also handle amdgpu buffer casts:
      target.addDynamicallyLegalOp<
          amdgpu::FatRawBufferCastOp, vector::BroadcastOp, vector::ShuffleOp,
          vector::ExtractOp, vector::InsertOp, vector::ScalableInsertOp,
          vector::ScalableExtractOp, vector::InsertStridedSliceOp,
          vector::ExtractStridedSliceOp, vector::TransferReadOp,
          vector::TransferWriteOp, vector::LoadOp, vector::StoreOp,
          vector::MaskedLoadOp, vector::MaskedStoreOp, vector::GatherOp,
          vector::ScatterOp, vector::ExpandLoadOp, vector::CompressStoreOp,
          vector::ShapeCastOp, vector::ConstantMaskOp, vector::CreateMaskOp,
          vector::MaskOp, vector::TransposeOp, vector::YieldOp,
          vector::FromElementsOp, vector::ToElementsOp>([&typeConverter](
                                                            Operation *op) {
        bool legal = typeConverter.isLegal(op);
        LLVM_DEBUG(if (!legal) llvm::dbgs()
                   << "UnsupportedFloatEmulation: illegal op: " << *op << "\n");
        return legal;
      });

      RewritePatternSet patterns(ctx);
      populateIreeUnsupportedFloatEmulationPatterns(patterns, typeConverter);

      if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
        signalPassFailure();
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
