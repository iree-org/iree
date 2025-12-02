// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/ConversionDialectInterface.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-pcf-resolve-tokens"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_RESOLVETOKENSPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

class LoadDependentDialectExtension : public DialectExtensionBase {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoadDependentDialectExtension)

  LoadDependentDialectExtension() : DialectExtensionBase(/*dialectNames=*/{}) {}

  void apply(MLIRContext *context,
             MutableArrayRef<Dialect *> dialects) const final {
    for (Dialect *dialect : dialects) {
      auto *iface = dyn_cast<PCFConversionDialectInterface>(dialect);
      if (!iface) {
        continue;
      }
      iface->loadTokenLoweringDependentDialects(context);
    }
  }

  /// Return a copy of this extension.
  std::unique_ptr<DialectExtensionBase> clone() const final {
    return std::make_unique<LoadDependentDialectExtension>(*this);
  }
};

struct ResolveTokensPass final
    : impl::ResolveTokensPassBase<ResolveTokensPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.addExtensions<LoadDependentDialectExtension>();
  }
  void runOnOperation() override;
};

/// Flatten the given value ranges into a single vector of values.
static SmallVector<Value> flattenValues(ArrayRef<ValueRange> values) {
  SmallVector<Value> result;
  for (const ValueRange &vals : values)
    llvm::append_range(result, vals);
  return result;
}

/// Helper function for converting branch ops. This function converts the
/// signature of the given block. If the new block signature is different from
/// `expectedTypes`, returns "failure".
static FailureOr<Block *> getConvertedBlock(ConversionPatternRewriter &rewriter,
                                            const TypeConverter *converter,
                                            Operation *branchOp, Block *block,
                                            TypeRange expectedTypes) {
  assert(converter && "expected non-null type converter");
  assert(!block->isEntryBlock() && "entry blocks have no predecessors");

  // There is nothing to do if the types already match.
  if (block->getArgumentTypes() == expectedTypes)
    return block;

  // Compute the new block argument types and convert the block.
  std::optional<TypeConverter::SignatureConversion> conversion =
      converter->convertBlockSignature(block);
  if (!conversion)
    return rewriter.notifyMatchFailure(branchOp,
                                       "could not compute block signature");
  if (expectedTypes != conversion->getConvertedTypes())
    return rewriter.notifyMatchFailure(
        branchOp,
        "mismatch between adaptor operand types and computed block signature");
  return rewriter.applySignatureConversion(block, *conversion, converter);
}

struct ConvertGenericOp : public OpConversionPattern<PCF::GenericOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(PCF::GenericOp genericOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool madeChange = false;
    if (llvm::any_of(genericOp.getRegionRefArgs(), [](BlockArgument b) {
          return cast<PCF::ShapedRefType>(b.getType()).isParentScopeOnlySync();
        })) {
      genericOp.setSyncOnReturn(true);
      madeChange = true;
    }

    SmallVector<Block *> blocksToConvert = llvm::map_to_vector(
        genericOp.getRegion().getBlocks(), [](Block &b) { return &b; });
    for (Block *block : blocksToConvert) {
      std::optional<TypeConverter::SignatureConversion> signatureConverter =
          getTypeConverter()->convertBlockSignature(block);
      if (signatureConverter) {
        madeChange = true;
        rewriter.applySignatureConversion(block, signatureConverter.value(),
                                          getTypeConverter());
      }
    }
    return success(madeChange);
  }
};

struct ConvertLoopOp : public OpConversionPattern<PCF::LoopOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(PCF::LoopOp loopOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool madeChange = false;
    if (llvm::any_of(loopOp.getRegionRefArgs(), [](BlockArgument b) {
          return cast<PCF::ShapedRefType>(b.getType()).isParentScopeOnlySync();
        })) {
      loopOp.setSyncOnReturn(true);
      madeChange = true;
    }

    std::optional<TypeConverter::SignatureConversion> signatureConverter =
        getTypeConverter()->convertBlockSignature(loopOp.getBody());
    if (signatureConverter) {
      madeChange = true;
      rewriter.applySignatureConversion(
          loopOp.getBody(), signatureConverter.value(), getTypeConverter());
    }
    return success(madeChange);
  }
};

struct ConvertAllocOp : public OpConversionPattern<PCF::AllocOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(PCF::AllocOp allocOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertType(allocOp.getResultType(),
                                               resultTypes))) {
      return failure();
    }

    auto syncScope = cast_if_present<PCF::SyncScopeAttr>(
        allocOp.getResultType().getSyncScope());
    SmallVector<Value> replacements;
    if (syncScope) {
      replacements = syncScope.allocate(rewriter);
    }
    auto newAlloc =
        PCF::AllocOp::create(rewriter, allocOp.getLoc(),
                             cast<PCF::ShapedRefType>(resultTypes.front()),
                             allocOp.getDynamicSizes());
    replacements.insert(replacements.begin(), newAlloc.getResult());
    rewriter.replaceOp(allocOp, replacements);
    return success();
  }
};

struct ConvertWriteSliceOp : public OpConversionPattern<PCF::WriteSliceOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(PCF::WriteSliceOp writeOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the destination with the unscoped sref.
    ValueRange splitDest = adaptor.getDest();

    rewriter.startOpModification(writeOp);
    writeOp.getDestMutable().assign(splitDest.front());

    // Enqueue the write via the attribute interface immediately after it.
    auto syncScope = cast_if_present<PCF::SyncScopeAttr>(
        writeOp.getDestType().getSyncScope());
    if (syncScope) {
      rewriter.setInsertionPointAfter(writeOp);
      syncScope.enqueueWrite(rewriter, splitDest.drop_front(), writeOp);
    }
    rewriter.finalizeOpModification(writeOp);
    return success();
  }
};

/// Convert the destination block signature if necessary.
struct ConvertBranchOp : public OpConversionPattern<cf::BranchOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *typeConverter = getTypeConverter();
    if (llvm::all_of(op->getOperandTypes(),
                     [&](Type t) { return typeConverter->isLegal(t); })) {
      // Nothing to do.
      return failure();
    }
    SmallVector<Value> flattenedAdaptor = flattenValues(adaptor.getOperands());
    FailureOr<Block *> convertedBlock =
        getConvertedBlock(rewriter, typeConverter, op, op.getSuccessor(),
                          TypeRange(ValueRange(flattenedAdaptor)));
    if (failed(convertedBlock))
      return failure();
    op.getDestOperandsMutable().assign(flattenedAdaptor);
    return success();
  }
};

struct ConvertOptimizationBarrier
    : public OpConversionPattern<Util::OptimizationBarrierOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(Util::OptimizationBarrierOp barrier, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<Util::OptimizationBarrierOp>(
        barrier, flattenValues(adaptor.getOperands()));
    return success();
  }
};

void ResolveTokensPass::runOnOperation() {
  auto *context = &getContext();

  TypeConverter typeConverter;
  ConversionTarget target(*context);
  RewritePatternSet patterns(&getContext());

  // Passthrough converter for everything else. Type conversions are iterated
  // in reverse, meaning this will be checked after subsequently added
  // specialized variants.
  typeConverter.addConversion(
      [](Type type, SmallVectorImpl<Type> &resultTypes) -> LogicalResult {
        resultTypes.push_back(type);
        return success();
      });

  // Expand shaped refs into one without sync scope and the scope's concrete
  // types.
  typeConverter.addConversion([=](PCF::ShapedRefType type,
                                  SmallVectorImpl<Type> &resultTypes)
                                  -> std::optional<LogicalResult> {
    auto syncScope = cast_if_present<PCF::SyncScopeAttr>(type.getSyncScope());
    if (!syncScope) {
      resultTypes.push_back(type);
      return success();
    }

    auto newRefType =
        PCF::ShapedRefType::get(type.getContext(), type.getShape(),
                                type.getElementType(), type.getScope());
    resultTypes.push_back(newRefType);
    for (Type expandedType : syncScope.getConcreteTypes(type.getContext())) {
      resultTypes.push_back(expandedType);
    }
    return success();
  });

  patterns
      .insert<ConvertGenericOp, ConvertLoopOp, ConvertAllocOp,
              ConvertWriteSliceOp, ConvertOptimizationBarrier, ConvertBranchOp>(
          typeConverter, context);

  // Verify that all operand, result, and region argument types have been
  // converted.
  auto isLegallyTypedOp = [&](Operation *op) -> bool {
    for (Type type : op->getResultTypes()) {
      if (!typeConverter.isLegal(type))
        return false;
    }
    for (Type type : op->getOperandTypes()) {
      if (!typeConverter.isLegal(type))
        return false;
    }
    for (auto &region : op->getRegions()) {
      for (auto type : region.getArgumentTypes()) {
        if (!typeConverter.isLegal(type))
          return false;
      }
    }
    if (auto funcInterface = dyn_cast<FunctionOpInterface>(op)) {
      if (llvm::any_of(funcInterface.getArgumentTypes(),
                       [&](Type t) { return !typeConverter.isLegal(t); })) {
        return false;
      }
      if (llvm::any_of(funcInterface.getResultTypes(),
                       [&](Type t) { return !typeConverter.isLegal(t); })) {
        return false;
      }
    }
    return true;
  };
  target.markUnknownOpDynamicallyLegal(isLegallyTypedOp);

  scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                       target);
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);

  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns),
                                 config))) {
    return signalPassFailure();
  }
}

} // namespace

} // namespace mlir::iree_compiler::IREE::PCF
