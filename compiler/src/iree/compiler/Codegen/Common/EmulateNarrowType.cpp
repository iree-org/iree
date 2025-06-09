// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_EMULATENARROWTYPEPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

struct ConvertHalInterfaceBindingSubspan final
    : OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto currentType = dyn_cast<MemRefType>(op.getType());
    if (!currentType) {
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "unhandled non-memref types");
    }
    auto newResultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(currentType));
    if (!newResultType) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to legalize memref type: {}", op.getType()));
    }
    Location loc = op.getLoc();
    OpFoldResult zero = rewriter.getIndexAttr(0);
    SmallVector<OpFoldResult> indices(currentType.getRank(), zero);

    // Get linearized type.
    int srcBits = currentType.getElementType().getIntOrFloatBitWidth();
    int dstBits = newResultType.getElementType().getIntOrFloatBitWidth();
    OpFoldResult elementOffset;
    Value byteOffset = adaptor.getByteOffset();
    if (byteOffset && !matchPattern(byteOffset, m_Zero())) {
      elementOffset = convertByteOffsetToElementOffset(
          rewriter, loc, byteOffset, currentType.getElementType());
    } else {
      elementOffset = rewriter.getIndexAttr(0);
    }
    SmallVector<OpFoldResult> sizes = getMixedValues(
        currentType.getShape(), adaptor.getDynamicDims(), rewriter);
    memref::LinearizedMemRefInfo linearizedMemRefInfo =
        memref::getLinearizedMemRefOffsetAndSize(rewriter, loc, srcBits,
                                                 dstBits, elementOffset, sizes);

    SmallVector<Value> dynamicLinearizedSize;
    if (newResultType.getRank() > 0 && !newResultType.hasStaticShape()) {
      dynamicLinearizedSize.push_back(getValueOrCreateConstantIndexOp(
          rewriter, loc, linearizedMemRefInfo.linearizedSize));
    }

    rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceBindingSubspanOp>(
        op, newResultType, adaptor.getLayout(), adaptor.getBinding(),
        byteOffset, dynamicLinearizedSize, adaptor.getAlignmentAttr(),
        adaptor.getDescriptorFlagsAttr());
    return success();
  }
};

static void populateIreeNarrowTypeEmulationPatterns(
    arith::NarrowTypeEmulationConverter &converter,
    RewritePatternSet &patterns) {
  patterns.add<ConvertHalInterfaceBindingSubspan>(converter,
                                                  patterns.getContext());
}

//===----------------------------------------------------------------------===//
// IREEConvertVectorStore
//===----------------------------------------------------------------------===//

// TODO(#20645): Delete the IREEConvertStoreVector pattern and switch to
// upstream patterns, after we have better control in upstream. Ideally, we can
// add an option to upstream patterns, which assumes that it is always an
// aligned store.
using VectorValue = TypedValue<VectorType>;
using MemRefValue = TypedValue<MemRefType>;

/// Extracts 1-D subvector from a 1-D vector.
///
/// Given the input rank-1 source vector, extracts `numElemsToExtract` elements
/// from `src`, starting at `offset`. The result is also a rank-1 vector:
///
///   vector<numElemsToExtract x !elemType>
///
/// (`!elType` is the element type of the source vector). As `offset` is a known
/// _static_ value, this helper hook emits `vector.extract_strided_slice`.
///
/// EXAMPLE:
///     %res = vector.extract_strided_slice %src
///       { offsets = [offset], sizes = [numElemsToExtract], strides = [1] }
static Value staticallyExtractSubvector(OpBuilder &rewriter, Location loc,
                                        Value src, int64_t offset,
                                        int64_t numElemsToExtract) {
  auto vectorType = cast<VectorType>(src.getType());
  assert(vectorType.getRank() == 1 && "expected source to be rank-1-D vector ");
  assert(offset + numElemsToExtract <= vectorType.getNumElements() &&
         "subvector out of bounds");

  // When extracting all available elements, just use the source vector as the
  // result.
  if (vectorType.getNumElements() == numElemsToExtract)
    return src;

  auto offsets = rewriter.getI64ArrayAttr({offset});
  auto sizes = rewriter.getI64ArrayAttr({numElemsToExtract});
  auto strides = rewriter.getI64ArrayAttr({1});

  auto resultVectorType =
      VectorType::get({numElemsToExtract}, vectorType.getElementType());
  return rewriter
      .create<vector::ExtractStridedSliceOp>(loc, resultVectorType, src,
                                             offsets, sizes, strides)
      ->getResult(0);
}

/// Inserts 1-D subvector into a 1-D vector.
///
/// Inserts the input rank-1 source vector into the destination vector starting
/// at `offset`. As `offset` is a known _static_ value, this helper hook emits
/// `vector.insert_strided_slice`.
///
/// EXAMPLE:
///   %res = vector.insert_strided_slice %src, %dest
///     {offsets = [%offset], strides [1]}
static Value staticallyInsertSubvector(OpBuilder &rewriter, Location loc,
                                       Value src, Value dest, int64_t offset) {
  [[maybe_unused]] auto srcVecTy = cast<VectorType>(src.getType());
  [[maybe_unused]] auto destVecTy = cast<VectorType>(dest.getType());
  assert(srcVecTy.getRank() == 1 && destVecTy.getRank() == 1 &&
         "expected source and dest to be rank-1 vector types");

  // If overwritting the destination vector, just return the source.
  if (srcVecTy.getNumElements() == destVecTy.getNumElements() && offset == 0)
    return src;

  auto offsets = rewriter.getI64ArrayAttr({offset});
  auto strides = rewriter.getI64ArrayAttr({1});
  return rewriter.create<vector::InsertStridedSliceOp>(loc, destVecTy, src,
                                                       dest, offsets, strides);
}

/// Extract `sliceNumElements` from source `vector` at `extractOffset`,
/// and insert it into an empty vector at `insertOffset`.
/// Inputs:
///   vec_in  = |0|1|2|3| : vector<4xi2>
///   extractOffset = 1
///   sliceNumElements = 2
///   insertOffset = 2
/// Output:
///   vec_out = |0|0|1|2| : vector<4xi2>
static Value extractSliceIntoByte(ConversionPatternRewriter &rewriter,
                                  Location loc, VectorValue vector,
                                  int64_t extractOffset,
                                  int64_t sliceNumElements,
                                  int64_t insertOffset) {
  assert(vector.getType().getRank() == 1 && "expected 1-D vector");
  auto vectorElementType = vector.getType().getElementType();
  // TODO: update and use `alignedConversionPrecondition` in the place of
  // these asserts.
  assert(
      sliceNumElements * vectorElementType.getIntOrFloatBitWidth() <= 8 &&
      "sliceNumElements * vector element size must be less than or equal to 8");
  assert(8 % vectorElementType.getIntOrFloatBitWidth() == 0 &&
         "vector element must be a valid sub-byte type");
  auto emulatedPerContainerElem = 8 / vectorElementType.getIntOrFloatBitWidth();
  auto emptyByteVector = rewriter.create<arith::ConstantOp>(
      loc, VectorType::get({emulatedPerContainerElem}, vectorElementType),
      rewriter.getZeroAttr(
          VectorType::get({emulatedPerContainerElem}, vectorElementType)));
  auto extracted = staticallyExtractSubvector(rewriter, loc, vector,
                                              extractOffset, sliceNumElements);
  return staticallyInsertSubvector(rewriter, loc, extracted, emptyByteVector,
                                   insertOffset);
}

/// Downcast two values to `downcastType`, then select values
/// based on `mask`, and casts the result to `upcastType`.
static Value downcastSelectAndUpcast(OpBuilder &builder, Location loc,
                                     VectorType downcastType,
                                     VectorType upcastType, Value mask,
                                     Value trueValue, Value falseValue) {
  assert(
      downcastType.getNumElements() * downcastType.getElementTypeBitWidth() ==
          upcastType.getNumElements() * upcastType.getElementTypeBitWidth() &&
      "expected input and output number of bits to match");
  if (trueValue.getType() != downcastType) {
    trueValue = builder.create<vector::BitCastOp>(loc, downcastType, trueValue);
  }
  if (falseValue.getType() != downcastType) {
    falseValue =
        builder.create<vector::BitCastOp>(loc, downcastType, falseValue);
  }
  Value selectedType =
      builder.create<arith::SelectOp>(loc, mask, trueValue, falseValue);
  // Upcast the selected value to the new type.
  return builder.create<vector::BitCastOp>(loc, upcastType, selectedType);
}

/// Emits `memref.generic_atomic_rmw` op to store a subbyte-sized value to a
/// byte in `linearizedMemref`, with a mask. The `valueToStore` is a vector of
/// subbyte-sized elements, with size of 8 bits, and the mask is used to select
/// which elements to store.
///
/// Inputs:
///   linearizedMemref = |2|2|2|2| : <4xi2> (<1xi8>)
///   storeIdx = 2
///   valueToStore = |3|3|3|3| : vector<4xi2>
///   mask = |0|0|1|1| : vector<4xi1>
///
/// Result:
///   linearizedMemref = |2|2|3|3| : <4xi2> (<1xi8>)
static void atomicRMW(OpBuilder &builder, Location loc,
                      MemRefValue linearizedMemref, Value storeIdx,
                      VectorValue valueToStore, Value mask) {
  assert(valueToStore.getType().getRank() == 1 && "expected 1-D vector");

  // Create an atomic load-modify-write region using
  // `memref.generic_atomic_rmw`.
  auto atomicOp = builder.create<memref::GenericAtomicRMWOp>(
      loc, linearizedMemref, ValueRange{storeIdx});
  Value origValue = atomicOp.getCurrentValue();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(atomicOp.getBody());

  // Load the original value from memory, and cast it to the original element
  // type.
  auto oneElemVecType = VectorType::get({1}, origValue.getType());
  Value origVecValue = builder.create<vector::FromElementsOp>(
      loc, oneElemVecType, ValueRange{origValue});

  // Construct the final masked value and yield it.
  Value maskedValue =
      downcastSelectAndUpcast(builder, loc, valueToStore.getType(),
                              oneElemVecType, mask, valueToStore, origVecValue);
  auto scalarMaskedValue =
      builder.create<vector::ExtractOp>(loc, maskedValue, 0);
  builder.create<memref::AtomicYieldOp>(loc, scalarMaskedValue);
}

/// Generate a non-atomic read-modify-write sequence for storing to the emulated
/// type. It has similar logic to `atomicRMWStore`, but without atomicity.
static void nonAtomicRMW(OpBuilder &builder, Location loc,
                         MemRefValue linearizedMemref, Value linearizedIndex,
                         VectorValue valueToStore, Value mask) {
  assert(valueToStore.getType().getRank() == 1 && "expected 1-D vector");

  auto oneElemVecType =
      VectorType::get({1}, linearizedMemref.getType().getElementType());
  Value origVecValue = builder.create<vector::LoadOp>(
      loc, oneElemVecType, linearizedMemref, ValueRange{linearizedIndex});
  origVecValue = builder.create<vector::BitCastOp>(loc, valueToStore.getType(),
                                                   origVecValue);

  Value maskedValue =
      downcastSelectAndUpcast(builder, loc, valueToStore.getType(),
                              oneElemVecType, mask, valueToStore, origVecValue);
  builder.create<vector::StoreOp>(loc, maskedValue, linearizedMemref,
                                  linearizedIndex);
}

// Emulate `vector.store` using a multi-byte container type.
//
// The container type is obtained through Op adaptor and would normally be
// generated via `NarrowTypeEmulationConverter`.
//
// EXAMPLE 1
// (aligned store of i4, emulated using i8 as the container type)
//
//      vector.store %src, %dest[%idx_1, %idx_2] : memref<4x8xi4>, vector<8xi4>
//
// is rewritten as:
//
//      %src_bitcast = vector.bitcast %src : vector<8xi4> to vector<4xi8>
//      vector.store %src_bitcast, %dest_bitcast[%idx]
//        : memref<16xi8>, vector<4xi8>
//
// EXAMPLE 2
// (unaligned store of i2, emulated using i8 as the container type)
//
//    vector.store %src, %dest[%c2, %c0] :memref<3x3xi2>, vector<3xi2>
//
// The i2 store is emulated through 2 x RMW sequences. The destination i2 memref
// is modelled using 3 bytes:
//
//    Byte 0     Byte 1     Byte 2
// +----------+----------+----------+
// | oooooooo | ooooNNNN | NNoooooo |
// +----------+----------+----------+
//
// N - (N)ew entries (i.e. to be overwritten by vector.store)
// o - (o)ld entries (to be preserved)
//
// For the generated output in the non-atomic case, see:
//  * @vector_store_i2_const_index_two_partial_stores`
// in:
//  * "vector-emulate-narrow-type-unaligned-non-atomic.mlir".
//
// NOTE: By default, all RMW sequences are atomic. Set `disableAtomicRMW` to
// `false` to generate non-atomic RMW sequences.
struct IREEConvertVectorStore final : OpConversionPattern<vector::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  IREEConvertVectorStore(MLIRContext *context, bool disableAtomicRMW,
                         PatternBenefit benefit)
      : OpConversionPattern<vector::StoreOp>(context, benefit),
        disableAtomicRMW(disableAtomicRMW) {}

  LogicalResult
  matchAndRewrite(vector::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // See #115653
    if (op.getValueToStore().getType().getRank() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "only 1-D vectors are supported ATM");

    auto loc = op.getLoc();

    auto valueToStore = cast<VectorValue>(op.getValueToStore());
    auto containerElemTy =
        cast<MemRefType>(adaptor.getBase().getType()).getElementType();
    Type emulatedElemTy = op.getValueToStore().getType().getElementType();
    int emulatedBits = emulatedElemTy.getIntOrFloatBitWidth();
    int containerBits = containerElemTy.getIntOrFloatBitWidth();

    // Check per-element alignment.
    if (containerBits % emulatedBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "impossible to pack emulated elements into container elements "
              "(bit-wise misalignment)");
    }
    int emulatedPerContainerElem = containerBits / emulatedBits;

    // Adjust the number of elements to store when emulating narrow types.
    // Here only the 1-D vector store is considered, and the N-D memref types
    // should be linearized.
    // For example, to emulate i4 to i8, the following op:
    //
    // vector.store %arg1, %0[%arg2, %arg3] : memref<4x8xi4>, vector<8xi4>
    //
    // can be replaced with
    //
    // %bitcast = vector.bitcast %arg1 : vector<8xi4> to vector<4xi8>
    // vector.store %bitcast, %alloc[%linear_index] : memref<16xi8>,
    // vector<4xi8>

    auto origElements = valueToStore.getType().getNumElements();
    // Note, per-element-alignment was already verified above.
    bool isDivisibleInSize = origElements % emulatedPerContainerElem == 0;

    auto stridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(loc, op.getBase());

    OpFoldResult linearizedIndices;
    memref::LinearizedMemRefInfo linearizedInfo;
    std::tie(linearizedInfo, linearizedIndices) =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, emulatedBits, containerBits,
            stridedMetadata.getConstifiedMixedOffset(),
            stridedMetadata.getConstifiedMixedSizes(),
            stridedMetadata.getConstifiedMixedStrides(),
            getAsOpFoldResult(adaptor.getIndices()));

    std::optional<int64_t> foldedNumFrontPadElems =
        isDivisibleInSize ? 0
                          : getConstantIntValue(linearizedInfo.intraDataOffset);

    if (!foldedNumFrontPadElems) {
      return rewriter.notifyMatchFailure(
          op, "subbyte store emulation: dynamic front padding size is "
              "not yet implemented");
    }

    auto memrefBase = cast<MemRefValue>(adaptor.getBase());

    // Conditions when atomic RMWs are not needed:
    // 1. The source vector size (in bits) is a multiple of byte size.
    // 2. The address of the store is aligned to the emulated width boundary.
    //
    // For example, to store a vector<4xi2> to <13xi2> at offset 4, does not
    // need unaligned emulation because the store address is aligned and the
    // source is a whole byte.
    bool emulationRequiresPartialStores =
        !isDivisibleInSize || *foldedNumFrontPadElems != 0;
    if (!emulationRequiresPartialStores) {
      // Basic case: storing full bytes.
      auto numElements = origElements / emulatedPerContainerElem;
      auto bitCast = rewriter.create<vector::BitCastOp>(
          loc, VectorType::get(numElements, containerElemTy),
          op.getValueToStore());
      rewriter.replaceOpWithNewOp<vector::StoreOp>(
          op, bitCast.getResult(), memrefBase,
          getValueOrCreateConstantIndexOp(rewriter, loc, linearizedIndices));
      return success();
    }

    // Next, handle the case when sub-byte read-modify-write
    // sequences are needed to emulate a vector store.
    // Here is an example:
    //
    // Vector to store: vector<7xi2>
    // Value to store: 11 11 11 11 11 11 11 (all ones)
    //
    // Destination: memref<12xi2>
    // Store offset: 2 (i.e. 4 bits into the 1st emulated byte).
    //
    // Input MLIR: vector.store %val, %dest[%c2] : memref<12xi2>, vector<7xi2>
    //
    // Destination memref before:
    //
    //    Byte 0     Byte 1     Byte 2
    // +----------+----------+----------+
    // | 00000000 | 00000000 | 00000000 |
    // +----------+----------+----------+
    //
    // Destination memref after:
    //
    //    Byte 0     Byte 1     Byte 2
    // +----------+----------+----------+
    // | 00001111 | 11111111 | 11000000 |
    // +----------+----------+----------+
    //
    // Note, stores to Byte 1 are "full-width" and hence don't require RMW (no
    // need for atomicity). Stores to Bytes 0 and Byte 2 are "partial", hence
    // requiring RMW access (atomicity is required).

    // The index into the target memref we are storing to.
    Value currentDestIndex =
        getValueOrCreateConstantIndexOp(rewriter, loc, linearizedIndices);
    // The index into the source vector we are currently processing.
    auto currentSourceIndex = 0;

    // Build a mask used for rmw.
    auto subWidthStoreMaskType =
        VectorType::get({emulatedPerContainerElem}, rewriter.getI1Type());

    auto storeFunc = disableAtomicRMW ? nonAtomicRMW : atomicRMW;

    // 1. Partial width store for the leading byte.
    // When the store address is not aligned to emulated width boundary, deal
    // with the unaligned part so that the rest elements are aligned to width
    // boundary.
    auto frontSubWidthStoreElem =
        (emulatedPerContainerElem - *foldedNumFrontPadElems) %
        emulatedPerContainerElem;
    if (frontSubWidthStoreElem > 0) {
      SmallVector<bool> frontMaskValues(emulatedPerContainerElem, false);
      if (*foldedNumFrontPadElems + origElements < emulatedPerContainerElem) {
        std::fill_n(frontMaskValues.begin() + *foldedNumFrontPadElems,
                    origElements, true);
        frontSubWidthStoreElem = origElements;
      } else {
        std::fill_n(frontMaskValues.end() - frontSubWidthStoreElem,
                    *foldedNumFrontPadElems, true);
      }
      auto frontMask = rewriter.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(subWidthStoreMaskType, frontMaskValues));

      currentSourceIndex = emulatedPerContainerElem - (*foldedNumFrontPadElems);
      auto value =
          extractSliceIntoByte(rewriter, loc, valueToStore, 0,
                               frontSubWidthStoreElem, *foldedNumFrontPadElems);

      storeFunc(rewriter, loc, memrefBase, currentDestIndex,
                cast<VectorValue>(value), frontMask.getResult());
    }

    if (currentSourceIndex >= origElements) {
      rewriter.eraseOp(op);
      return success();
    }

    // Increment the destination index by 1 to align to the emulated width
    // boundary.
    auto constantOne = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    currentDestIndex = rewriter.create<arith::AddIOp>(
        loc, rewriter.getIndexType(), currentDestIndex, constantOne);

    // 2. Full width store for the inner output bytes.
    // After the previous step, the store address is aligned to the emulated
    // width boundary.
    int64_t fullWidthStoreSize =
        (origElements - currentSourceIndex) / emulatedPerContainerElem;
    int64_t numNonFullWidthElements =
        fullWidthStoreSize * emulatedPerContainerElem;
    if (fullWidthStoreSize > 0) {
      auto fullWidthStorePart = staticallyExtractSubvector(
          rewriter, loc, valueToStore, currentSourceIndex,
          numNonFullWidthElements);

      auto originType = cast<VectorType>(fullWidthStorePart.getType());
      auto memrefElemType = getElementTypeOrSelf(memrefBase.getType());
      auto storeType = VectorType::get(
          {originType.getNumElements() / emulatedPerContainerElem},
          memrefElemType);
      auto bitCast = rewriter.create<vector::BitCastOp>(loc, storeType,
                                                        fullWidthStorePart);
      rewriter.create<vector::StoreOp>(loc, bitCast.getResult(), memrefBase,
                                       currentDestIndex);

      currentSourceIndex += numNonFullWidthElements;
      currentDestIndex = rewriter.create<arith::AddIOp>(
          loc, rewriter.getIndexType(), currentDestIndex,
          rewriter.create<arith::ConstantIndexOp>(loc, fullWidthStoreSize));
    }

    // 3. Partial width store for the trailing output byte.
    // It is needed when the residual length is smaller than the emulated width,
    // which is not covered in step 2 above.
    auto remainingElements = origElements - currentSourceIndex;
    if (remainingElements != 0) {
      auto subWidthStorePart =
          extractSliceIntoByte(rewriter, loc, cast<VectorValue>(valueToStore),
                               currentSourceIndex, remainingElements, 0);

      // Generate back mask.
      auto maskValues = SmallVector<bool>(emulatedPerContainerElem, 0);
      std::fill_n(maskValues.begin(), remainingElements, 1);
      auto backMask = rewriter.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(subWidthStoreMaskType, maskValues));

      storeFunc(rewriter, loc, memrefBase, currentDestIndex,
                cast<VectorValue>(subWidthStorePart), backMask.getResult());
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const bool disableAtomicRMW;
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct EmulateNarrowTypePass final
    : impl::EmulateNarrowTypePassBase<EmulateNarrowTypePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, vector::VectorDialect,
                    affine::AffineDialect, IREE::HAL::HALDialect>();
  }

  void runOnOperation() override {
    // The number of bits used in a load/store op.
    constexpr unsigned kLoadStoreEmulateBitwidth = 8;
    static_assert(
        llvm::isPowerOf2_32(kLoadStoreEmulateBitwidth) &&
        "only power of 2 is supported for narrow type load/store emulation");

    MLIRContext *ctx = &getContext();

    arith::NarrowTypeEmulationConverter typeConverter(
        kLoadStoreEmulateBitwidth);
    memref::populateMemRefNarrowTypeEmulationConversions(typeConverter);

    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<func::FuncOp>([&typeConverter](Operation *op) {
      return typeConverter.isLegal(cast<func::FuncOp>(op).getFunctionType());
    });
    auto opLegalCallback = [&typeConverter](Operation *op) {
      return typeConverter.isLegal(op);
    };
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(opLegalCallback);
    target.addDynamicallyLegalDialect<
        arith::ArithDialect, vector::VectorDialect, memref::MemRefDialect,
        affine::AffineDialect, IREE::HAL::HALDialect>(opLegalCallback);

    RewritePatternSet patterns(ctx);
    patterns.insert<IREEConvertVectorStore>(ctx, /*disableAtomicRMW=*/false,
                                            /*benefit=*/100);
    arith::populateArithNarrowTypeEmulationPatterns(typeConverter, patterns);
    memref::populateMemRefNarrowTypeEmulationPatterns(typeConverter, patterns);
    populateIREEResolveExtractStridedMetadataPatterns(patterns);
    vector::populateVectorNarrowTypeEmulationPatterns(typeConverter, patterns);
    populateIreeNarrowTypeEmulationPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      getOperation()->emitOpError("failed to emulate bit width");
      return signalPassFailure();
    }

    RewritePatternSet sinkBroadcast(ctx);
    vector::populateSinkVectorOpsPatterns(sinkBroadcast);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(sinkBroadcast)))) {
      getOperation()->emitOpError("failed in sinking of broadcasts");
      return signalPassFailure();
    }

    // Also do the `bitcast -> extui/extsi` rewrite.
    RewritePatternSet foldExtPatterns(ctx);
    vector::populateVectorNarrowTypeRewritePatterns(foldExtPatterns);
    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(foldExtPatterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
