// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Util/Analysis/Constant/ConstExpr.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/GlobalOptimization/QuantizationUtils.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "iree-global-opt-pack-sub-byte-quantized-constants"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_PACKSUBBYTEQUANTIZEDCONSTANTSPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

using IREE::LinalgExt::DequantizeAffineOp;
using IREE::LinalgExt::QuantizeAffineOp;

namespace {

/// Bit widths quantized values can be packed into, narrowest first. Each
/// divides 8, so a byte holds a whole number of values.
constexpr unsigned kPackedBitWidths[] = {2, 4};

/// Returns the attribute holding `value` when it is a constant or a load of an
/// immutable global, or null otherwise. For a global this is its initial
/// value, which may be a parameter reference rather than data.
static Attribute getConstantAttr(Value value,
                                 SymbolTableCollection &symbolTables) {
  Attribute attr;
  if (matchPattern(value, m_Constant(&attr))) {
    return attr;
  }
  auto loadOp = value.getDefiningOp<IREE::Util::GlobalLoadOpInterface>();
  if (!loadOp) {
    return {};
  }
  IREE::Util::GlobalOpInterface globalOp =
      IREE::Util::lookupGlobalOp(loadOp, loadOp.getGlobalAttr(), symbolTables);
  if (!globalOp || globalOp.isGlobalMutable()) {
    return {};
  }
  return globalOp.getGlobalInitialValue();
}

/// Returns the integers in `attr`, read as unsigned if `isUnsigned`, or nullopt
/// when `attr` holds no readable integers. A splat gives a single value.
static std::optional<SmallVector<llvm::APSInt>>
getIntegerValues(Attribute attr, bool isUnsigned) {
  using Values = std::optional<SmallVector<llvm::APSInt>>;
  auto toAPSInt = [&](const APInt &value) {
    return llvm::APSInt(value, isUnsigned);
  };
  return llvm::TypeSwitch<Attribute, Values>(attr)
      .Case([&](IntegerAttr intAttr) -> Values {
        return SmallVector<llvm::APSInt>{toAPSInt(intAttr.getValue())};
      })
      .Case([&](DenseIntElementsAttr denseAttr) -> Values {
        if (denseAttr.isSplat()) {
          return SmallVector<llvm::APSInt>{
              toAPSInt(denseAttr.getSplatValue<APInt>())};
        }
        return llvm::map_to_vector(denseAttr.getValues<APInt>(), toAPSInt);
      })
      .Case<DenseI8ResourceElementsAttr, DenseI16ResourceElementsAttr,
            DenseI32ResourceElementsAttr, DenseI64ResourceElementsAttr>(
          [&](auto resourceAttr) -> Values {
            unsigned bitWidth = resourceAttr.getType().getElementTypeBitWidth();
            auto values = resourceAttr.tryGetAsArrayRef();
            if (!values) {
              return std::nullopt;
            }
            return llvm::map_to_vector(*values, [&](auto value) {
              return toAPSInt(APInt(bitWidth, value, /*isSigned=*/true));
            });
          })
      .Default(std::nullopt);
}

/// Returns true when every value in `values` lies in `range`.
static bool allValuesInRange(ArrayRef<llvm::APSInt> values, QuantMinMax range) {
  llvm::APSInt min = llvm::APSInt::get(range.min);
  llvm::APSInt max = llvm::APSInt::get(range.max);
  return llvm::all_of(values, [&](const llvm::APSInt &value) {
    return llvm::APSInt::compareValues(value, min) >= 0 &&
           llvm::APSInt::compareValues(value, max) <= 0;
  });
}

/// Returns the range of the values `op` reads: its own quant_min/quant_max,
/// or else the range of a producing quantize_affine, which clamps its result
/// to it. Returns nullopt when neither is known.
static std::optional<QuantMinMax> getInputRange(DequantizeAffineOp op) {
  std::optional<int64_t> quantMin = op.getQuantMin();
  std::optional<int64_t> quantMax = op.getQuantMax();
  if (quantMin && quantMax) {
    return QuantMinMax{*quantMin, *quantMax};
  }
  auto quantizeOp = op.getInput().getDefiningOp<QuantizeAffineOp>();
  // The quantize range describes the same values only when both ops interpret
  // them with the same signedness.
  if (!quantizeOp || quantizeOp.getStorageUnsigned() != op.getInputUnsigned()) {
    return std::nullopt;
  }
  return QuantMinMax{quantizeOp.getQuantMin(), quantizeOp.getQuantMax()};
}

/// The storage a dequantize_affine input is packed into.
struct PackedStorage {
  unsigned bitWidth;
  /// The zero point values, read with the op's signedness. Empty without a
  /// zero point.
  SmallVector<llvm::APSInt> zeroPoint;
};

/// Returns the narrowest storage of at least `minBitWidth` bits the input of
/// `op` can be packed into, or nullopt if its type, range or zero point rules
/// out packing.
static std::optional<PackedStorage>
getPackedStorage(DequantizeAffineOp op, unsigned minBitWidth,
                 SymbolTableCollection &symbolTables) {
  // Quantized values arrive in i8; wider storage would need a truncation
  // before packing.
  auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
  if (!inputType || !inputType.getElementType().isInteger(8) ||
      inputType.getRank() == 0 || !inputType.hasStaticShape() ||
      inputType.getNumElements() == 0) {
    LDBG() << "input is not a static, non-empty i8 tensor: " << op;
    return std::nullopt;
  }
  std::optional<QuantMinMax> range = getInputRange(op);
  if (!range) {
    LDBG() << "input range is unknown: " << op;
    return std::nullopt;
  }
  SmallVector<llvm::APSInt> zeroPoint;
  unsigned zeroPointBitWidth = 0;
  if (Value zeroPointValue = op.getZeroPoint()) {
    auto elementType =
        dyn_cast<IntegerType>(getElementTypeOrSelf(zeroPointValue.getType()));
    Attribute attr = getConstantAttr(zeroPointValue, symbolTables);
    std::optional<SmallVector<llvm::APSInt>> values =
        elementType && attr ? getIntegerValues(attr, op.getZpUnsigned())
                            : std::nullopt;
    if (!values) {
      LDBG() << "zero point is not an integer constant: " << op;
      return std::nullopt;
    }
    zeroPoint = std::move(*values);
    zeroPointBitWidth = elementType.getWidth();
  } else if (op.getInputUnsigned()) {
    // Packed storage is signed, and an unsigned input becomes signed by
    // shifting both its values and its zero point.
    LDBG() << "unsigned input has no zero point to shift: " << op;
    return std::nullopt;
  }
  for (unsigned bitWidth : kPackedBitWidths) {
    if (bitWidth < minBitWidth) {
      continue;
    }
    QuantMinMax storage = *getStorageRange(bitWidth, op.getInputUnsigned());
    if (range->min < storage.min || range->max > storage.max) {
      continue;
    }
    // Packing works row by row, while narrow-type emulation indexes the
    // flattened tensor; the layouts agree only when every row starts on a byte
    // boundary.
    if (inputType.getShape().back() % (8 / bitWidth) != 0) {
      continue;
    }
    // dequantize_affine requires the zero point to fit the input element
    // type, so it must also fit i<bitWidth>. For an unsigned input, the
    // shifted zero point is signed and must also fit the zero point's own
    // element type.
    if (!allValuesInRange(zeroPoint, storage) ||
        (op.getInputUnsigned() && zeroPointBitWidth < bitWidth)) {
      continue;
    }
    return PackedStorage{bitWidth, std::move(zeroPoint)};
  }
  LDBG() << "range, shape or zero point fits no packed bit width: " << op;
  return std::nullopt;
}

/// Returns true when `value` is a constant expression whose roots are inline
/// constants or loads of immutable globals with an inline initial value.
static bool
isComputedFromInlineData(Value value,
                         const IREE::Util::ConstExprAnalysis &constExprs,
                         SymbolTableCollection &symbolTables) {
  const IREE::Util::ConstExprAnalysis::ConstValueInfo *info =
      constExprs.lookup(value);
  if (!info || !info->isConstExpr()) {
    return false;
  }
  return llvm::all_of(info->roots, [&](Value root) {
    return isa_and_present<ElementsAttr, IntegerAttr, FloatAttr>(
        getConstantAttr(root, symbolTables));
  });
}

/// Packs the i8 tensor `input`, whose values fit in `bitWidth` bits, along its
/// innermost dimension and returns the result bitcast to i<bitWidth>. Unsigned
/// values are stored minus 2^(bitWidth - 1), which makes them signed.
static Value buildPackedInput(OpBuilder &builder, Location loc, Value input,
                              unsigned bitWidth, bool isUnsigned) {
  auto inputType = cast<RankedTensorType>(input.getType());
  int64_t rank = inputType.getRank();
  int64_t valuesPerByte = 8 / bitWidth;
  SmallVector<int64_t> packedShape(inputType.getShape());
  packedShape.back() /= valuesPerByte;
  auto packedType = RankedTensorType::get(packedShape, builder.getI8Type());

  // Lane `j` holds the elements that go to bits
  // [j * bitWidth, (j + 1) * bitWidth) of each byte.
  SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes =
      getAsIndexOpFoldResult(builder.getContext(), packedShape);
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
  strides.back() = builder.getIndexAttr(valuesPerByte);
  SmallVector<Value> lanes;
  for (int64_t lane = 0; lane < valuesPerByte; ++lane) {
    offsets.back() = builder.getIndexAttr(lane);
    lanes.push_back(tensor::ExtractSliceOp::create(
        builder, loc, packedType, input, offsets, sizes, strides));
  }

  Value init =
      tensor::EmptyOp::create(builder, loc, packedShape, builder.getI8Type());
  SmallVector<AffineMap> maps(valuesPerByte + 1,
                              builder.getMultiDimIdentityMap(rank));
  SmallVector<utils::IteratorType> iteratorTypes(rank,
                                                 utils::IteratorType::parallel);
  auto packingOp = linalg::GenericOp::create(
      builder, loc, packedType, lanes, init, maps, iteratorTypes,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        Type i8Type = nested.getI8Type();
        Value mask = arith::ConstantIntOp::create(nested, nestedLoc, i8Type,
                                                  (1 << bitWidth) - 1);
        Value byte;
        for (int64_t lane = 0; lane < valuesPerByte; ++lane) {
          Value bits = args[lane];
          // The shift of the last lane drops the bits above its field.
          if (lane != valuesPerByte - 1) {
            bits = arith::AndIOp::create(nested, nestedLoc, bits, mask);
          }
          if (lane == 0) {
            byte = bits;
            continue;
          }
          Value shift = arith::ConstantIntOp::create(nested, nestedLoc, i8Type,
                                                     lane * bitWidth);
          bits = arith::ShLIOp::create(nested, nestedLoc, bits, shift);
          byte = arith::OrIOp::create(nested, nestedLoc, byte, bits);
        }
        if (isUnsigned) {
          // Within a bitWidth-bit field, subtracting 2^(bitWidth - 1) flips
          // the top bit, so one xor shifts every value in the byte.
          APInt topBits =
              APInt::getSplat(8, APInt::getOneBitSet(bitWidth, bitWidth - 1));
          Value flip = arith::ConstantOp::create(
              nested, nestedLoc, nested.getIntegerAttr(i8Type, topBits));
          byte = arith::XOrIOp::create(nested, nestedLoc, byte, flip);
        }
        linalg::YieldOp::create(nested, nestedLoc, byte);
      });

  auto narrowType = RankedTensorType::get(inputType.getShape(),
                                          builder.getIntegerType(bitWidth));
  return IREE::Flow::TensorBitCastOp::create(builder, loc, narrowType,
                                             packingOp.getResult(0),
                                             /*source_dims=*/ValueRange{},
                                             /*result_dims=*/ValueRange{});
}

/// Makes `op`, whose unsigned input is now stored minus 2^(bitWidth - 1), read
/// that input as signed. Shifting the zero point by the same amount keeps
/// every `input - zero_point` unchanged. A zero point that becomes 0 is
/// dropped, which makes the op symmetric.
static void shiftToSignedInput(OpBuilder &builder, DequantizeAffineOp op,
                               const PackedStorage &storage) {
  int64_t offset = int64_t{1} << (storage.bitWidth - 1);
  Value zeroPoint = op.getZeroPoint();
  Type zeroPointType = zeroPoint.getType();
  unsigned zeroPointBitWidth =
      getElementTypeOrSelf(zeroPointType).getIntOrFloatBitWidth();
  SmallVector<APInt> shifted =
      llvm::map_to_vector(storage.zeroPoint, [&](const llvm::APSInt &value) {
        return APInt(zeroPointBitWidth, value.getExtValue() - offset,
                     /*isSigned=*/true);
      });
  if (llvm::all_of(shifted,
                   [](const APInt &value) { return value.isZero(); })) {
    // The maps are ordered input, scale, zero point, output.
    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    maps.erase(maps.begin() + 2);
    op.setIndexingMapsAttr(builder.getAffineMapArrayAttr(maps));
    op.getZeroPointMutable().clear();
  } else {
    TypedAttr shiftedAttr;
    if (auto shapedType = dyn_cast<ShapedType>(zeroPointType)) {
      shiftedAttr = DenseIntElementsAttr::get(shapedType, shifted);
    } else {
      shiftedAttr = builder.getIntegerAttr(zeroPointType, shifted.front());
    }
    op.getZeroPointMutable().assign(
        arith::ConstantOp::create(builder, op.getLoc(), shiftedAttr));
  }
  if (std::optional<int64_t> quantMin = op.getQuantMin()) {
    op.setQuantMin(*quantMin - offset);
  }
  if (std::optional<int64_t> quantMax = op.getQuantMax()) {
    op.setQuantMax(*quantMax - offset);
  }
  op.setInputUnsigned(false);
  op.setZpUnsigned(false);
}

struct PackSubByteQuantizedConstantsPass final
    : impl::PackSubByteQuantizedConstantsPassBase<
          PackSubByteQuantizedConstantsPass> {
  using Base::Base;

  void runOnOperation() override {
    // ConstExprAnalysis is not updated by the rewrites, so collect every
    // candidate first.
    SymbolTableCollection symbolTables;
    SmallVector<std::pair<DequantizeAffineOp, PackedStorage>> candidates;
    {
      const auto &constExprs = getAnalysis<IREE::Util::ConstExprAnalysis>();
      getOperation().walk([&](DequantizeAffineOp op) {
        std::optional<PackedStorage> storage =
            getPackedStorage(op, minBitWidth, symbolTables);
        if (!storage) {
          return;
        }
        if (!isComputedFromInlineData(op.getInput(), constExprs,
                                      symbolTables)) {
          LDBG() << "input is not computed from inline data: " << op;
          return;
        }
        candidates.emplace_back(op, std::move(*storage));
      });
    }
    // An input that another op still reads as i8 would be stored twice, so it
    // is only packed when every reader is a dequantize that gets packed.
    llvm::SmallPtrSet<Operation *, 16> candidateOps;
    for (const auto &[op, storage] : candidates) {
      candidateOps.insert(op);
    }
    llvm::erase_if(candidates, [&](const auto &candidate) {
      DequantizeAffineOp op = candidate.first;
      bool isReadUnpacked =
          llvm::any_of(op.getInput().getUsers(), [&](Operation *user) {
            return !candidateOps.contains(user);
          });
      if (isReadUnpacked) {
        LDBG() << "input is also read unpacked: " << op;
      }
      return isReadUnpacked;
    });
    for (auto &[op, storage] : candidates) {
      OpBuilder builder(op);
      bool isUnsigned = op.getInputUnsigned();
      Value packed = buildPackedInput(builder, op.getLoc(), op.getInput(),
                                      storage.bitWidth, isUnsigned);
      op.getInputMutable().assign(packed);
      if (isUnsigned) {
        shiftToSignedInput(builder, op, storage);
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
