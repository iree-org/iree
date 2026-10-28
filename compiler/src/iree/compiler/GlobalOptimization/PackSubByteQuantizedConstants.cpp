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
#include "llvm/ADT/DenseMap.h"
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

#define DEBUG_TYPE "iree-global-opt-pack-sub-byte-quantized-constants"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_PACKSUBBYTEQUANTIZEDCONSTANTSPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

using IREE::LinalgExt::DequantizeAffineOp;

namespace {

/// Bit widths quantized values can be packed into, narrowest first. Each
/// divides 8, so a byte holds a whole number of values.
constexpr unsigned kPackedBitWidths[] = {2, 4};

/// Returns the global that `value` loads, or null if it is not a global load.
static IREE::Util::GlobalOpInterface
getLoadedGlobal(Value value, SymbolTableCollection &symbolTables) {
  auto loadOp = value.getDefiningOp<IREE::Util::GlobalLoadOpInterface>();
  if (!loadOp) {
    return {};
  }
  return IREE::Util::lookupGlobalOp(loadOp, loadOp.getGlobalAttr(),
                                    symbolTables);
}

/// Returns the attribute holding `value` when it is a constant or a load of an
/// immutable global, or null otherwise. For a global this is its initial
/// value, which may be a parameter reference rather than data.
static Attribute getConstantAttr(Value value,
                                 SymbolTableCollection &symbolTables) {
  Attribute attr;
  if (matchPattern(value, m_Constant(&attr))) {
    return attr;
  }
  IREE::Util::GlobalOpInterface globalOp = getLoadedGlobal(value, symbolTables);
  if (!globalOp || globalOp.isGlobalMutable()) {
    return {};
  }
  return globalOp.getGlobalInitialValue();
}

/// Returns the integers in `attr`, read as unsigned if `isUnsigned`, or nullopt
/// when `attr` holds no readable integers. A splat gives a single value.
static std::optional<SmallVector<APSInt>> getIntegerValues(Attribute attr,
                                                           bool isUnsigned) {
  using Values = std::optional<SmallVector<APSInt>>;
  auto toAPSInt = [&](const APInt &value) { return APSInt(value, isUnsigned); };
  return llvm::TypeSwitch<Attribute, Values>(attr)
      .Case([&](IntegerAttr intAttr) -> Values {
        return SmallVector<APSInt>{toAPSInt(intAttr.getValue())};
      })
      .Case([&](DenseIntElementsAttr denseAttr) -> Values {
        if (denseAttr.isSplat()) {
          return SmallVector<APSInt>{
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
static bool allValuesInRange(ArrayRef<APSInt> values, QuantMinMax range) {
  APSInt min = APSInt::get(range.min);
  APSInt max = APSInt::get(range.max);
  return llvm::all_of(values, [&](const APSInt &value) {
    return APSInt::compareValues(value, min) >= 0 &&
           APSInt::compareValues(value, max) <= 0;
  });
}

/// Returns the narrowest bit width the input of `op` can be packed into, or
/// nullopt if its type, range or zero point rules out packing.
static std::optional<unsigned>
getPackedBitWidth(DequantizeAffineOp op, SymbolTableCollection &symbolTables) {
  // Quantized values arrive in i8; wider storage would need a truncation
  // before packing.
  auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
  if (!inputType || !inputType.getElementType().isInteger(8) ||
      inputType.getRank() == 0 || !inputType.hasStaticShape() ||
      inputType.getNumElements() == 0) {
    LDBG() << "input is not a static, non-empty i8 tensor: " << op;
    return std::nullopt;
  }
  std::optional<int64_t> quantMin = op.getQuantMin();
  if (!quantMin) {
    LDBG() << "input range is unknown: " << op;
    return std::nullopt;
  }
  QuantMinMax range{*quantMin, *op.getQuantMax()};
  SmallVector<APSInt> zeroPoint;
  if (Value zeroPointValue = op.getZeroPoint()) {
    // A reshape keeps the zero point values, which is all the check reads.
    while (isa_and_present<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(
        zeroPointValue.getDefiningOp())) {
      zeroPointValue = zeroPointValue.getDefiningOp()->getOperand(0);
    }
    Attribute attr = getConstantAttr(zeroPointValue, symbolTables);
    std::optional<SmallVector<APSInt>> values =
        attr ? getIntegerValues(attr, op.getZpUnsigned()) : std::nullopt;
    if (!values) {
      LDBG() << "zero point is not an integer constant: " << op;
      return std::nullopt;
    }
    zeroPoint = std::move(*values);
  }
  for (unsigned bitWidth : kPackedBitWidths) {
    QuantMinMax storage = *getStorageRange(bitWidth, op.getInputUnsigned());
    if (range.min < storage.min || range.max > storage.max) {
      continue;
    }
    // Packing works row by row, while narrow-type emulation indexes the
    // flattened tensor; the layouts agree only when every row starts on a byte
    // boundary.
    if (inputType.getShape().back() % (8 / bitWidth) != 0) {
      continue;
    }
    // dequantize_affine requires the zero point to fit the input element
    // type, so it must also fit i<bitWidth>.
    if (!allValuesInRange(zeroPoint, storage)) {
      continue;
    }
    return bitWidth;
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

/// Returns the values that hold the same data as `value`: every load of the
/// global that `value` loads, or else `value` itself. Returns nullopt when the
/// global has users other than loads.
static std::optional<SmallVector<Value>>
getValuesWithSameData(Value value, const SymbolUserMap &symbolUsers,
                      SymbolTableCollection &symbolTables) {
  IREE::Util::GlobalOpInterface globalOp = getLoadedGlobal(value, symbolTables);
  if (!globalOp) {
    return SmallVector<Value>{value};
  }
  SmallVector<Value> values;
  for (Operation *user : symbolUsers.getUsers(globalOp)) {
    auto otherLoadOp = dyn_cast<IREE::Util::GlobalLoadOpInterface>(user);
    if (!otherLoadOp) {
      return std::nullopt;
    }
    values.push_back(otherLoadOp.getLoadedGlobalValue());
  }
  return values;
}

/// Packs the i8 tensor `input`, whose values fit in `bitWidth` bits and whose
/// innermost dimension is a multiple of 8 / `bitWidth`, along that dimension
/// and returns the result bitcast to i<bitWidth>.
static Value buildPackedInput(OpBuilder &builder, Location loc, Value input,
                              unsigned bitWidth) {
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
        Value byte = arith::AndIOp::create(nested, nestedLoc, args[0], mask);
        for (int64_t lane = 1; lane < valuesPerByte; ++lane) {
          Value bits = args[lane];
          // The shift of the last lane drops the bits above its field.
          if (lane != valuesPerByte - 1) {
            bits = arith::AndIOp::create(nested, nestedLoc, bits, mask);
          }
          Value shift = arith::ConstantIntOp::create(nested, nestedLoc, i8Type,
                                                     lane * bitWidth);
          bits = arith::ShLIOp::create(nested, nestedLoc, bits, shift);
          byte = arith::OrIOp::create(nested, nestedLoc, byte, bits);
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

struct PackSubByteQuantizedConstantsPass final
    : impl::PackSubByteQuantizedConstantsPassBase<
          PackSubByteQuantizedConstantsPass> {
  void runOnOperation() override {
    // ConstExprAnalysis is not updated by the rewrites, so collect every
    // candidate first.
    SymbolTableCollection symbolTables;
    SmallVector<std::pair<DequantizeAffineOp, unsigned>> candidates;
    llvm::SmallPtrSet<Operation *, 16> candidateOps;
    {
      const auto &constExprs = getAnalysis<IREE::Util::ConstExprAnalysis>();
      getOperation().walk([&](DequantizeAffineOp op) {
        std::optional<unsigned> bitWidth = getPackedBitWidth(op, symbolTables);
        if (!bitWidth) {
          return;
        }
        if (!isComputedFromInlineData(op.getInput(), constExprs,
                                      symbolTables)) {
          LDBG() << "input is not computed from inline data: " << op;
          return;
        }
        candidates.emplace_back(op, *bitWidth);
        candidateOps.insert(op);
      });
    }
    if (candidates.empty()) {
      return;
    }
    // An input that another op still reads as i8 would be stored twice, so it
    // is only packed when every reader is a dequantize that gets packed. Every
    // load of a global reads the same data, in any function.
    SymbolUserMap symbolUsers(symbolTables, getOperation());
    llvm::erase_if(candidates, [&](const auto &candidate) {
      DequantizeAffineOp op = candidate.first;
      std::optional<SmallVector<Value>> sameData =
          getValuesWithSameData(op.getInput(), symbolUsers, symbolTables);
      bool isReadUnpacked =
          !sameData || llvm::any_of(*sameData, [&](Value value) {
            return llvm::any_of(value.getUsers(), [&](Operation *user) {
              return !candidateOps.contains(user);
            });
          });
      if (isReadUnpacked) {
        LDBG() << "input is also read unpacked: " << op;
      }
      return isReadUnpacked;
    });
    // Each input is packed once, at the widest bit width its dequantizes need,
    // which holds the range of every one of them. The packing goes next to the
    // input's definition: the const-expr analysis does not follow uses into
    // nested regions, so packing next to a dequantize in control flow would not
    // be hoisted.
    DenseMap<Value, unsigned> inputBitWidths;
    for (auto [op, bitWidth] : candidates) {
      unsigned &inputBitWidth = inputBitWidths[op.getInput()];
      inputBitWidth = std::max(inputBitWidth, bitWidth);
    }
    DenseMap<Value, Value> packedInputs;
    for (DequantizeAffineOp op : llvm::make_first_range(candidates)) {
      Value input = op.getInput();
      Value &packed = packedInputs[input];
      if (!packed) {
        OpBuilder builder(op.getContext());
        builder.setInsertionPointAfterValue(input);
        packed = buildPackedInput(builder, input.getLoc(), input,
                                  inputBitWidths[input]);
      }
      op.getInputMutable().assign(packed);
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
