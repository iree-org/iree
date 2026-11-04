// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
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
using ConstValueInfo = IREE::Util::ConstExprAnalysis::ConstValueInfo;

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

/// Returns the value that the reshapes producing `value` read, or `value`
/// itself if no reshape produces it. Reshapes keep the data unchanged.
static Value getReshapeSource(Value value) {
  while (isa_and_present<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(
      value.getDefiningOp())) {
    value = value.getDefiningOp()->getOperand(0);
  }
  return value;
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
    // The check reads only the zero point values, which reshapes keep.
    Attribute attr =
        getConstantAttr(getReshapeSource(zeroPointValue), symbolTables);
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

/// Returns true when `info` is a constant expression whose roots are inline
/// constants or loads of immutable globals with an inline initial value.
static bool isComputedFromInlineData(const ConstValueInfo &info,
                                     SymbolTableCollection &symbolTables) {
  if (!info.isConstExpr()) {
    return false;
  }
  return llvm::all_of(info.roots, [&](Value root) {
    return isa_and_present<ElementsAttr, IntegerAttr, FloatAttr>(
        getConstantAttr(root, symbolTables));
  });
}

/// Returns true when packing the constant expression of `info` into
/// `bitWidth` bits takes no more bytes than its roots, which hoisting accepts
/// under any size threshold. Returns false if a root has a dynamic shape, whose
/// size is unknown.
static bool isPackingNoLargerThanRoots(const ConstValueInfo &info,
                                       unsigned bitWidth) {
  int64_t rootBytes = 0;
  for (Value root : info.roots) {
    auto rootType = dyn_cast<ShapedType>(root.getType());
    if (!rootType) {
      continue;
    }
    if (!rootType.hasStaticShape()) {
      return false;
    }
    rootBytes += IREE::Util::getRoundedPhysicalStorageSize(rootType);
  }
  auto type = cast<ShapedType>(info.constValue.getType());
  return type.getNumElements() * bitWidth / 8 <= rootBytes;
}

/// Returns every use that reads the data of `value`: reshapes are followed up
/// to their source and back down, across all loads when the source is a global
/// load. Returns nullopt if the global has users other than loads.
static std::optional<SmallVector<OpOperand *>>
getUsesOfSameData(Value value, const SymbolUserMap &symbolUsers,
                  SymbolTableCollection &symbolTables) {
  Value source = getReshapeSource(value);
  SmallVector<Value> worklist;
  if (IREE::Util::GlobalOpInterface globalOp =
          getLoadedGlobal(source, symbolTables)) {
    for (Operation *user : symbolUsers.getUsers(globalOp)) {
      auto loadOp = dyn_cast<IREE::Util::GlobalLoadOpInterface>(user);
      if (!loadOp) {
        return std::nullopt;
      }
      worklist.push_back(loadOp.getLoadedGlobalValue());
    }
  } else {
    worklist.push_back(source);
  }
  SmallVector<OpOperand *> uses;
  while (!worklist.empty()) {
    Value data = worklist.pop_back_val();
    for (OpOperand &use : data.getUses()) {
      if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(use.getOwner())) {
        worklist.push_back(use.getOwner()->getResult(0));
      } else {
        uses.push_back(&use);
      }
    }
  }
  return uses;
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
  return IREE::TensorExt::BitCastOp::create(builder, loc, narrowType,
                                            packingOp.getResult(0),
                                            /*source_dims=*/ValueRange{},
                                            /*result_dims=*/ValueRange{});
}

/// A dequantize whose input fits `bitWidth` bits.
struct PackingCandidate {
  DequantizeAffineOp op;
  unsigned bitWidth;
};

/// Returns the dequantizes in `moduleOp` whose input fits a packed bit width
/// and is computed from no fewer bytes of inline data than its packing takes.
static SmallVector<PackingCandidate>
collectCandidates(ModuleOp moduleOp,
                  const IREE::Util::ConstExprAnalysis &constExprs,
                  SymbolTableCollection &symbolTables) {
  SmallVector<PackingCandidate> candidates;
  moduleOp.walk([&](DequantizeAffineOp op) {
    std::optional<unsigned> bitWidth = getPackedBitWidth(op, symbolTables);
    if (!bitWidth) {
      return;
    }
    const ConstValueInfo *info = constExprs.lookup(op.getInput());
    if (!info || !isComputedFromInlineData(*info, symbolTables)) {
      LDBG() << "input is not computed from inline data: " << op;
      return;
    }
    if (!isPackingNoLargerThanRoots(*info, *bitWidth)) {
      LDBG() << "packed input may be larger than its inline data: " << op;
      return;
    }
    candidates.push_back({op, *bitWidth});
  });
  return candidates;
}

/// Drops the candidates whose data another op also reads as i8, or another
/// candidate reads in a different shape, since packing it would store the data
/// twice. Every load of a global, in any function, and every reshape reads the
/// same data, and only the input operand of a candidate reads it packed.
static void
dropCandidatesWithOtherReaders(SmallVectorImpl<PackingCandidate> &candidates,
                               ModuleOp moduleOp,
                               SymbolTableCollection &symbolTables) {
  if (candidates.empty()) {
    return;
  }
  llvm::SmallPtrSet<OpOperand *, 16> candidateInputs;
  for (PackingCandidate candidate : candidates) {
    candidateInputs.insert(&candidate.op.getInputMutable());
  }
  SymbolUserMap symbolUsers(symbolTables, moduleOp);
  llvm::erase_if(candidates, [&](PackingCandidate candidate) {
    Value input = candidate.op.getInput();
    std::optional<SmallVector<OpOperand *>> uses =
        getUsesOfSameData(input, symbolUsers, symbolTables);
    bool hasOtherReader = !uses || llvm::any_of(*uses, [&](OpOperand *use) {
      return !candidateInputs.contains(use) ||
             use->get().getType() != input.getType();
    });
    if (hasOtherReader) {
      LDBG() << "input is also read unpacked or in another shape: "
             << candidate.op;
    }
    return hasOtherReader;
  });
}

/// Packs each candidate input once, at the widest bit width its dequantizes
/// need, which holds the range of every one of them. The packing goes next to
/// the input's definition: the const-expr analysis does not follow uses into
/// nested regions, so packing next to a dequantize in control flow would not
/// be hoisted.
static void packCandidates(ArrayRef<PackingCandidate> candidates) {
  DenseMap<Value, unsigned> inputBitWidths;
  for (PackingCandidate candidate : candidates) {
    unsigned &inputBitWidth = inputBitWidths[candidate.op.getInput()];
    inputBitWidth = std::max(inputBitWidth, candidate.bitWidth);
  }
  DenseMap<Value, Value> packedInputs;
  for (PackingCandidate candidate : candidates) {
    Value input = candidate.op.getInput();
    Value &packed = packedInputs[input];
    if (!packed) {
      OpBuilder builder(candidate.op.getContext());
      builder.setInsertionPointAfterValue(input);
      packed = buildPackedInput(builder, input.getLoc(), input,
                                inputBitWidths[input]);
    }
    candidate.op.getInputMutable().assign(packed);
  }
}

struct PackSubByteQuantizedConstantsPass final
    : impl::PackSubByteQuantizedConstantsPassBase<
          PackSubByteQuantizedConstantsPass> {
  void runOnOperation() override {
    // ConstExprAnalysis is not updated by the rewrites, so collect every
    // candidate first.
    SymbolTableCollection symbolTables;
    SmallVector<PackingCandidate> candidates = collectCandidates(
        getOperation(), getAnalysis<IREE::Util::ConstExprAnalysis>(),
        symbolTables);
    dropCandidatesWithOtherReaders(candidates, getOperation(), symbolTables);
    packCandidates(candidates);
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
