// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/Util/Analysis/Constant/ConstExpr.h"
#include "iree/compiler/Dialect/Util/Analysis/Constant/OpOracle.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/GlobalOptimization/QuantizationUtils.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

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
/// nullopt if its type, shape, range or zero point rules out packing.
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
  SmallVector<APSInt> zeroPointValues;
  if (Value zeroPoint = op.getZeroPoint()) {
    // The check reads only the zero point values, which reshapes keep.
    Attribute attr = getConstantAttr(getReshapeSource(zeroPoint), symbolTables);
    std::optional<SmallVector<APSInt>> values =
        attr ? getIntegerValues(attr, op.getZpUnsigned()) : std::nullopt;
    if (!values) {
      LDBG() << "zero point is not an integer constant: " << op;
      return std::nullopt;
    }
    zeroPointValues = std::move(*values);
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
    if (!allValuesInRange(zeroPointValues, storage)) {
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

/// Returns the bytes `values` take, counting values that are not shaped as
/// zero, or nullopt if one has a dynamic shape.
static std::optional<int64_t> getStorageBytes(ArrayRef<Value> values) {
  int64_t bytes = 0;
  for (Value value : values) {
    auto type = dyn_cast<ShapedType>(value.getType());
    if (!type) {
      continue;
    }
    if (!type.hasStaticShape()) {
      return std::nullopt;
    }
    bytes += IREE::Util::getRoundedPhysicalStorageSize(type);
  }
  return bytes;
}

/// Returns true when packing the input that `info` describes into `bitWidth`
/// bits stores no more bytes than leaving it unpacked. Unpacked, hoisting
/// stores an input that is a leaf no larger than its roots in a global of its
/// own, which packing shrinks. Any other input may be computed where it is
/// used, from roots that stay, so its packing must take no more bytes than
/// `freedRoots`, the roots that packing frees. Either way, hoisting compares
/// the packing with all its roots and accepts it at any size threshold.
static bool isPackingNoLargerThanUnpacked(const ConstValueInfo &info,
                                          unsigned bitWidth,
                                          ArrayRef<Value> freedRoots) {
  std::optional<int64_t> freedBytes = getStorageBytes(freedRoots);
  std::optional<int64_t> rootBytes =
      getStorageBytes(llvm::to_vector(info.roots));
  if (!freedBytes || !rootBytes) {
    return false;
  }
  auto inputType = cast<ShapedType>(info.constValue.getType());
  int64_t packedBytes = IREE::Util::getRoundedPhysicalStorageSize(
      inputType.getNumElements(),
      IntegerType::get(inputType.getContext(), bitWidth));
  if (packedBytes <= *freedBytes) {
    return true;
  }
  return !info.isRoot && IREE::Util::isHoistableConstExprLeaf(&info) &&
         IREE::Util::getRoundedPhysicalStorageSize(inputType) <= *rootBytes;
}

/// Returns every use that reads the data of `value`: reshapes are followed up
/// to their source and back down, across all loads when the source is a global
/// load. Returns nullopt if the global is public, so that code outside the
/// module can read it, or has users other than loads.
static std::optional<SmallVector<OpOperand *>>
getUsesOfSameData(Value value, const SymbolUserMap &symbolUsers,
                  SymbolTableCollection &symbolTables) {
  Value source = getReshapeSource(value);
  SmallVector<Value> worklist;
  if (IREE::Util::GlobalOpInterface globalOp =
          getLoadedGlobal(source, symbolTables)) {
    if (!globalOp.isGlobalPrivate()) {
      return std::nullopt;
    }
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

/// Returns true when packing frees the data of `root`: every op that reads it,
/// directly or through other ops, is in `producers`, which compute the
/// candidate inputs, and every read leads to one of `candidateInputs`.
static bool
isFreedByPacking(Value root, const SetVector<Operation *> &producers,
                 const SmallPtrSetImpl<OpOperand *> &candidateInputs,
                 const SymbolUserMap &symbolUsers,
                 SymbolTableCollection &symbolTables) {
  std::optional<SmallVector<OpOperand *>> uses =
      getUsesOfSameData(root, symbolUsers, symbolTables);
  if (!uses) {
    return false;
  }
  SmallVector<OpOperand *> worklist = std::move(*uses);
  SmallPtrSet<Operation *, 8> visited;
  while (!worklist.empty()) {
    OpOperand *use = worklist.pop_back_val();
    if (candidateInputs.contains(use)) {
      continue;
    }
    Operation *owner = use->getOwner();
    if (!producers.contains(owner)) {
      return false;
    }
    if (!visited.insert(owner).second) {
      continue;
    }
    for (Value result : owner->getResults()) {
      for (OpOperand &resultUse : result.getUses()) {
        worklist.push_back(&resultUse);
      }
    }
  }
  return true;
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
  /// The constant expression of the input.
  const ConstValueInfo *info;
};

/// Returns the dequantizes in `moduleOp` whose input fits a packed bit width
/// and is computed from inline data at the top level of a function.
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
    // Hoisting only reaches the top level of a function, where the packing must
    // go next to the input.
    if (!isa<FunctionOpInterface>(
            op.getInput().getParentRegion()->getParentOp())) {
      LDBG() << "input is defined inside control flow: " << op;
      return;
    }
    candidates.push_back({op, *bitWidth, info});
  });
  return candidates;
}

/// Drops the candidates whose packing could store more bytes than leaving the
/// input unpacked, or might not be hoisted. The input data must not be read by
/// another op as i8, or by another candidate in a different shape, which would
/// keep it stored twice. Every load of a global, in any function, and every
/// reshape reads the same data, and only the input operand of a candidate reads
/// it packed. Dropping a candidate can keep data that others rely on being
/// freed, so this repeats until none is dropped.
static void
dropCandidatesThatGrowStorage(SmallVectorImpl<PackingCandidate> &candidates,
                              ModuleOp moduleOp,
                              SymbolTableCollection &symbolTables) {
  if (candidates.empty()) {
    return;
  }
  SymbolUserMap symbolUsers(symbolTables, moduleOp);
  BackwardSliceOptions sliceOptions;
  sliceOptions.inclusive = true;
  sliceOptions.omitBlockArguments = true;
  size_t numCandidates;
  do {
    numCandidates = candidates.size();
    SmallPtrSet<OpOperand *, 16> candidateInputs;
    SetVector<Operation *> producers;
    for (PackingCandidate &candidate : candidates) {
      candidateInputs.insert(&candidate.op.getInputMutable());
      [[maybe_unused]] LogicalResult result =
          getBackwardSlice(candidate.op.getInput(), &producers, sliceOptions);
      assert(result.succeeded() && "expected the backward slice to succeed");
    }
    // Whether packing frees a root depends only on the root in each round.
    DenseMap<Value, bool> isRootFreed;
    llvm::erase_if(candidates, [&](PackingCandidate &candidate) {
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
        return true;
      }
      SmallVector<Value> freedRoots;
      for (Value root : candidate.info->roots) {
        if (!isa<ShapedType>(root.getType())) {
          continue;
        }
        auto [it, inserted] = isRootFreed.try_emplace(root);
        if (inserted) {
          it->second = isFreedByPacking(root, producers, candidateInputs,
                                        symbolUsers, symbolTables);
        }
        if (it->second) {
          freedRoots.push_back(root);
        }
      }
      if (!isPackingNoLargerThanUnpacked(*candidate.info, candidate.bitWidth,
                                         freedRoots)) {
        LDBG() << "packing may store more bytes than the unpacked input: "
               << candidate.op;
        return true;
      }
      return false;
    });
  } while (candidates.size() != numCandidates);
}

/// Packs each candidate input value once, at the widest bit width that the
/// dequantizes reading its data need, which holds the range of every one of
/// them. Every load of a global packs at the same width, so the packings in
/// different functions are identical and are merged later. The packing goes
/// next to the input's definition: the const-expr analysis does not follow uses
/// into nested regions, so packing next to a dequantize in control flow would
/// not be hoisted.
static void packCandidates(MutableArrayRef<PackingCandidate> candidates,
                           SymbolTableCollection &symbolTables) {
  DenseMap<Operation *, unsigned> globalBitWidths;
  DenseMap<Value, unsigned> inputBitWidths;
  // Returns the widest bit width for the data that `input` reads.
  auto getBitWidth = [&](Value input) -> unsigned & {
    if (IREE::Util::GlobalOpInterface globalOp =
            getLoadedGlobal(getReshapeSource(input), symbolTables)) {
      return globalBitWidths[globalOp.getOperation()];
    }
    return inputBitWidths[input];
  };
  for (PackingCandidate &candidate : candidates) {
    unsigned &bitWidth = getBitWidth(candidate.op.getInput());
    bitWidth = std::max(bitWidth, candidate.bitWidth);
  }
  DenseMap<Value, Value> packedInputs;
  for (PackingCandidate &candidate : candidates) {
    Value input = candidate.op.getInput();
    Value &packed = packedInputs[input];
    if (!packed) {
      OpBuilder builder(candidate.op.getContext());
      builder.setInsertionPointAfterValue(input);
      packed =
          buildPackedInput(builder, input.getLoc(), input, getBitWidth(input));
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
    dropCandidatesThatGrowStorage(candidates, getOperation(), symbolTables);
    packCandidates(candidates, symbolTables);
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
