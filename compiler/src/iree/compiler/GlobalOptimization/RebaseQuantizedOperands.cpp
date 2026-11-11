// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/Utils/EncodingUtils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "iree/compiler/Dialect/Stream/IR/StreamInterfaces.h"
#include "iree/compiler/Dialect/Util/Analysis/Constant/ConstExpr.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/GlobalOptimization/QuantizationUtils.h"
#include "iree/compiler/GlobalOptimization/QuantizedContraction.h"
#include "iree/compiler/GlobalOptimization/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-global-opt-rebase-quantized-operands"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_REBASEQUANTIZEDOPERANDSPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

using detail::getIntegerContractionMaps;
using detail::getOperandRanges;
using detail::getQuantizedContraction;
using detail::QuantizedContraction;
using IREE::LinalgExt::DequantizeAffineOp;
using IREE::LinalgExt::QuantizeAffineOp;

/// Whether the lhs and rhs storage is unsigned.
using OperandSignedness = std::array<bool, 2>;

/// Which of the lhs and rhs to rebase.
using OperandSelection = std::array<bool, 2>;

/// Values moved by an offset, keyed by the original value and the offset.
using ShiftedValues = DenseMap<std::pair<Value, int64_t>, Value>;

/// An op that ConvertQDQToIntegerMath will rewrite and the operands chosen for
/// rebasing it.
struct Candidate {
  /// The contraction or convolution.
  linalg::LinalgOp op;
  /// Its quantized operands and integer reduction.
  QuantizedContraction contraction;
  /// Indexing maps of the integer contraction: lhs, rhs and result.
  SmallVector<AffineMap> integerMaps;
  /// Static loop ranges of the integer contraction.
  SmallVector<int64_t> iterationSizes;
  /// Encoding resolvers of the targets `op` runs on, asked for contractions
  /// only.
  SetVector<Attribute> resolvers;
  /// The operands chosen for rebasing.
  OperandSelection rebase = {false, false};

  /// Returns the dequantize feeding the lhs for `index` 0 and the rhs for 1.
  DequantizeAffineOp getDequantize(unsigned index) const {
    return index == 0 ? contraction.lhs.dequantize : contraction.rhs.dequantize;
  }
};

/// Returns the bit width of the storage type `op` reads.
static unsigned getStorageBitWidth(DequantizeAffineOp op) {
  return op.getInputType().getElementTypeBitWidth();
}

/// Returns the range of the values `op` reads.
static QuantMinMax getInputRange(DequantizeAffineOp op) {
  return getOperandRanges(op)->input;
}

/// Returns true when every resolver of `candidate` gives the integer
/// contraction a tiled layout if its lhs and rhs storage have the signedness in
/// `isUnsigned`. The encodings carry the indexing maps, loop ranges and element
/// types of the integer contraction.
static bool isTiled(const Candidate &candidate, OperandSignedness isUnsigned) {
  MLIRContext *context = candidate.op->getContext();
  SmallVector<Type> elementTypes;
  for (unsigned index : {0, 1}) {
    elementTypes.push_back(IntegerType::get(
        context, getStorageBitWidth(candidate.getDequantize(index)),
        isUnsigned[index] ? IntegerType::Unsigned : IntegerType::Signless));
  }
  elementTypes.push_back(IntegerType::get(context, kAccumulatorWidth));
  for (unsigned index : {0, 1}) {
    int64_t operandIndex =
        index == 0 ? IREE::Encoding::MATMUL_LHS : IREE::Encoding::MATMUL_RHS;
    auto inputType = dyn_cast<RankedTensorType>(
        candidate.getDequantize(index).getInputType());
    if (!inputType) {
      return false;
    }
    auto encoding = IREE::Encoding::EncodingAttr::get(
        context, operandIndex, IREE::Encoding::EncodingOpType::matmul,
        elementTypes, /*originalElementType=*/{}, candidate.integerMaps,
        candidate.iterationSizes);
    RankedTensorType encodedType = inputType.cloneWithEncoding(encoding);
    for (Attribute resolver : candidate.resolvers) {
      auto materializer =
          dyn_cast<IREE::Encoding::LayoutMaterializerAttr>(resolver);
      if (!materializer ||
          IREE::Codegen::isIdentityLayout(
              getEncodingInfoFromLayout(encodedType, materializer))) {
        return false;
      }
    }
  }
  return true;
}

/// Returns the data a dequantize reads, before any reshapes.
static Value getStorageSource(DequantizeAffineOp op) {
  return getReshapeSource(op.getInput());
}

/// Returns true when `value` holds integers of 2 to 7 bits, which constant
/// evaluation does not fold.
static bool holdsSubByteIntegers(Value value) {
  auto type = dyn_cast<IntegerType>(getElementTypeOrSelf(value.getType()));
  return type && type.getWidth() > 1 && type.getWidth() < 8;
}

/// Returns true when the storage `op` reads comes from a quantize or is
/// computed from inline constants of at least a byte.
static bool hasRebasableSource(DequantizeAffineOp op,
                               const IREE::Util::ConstExprAnalysis &constExprs,
                               const SymbolUserMap &symbolUsers,
                               SymbolTableCollection &symbolTables) {
  Value source = getStorageSource(op);
  if (source.getDefiningOp<QuantizeAffineOp>()) {
    return true;
  }
  const IREE::Util::ConstExprAnalysis::ConstValueInfo *info =
      constExprs.lookup(source);
  return info && isComputedFromInlineData(*info, symbolUsers, symbolTables) &&
         llvm::none_of(info->roots, holdsSubByteIntegers);
}

/// Returns true when the operands in `rebase` have rebasable storage and every
/// rebased symmetric operand, which gains a zero point, meets a constant
/// expression.
static bool canRebase(const Candidate &candidate, OperandSelection rebase,
                      const IREE::Util::ConstExprAnalysis &constExprs,
                      const SymbolUserMap &symbolUsers,
                      SymbolTableCollection &symbolTables) {
  for (unsigned index : {0, 1}) {
    if (!rebase[index]) {
      continue;
    }
    DequantizeAffineOp op = candidate.getDequantize(index);
    if (!hasRebasableSource(op, constExprs, symbolUsers, symbolTables)) {
      return false;
    }
    if (op.isSymmetric() && !constExprs.isConstExprValue(getStorageSource(
                                candidate.getDequantize(1 - index)))) {
      return false;
    }
  }
  return true;
}

/// Chooses the operands of `candidate` to rebase: none if it has no resolvers
/// or its current signedness gets a tiled layout, otherwise the first of
/// {lhs}, {rhs}, {lhs, rhs} that is allowed and gets one.
static OperandSelection chooseRebase(
    const Candidate &candidate, const IREE::Util::ConstExprAnalysis &constExprs,
    const SymbolUserMap &symbolUsers, SymbolTableCollection &symbolTables) {
  if (candidate.resolvers.empty()) {
    return {false, false};
  }
  OperandSignedness isUnsigned = {
      candidate.getDequantize(0).getInputUnsigned(),
      candidate.getDequantize(1).getInputUnsigned()};
  if (isTiled(candidate, isUnsigned)) {
    LDBG() << "current signedness is tiled: " << candidate.op;
    return {false, false};
  }
  for (OperandSelection rebase :
       {OperandSelection{true, false}, OperandSelection{false, true},
        OperandSelection{true, true}}) {
    OperandSignedness rebased = {isUnsigned[0] != rebase[0],
                                 isUnsigned[1] != rebase[1]};
    if (canRebase(candidate, rebase, constExprs, symbolUsers, symbolTables) &&
        isTiled(candidate, rebased)) {
      LDBG() << "rebasing lhs=" << rebase[0] << " rhs=" << rebase[1] << ": "
             << candidate.op;
      return rebase;
    }
  }
  LDBG() << "no allowed rebase is tiled: " << candidate.op;
  return {false, false};
}

/// A quantized value, the dequantizes that read it, the contraction operands
/// those dequantizes feed, and the offset that moves it.
struct RebaseGroup {
  /// The quantize producing the value, or null for a constant expression.
  QuantizeAffineOp quantize;
  /// The dequantizes that read the value.
  SmallVector<DequantizeAffineOp> readers;
  /// The candidates and operand indices that consume a reader.
  SmallVector<std::pair<Candidate *, unsigned>> operands;
  /// Whether the value is stored unsigned.
  bool isUnsigned = false;
  /// The amount that moves the value to the other signedness.
  int64_t offset = 0;
  /// Whether the group lost the agreement of an operand and stays as is.
  bool isDropped = false;
};

/// Returns `range` grown to hold the zero point of `op`: its values when they
/// are constant, or the whole storage grid of signedness `isUnsigned`
/// otherwise. A symmetric `op` leaves `range` unchanged.
template <typename OpTy>
static QuantMinMax unionWithZeroPoint(QuantMinMax range, OpTy op,
                                      unsigned storageBitWidth, bool isUnsigned,
                                      const SymbolUserMap &symbolUsers,
                                      SymbolTableCollection &symbolTables) {
  Value zeroPoint = op.getZeroPoint();
  if (!zeroPoint) {
    return range;
  }
  std::optional<QuantMinMax> values = getConstantIntegerRange(
      zeroPoint, op.getZpUnsigned(), symbolUsers, symbolTables);
  return unionRanges(
      range, values ? *values : *getStorageRange(storageBitWidth, isUnsigned));
}

/// Returns true when the zero point of `op`, if any, has an element type wide
/// enough for the values of a `gridBitWidth`-bit grid.
template <typename OpTy>
static bool hasWideEnoughZeroPoint(OpTy op, unsigned gridBitWidth) {
  Value zeroPoint = op.getZeroPoint();
  return !zeroPoint ||
         getElementTypeOrSelf(zeroPoint.getType()).getIntOrFloatBitWidth() >=
             gridBitWidth;
}

/// Returns true when the float type `op` computes in holds every integer of a
/// `gridBitWidth`-bit grid.
static bool computesGridExactly(QuantizeAffineOp op, unsigned gridBitWidth) {
  return cast<FloatType>(op.getScaleElementType()).getFPMantissaWidth() >=
         gridBitWidth;
}

/// Builds the group of the value `op` reads. The grid it moves on is the
/// narrowest one that holds the bounds of its quantize and of every reader and
/// the values of every zero point. Fails if the value has a reader that is not
/// a dequantize, readers that disagree on its signedness, a zero point too
/// narrow for that grid, or a quantize whose float type does not hold it.
static FailureOr<RebaseGroup>
buildRebaseGroup(DequantizeAffineOp op,
                 const DenseMap<Operation *, Candidate *> &candidateOf,
                 const SymbolUserMap &symbolUsers,
                 SymbolTableCollection &symbolTables) {
  std::optional<SmallVector<OpOperand *>> uses =
      getUsesOfSameData(op.getInput(), symbolUsers, symbolTables);
  if (!uses) {
    return failure();
  }
  RebaseGroup group;
  group.quantize = getStorageSource(op).getDefiningOp<QuantizeAffineOp>();
  group.isUnsigned = op.getInputUnsigned();
  unsigned storageBitWidth = getStorageBitWidth(op);
  QuantMinMax range = getInputRange(op);
  for (OpOperand *use : *uses) {
    auto reader = dyn_cast<DequantizeAffineOp>(use->getOwner());
    if (!reader || use->getOperandNumber() != 0 ||
        reader.getInputUnsigned() != group.isUnsigned) {
      LDBG() << "value has a reader that cannot be rebased with " << op;
      return failure();
    }
    range = unionRanges(range, getInputRange(reader));
    range = unionWithZeroPoint(range, reader, storageBitWidth, group.isUnsigned,
                               symbolUsers, symbolTables);
    for (OpOperand &readerUse : reader->getResult(0).getUses()) {
      if (Candidate *candidate = candidateOf.lookup(readerUse.getOwner())) {
        group.operands.push_back({candidate, readerUse.getOperandNumber()});
      }
    }
    group.readers.push_back(reader);
  }
  if (QuantizeAffineOp quantize = group.quantize) {
    if (quantize.getStorageUnsigned() != group.isUnsigned) {
      return failure();
    }
    range =
        unionRanges(range, {quantize.getQuantMin(), quantize.getQuantMax()});
    range = unionWithZeroPoint(range, quantize, storageBitWidth,
                               group.isUnsigned, symbolUsers, symbolTables);
  }
  std::optional<unsigned> gridBitWidth =
      getGridBitWidth(range, group.isUnsigned, storageBitWidth);
  if (!gridBitWidth ||
      (group.quantize &&
       (!hasWideEnoughZeroPoint(group.quantize, *gridBitWidth) ||
        !computesGridExactly(group.quantize, *gridBitWidth))) ||
      !llvm::all_of(group.readers, [&](DequantizeAffineOp reader) {
        return hasWideEnoughZeroPoint(reader, *gridBitWidth);
      })) {
    LDBG() << "no grid holds every bound and zero point exactly: " << op;
    return failure();
  }
  group.offset = getSignednessFlipOffset(*gridBitWidth, group.isUnsigned);
  return group;
}

/// Returns true when every contraction operand `group` feeds is rebased.
static bool isAgreed(const RebaseGroup &group) {
  return llvm::all_of(group.operands, [](auto operand) {
    auto [candidate, index] = operand;
    return candidate->rebase[index];
  });
}

/// Returns the operand ranges the i32 bound uses for `op` after it moves by
/// `offset` to the other signedness, bounded as an asymmetric operand.
static std::optional<QuantizedOperandRanges>
getRebasedOperandRanges(DequantizeAffineOp op, int64_t offset) {
  QuantMinMax range = getInputRange(op);
  return getQuantizedOperandRanges(
      getStorageBitWidth(op), !op.getInputUnsigned(),
      QuantMinMax{range.min + offset, range.max + offset},
      /*isSymmetric=*/false);
}

/// Returns true when the integer intermediates of `candidate` still fit i32
/// after its operands move by the offsets of their groups.
static bool
fitsAccumulator(const Candidate &candidate,
                const DenseMap<Operation *, RebaseGroup *> &groupOf) {
  SmallVector<QuantizedOperandRanges, 2> ranges;
  for (unsigned index : {0, 1}) {
    DequantizeAffineOp op = candidate.getDequantize(index);
    std::optional<QuantizedOperandRanges> operandRanges =
        candidate.rebase[index]
            ? getRebasedOperandRanges(op, groupOf.lookup(op)->offset)
            : getOperandRanges(op);
    if (!operandRanges) {
      return false;
    }
    ranges.push_back(*operandRanges);
  }
  return getMaxReductionExtent(ranges[0], ranges[1]) >=
         candidate.contraction.reductionExtent;
}

/// Returns `offset` in the integer type `type`, wrapped to its width.
static IntegerAttr getOffsetAttr(IntegerType type, int64_t offset) {
  return IntegerAttr::get(type, APInt(type.getWidth(), offset,
                                      /*isSigned=*/true,
                                      /*implicitTrunc=*/true));
}

/// Returns a constant of `type` holding `values` plus `offset`. A single value
/// fills a tensor.
static Value buildOffsetConstant(OpBuilder &builder, Location loc, Type type,
                                 ArrayRef<APSInt> values, int64_t offset) {
  auto elementType = cast<IntegerType>(getElementTypeOrSelf(type));
  APInt delta = getOffsetAttr(elementType, offset).getValue();
  SmallVector<APInt> shifted = llvm::map_to_vector(
      values, [&](const APSInt &value) -> APInt { return value + delta; });
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return arith::ConstantOp::create(
        builder, loc, DenseElementsAttr::get(tensorType, shifted));
  }
  return arith::ConstantOp::create(
      builder, loc, IntegerAttr::get(elementType, shifted.front()));
}

/// Returns `value` plus `offset` in the element type of `value`, elementwise
/// for tensors.
static Value buildOffsetValue(OpBuilder &builder, Location loc, Value value,
                              int64_t offset) {
  Type type = value.getType();
  auto elementType = cast<IntegerType>(getElementTypeOrSelf(type));
  Value deltaValue = arith::ConstantOp::create(
      builder, loc, getOffsetAttr(elementType, offset));
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType) {
    return arith::AddIOp::create(builder, loc, value, deltaValue);
  }
  Value init = tensor::EmptyOp::create(
      builder, loc, tensor::getMixedSizes(builder, loc, value), elementType);
  AffineMap identity = builder.getMultiDimIdentityMap(tensorType.getRank());
  return linalg::GenericOp::create(
             builder, loc, tensorType, ValueRange{value}, ValueRange{init},
             SmallVector<AffineMap>{identity, identity},
             SmallVector<utils::IteratorType>(tensorType.getRank(),
                                              utils::IteratorType::parallel),
             [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
               Value sum = arith::AddIOp::create(nested, nestedLoc, args[0],
                                                 deltaValue);
               linalg::YieldOp::create(nested, nestedLoc, sum);
             })
      .getResult(0);
}

/// Returns `value` moved by `offset`, built once per value and offset right
/// after `value` is defined.
static Value getOrCreateShiftedValue(RewriterBase &rewriter, Value value,
                                     int64_t offset,
                                     ShiftedValues &shiftedValues) {
  Value &shifted = shiftedValues[{value, offset}];
  if (!shifted) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfterValue(value);
    shifted = buildOffsetValue(rewriter, value.getLoc(), value, offset);
  }
  return shifted;
}

/// Moves the zero point of `op` by `offset` and marks it with the new
/// signedness. A zero point read from a constant or an immutable global that
/// only loads use stays a constant. A symmetric `op` gets a scalar zero point
/// of `offset` in its storage type `storageType`.
template <typename OpTy>
static void rebaseZeroPoint(RewriterBase &rewriter, OpTy op,
                            IntegerType storageType, int64_t offset,
                            bool isUnsigned, ShiftedValues &shiftedValues,
                            const SymbolUserMap &symbolUsers,
                            SymbolTableCollection &symbolTables) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  Value zeroPoint = op.getZeroPoint();
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (!zeroPoint) {
    // A zero point of zero moves to `offset` on the new grid. A scalar zero
    // point reads no index.
    zeroPoint = arith::ConstantOp::create(rewriter, op.getLoc(),
                                          getOffsetAttr(storageType, offset));
    maps.insert(maps.begin() + 2, AffineMap::get(maps.front().getNumDims(), 0,
                                                 {}, rewriter.getContext()));
  } else if (std::optional<SmallVector<APSInt>> values =
                 getConstantIntegerValues(zeroPoint, /*isUnsigned=*/false,
                                          symbolUsers, symbolTables)) {
    zeroPoint = buildOffsetConstant(rewriter, op.getLoc(), zeroPoint.getType(),
                                    *values, offset);
  } else {
    zeroPoint =
        getOrCreateShiftedValue(rewriter, zeroPoint, offset, shiftedValues);
  }
  rewriter.modifyOpInPlace(op, [&] {
    op.getZeroPointMutable().assign(zeroPoint);
    op.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
    op.setZpUnsigned(isUnsigned);
  });
}

/// Rebases the value of `group` and all its readers.
static void rebaseGroup(RewriterBase &rewriter, const RebaseGroup &group,
                        ShiftedValues &shiftedValues,
                        const SymbolUserMap &symbolUsers,
                        SymbolTableCollection &symbolTables,
                        SmallVectorImpl<Operation *> &rebasedOps) {
  bool isUnsigned = !group.isUnsigned;
  if (QuantizeAffineOp quantize = group.quantize) {
    rebaseZeroPoint(
        rewriter, quantize,
        cast<IntegerType>(quantize.getOutputType().getElementType()),
        group.offset, isUnsigned, shiftedValues, symbolUsers, symbolTables);
    rewriter.modifyOpInPlace(quantize, [&] {
      quantize.setQuantMin(quantize.getQuantMin() + group.offset);
      quantize.setQuantMax(quantize.getQuantMax() + group.offset);
      quantize.setStorageUnsigned(isUnsigned);
    });
    rebasedOps.push_back(quantize);
  } else {
    // Shift each constant source once, before any reshape, so that readers of
    // different shapes share the shift.
    SetVector<Value> sources;
    for (DequantizeAffineOp reader : group.readers) {
      sources.insert(getStorageSource(reader));
    }
    for (Value source : sources) {
      // Move only the users from before the shift, not its size queries.
      SmallPtrSet<Operation *, 4> users(llvm::from_range, source.getUsers());
      Value shifted = getOrCreateShiftedValue(rewriter, source, group.offset,
                                              shiftedValues);
      rewriter.replaceUsesWithIf(source, shifted, [&](OpOperand &use) {
        return users.contains(use.getOwner());
      });
    }
  }
  for (DequantizeAffineOp reader : group.readers) {
    rebaseZeroPoint(rewriter, reader,
                    cast<IntegerType>(reader.getInputType().getElementType()),
                    group.offset, isUnsigned, shiftedValues, symbolUsers,
                    symbolTables);
    rewriter.modifyOpInPlace(reader, [&] {
      if (std::optional<int64_t> quantMin = reader.getQuantMin()) {
        reader.setQuantMin(*quantMin + group.offset);
        reader.setQuantMax(*reader.getQuantMax() + group.offset);
      }
      reader.setInputUnsigned(isUnsigned);
    });
    rebasedOps.push_back(reader);
  }
}

/// Rebases quantized contraction operands to the signedness the target tiles.
struct RebaseQuantizedOperandsPass final
    : impl::RebaseQuantizedOperandsPassBase<RebaseQuantizedOperandsPass> {
  void runOnOperation() override;
};

void RebaseQuantizedOperandsPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();

  // Ask the one dialect that resolves the layouts of the targets an op runs on.
  DialectInterfaceCollection<IREE::Stream::AffinityAnalysisDialectInterface>
      affinityInterfaces(context);
  if (!llvm::hasSingleElement(affinityInterfaces)) {
    return markAllAnalysesPreserved();
  }
  const IREE::Stream::AffinityAnalysisDialectInterface &affinityInterface =
      *affinityInterfaces.begin();

  // Collect every op ConvertQDQToIntegerMath will rewrite. Without a
  // contraction there is nothing to rebase.
  SmallVector<Candidate, 0> candidates;
  moduleOp.walk([&](linalg::LinalgOp op) {
    FailureOr<QuantizedContraction> contraction = getQuantizedContraction(op);
    if (failed(contraction)) {
      return;
    }
    Candidate &candidate = candidates.emplace_back();
    candidate.op = op;
    candidate.contraction = *contraction;
    candidate.integerMaps =
        getIntegerContractionMaps(op, *contraction).indexingMaps;
    candidate.iterationSizes = op.getStaticLoopRanges();
  });
  if (llvm::none_of(candidates, [](const Candidate &candidate) {
        return linalg::isaContractionOpInterface(candidate.op);
      })) {
    return markAllAnalysesPreserved();
  }
  DenseMap<Operation *, Candidate *> candidateOf;
  for (Candidate &candidate : candidates) {
    candidateOf[candidate.op] = &candidate;
  }

  // Ask for the resolvers of every target each contraction runs on, before any
  // rewrite. A failing analysis skips the rebase.
  IREE::Stream::AffinityAnalysis affinityAnalysis(moduleOp);
  if (failed(affinityAnalysis.run())) {
    LDBG() << "affinity analysis failed";
    return markAllAnalysesPreserved();
  }
  SmallVector<IREE::Stream::AffinityAndOpPair> queries;
  for (Candidate &candidate : candidates) {
    SmallVector<IREE::Stream::AffinityAttr> affinities;
    if (!linalg::isaContractionOpInterface(candidate.op) ||
        !affinityAnalysis.tryInferExecutionAffinity(candidate.op, affinities)) {
      continue;
    }
    for (IREE::Stream::AffinityAttr affinity : affinities) {
      queries.push_back({affinity, candidate.op});
    }
  }
  if (queries.empty()) {
    return markAllAnalysesPreserved();
  }
  DenseMap<IREE::Stream::AffinityAndOpPair, SetVector<Attribute>> layoutAttrs;
  {
    // Silence the error the resolver reports when it fails.
    ScopedDiagnosticHandler silenceResolver(
        context, [](Diagnostic &) { return success(); });
    if (failed(affinityInterface.makeLayoutAttrResolver(moduleOp)(
            queries, layoutAttrs))) {
      LDBG() << "layout resolution failed";
      return markAllAnalysesPreserved();
    }
  }
  for (const IREE::Stream::AffinityAndOpPair &query : queries) {
    const SetVector<Attribute> &resolvers = layoutAttrs[query];
    candidateOf.lookup(query.second)
        ->resolvers.insert(resolvers.begin(), resolvers.end());
  }

  // Choose the operands to rebase per contraction.
  SymbolTableCollection symbolTables;
  SymbolUserMap symbolUsers(symbolTables, moduleOp);
  const auto &constExprs = getAnalysis<IREE::Util::ConstExprAnalysis>();
  for (Candidate &candidate : candidates) {
    LDBG() << candidate.resolvers.size() << " resolvers for " << candidate.op;
    candidate.rebase =
        chooseRebase(candidate, constExprs, symbolUsers, symbolTables);
  }

  // Build the group of every value a contraction wants to rebase, once, through
  // the first dequantize that reads it. A value that cannot be rebased with all
  // its readers keeps the contraction from rebasing.
  SmallVector<RebaseGroup> groups;
  DenseSet<Operation *> grouped;
  for (Candidate &candidate : candidates) {
    for (unsigned index : {0, 1}) {
      DequantizeAffineOp op = candidate.getDequantize(index);
      if (!candidate.rebase[index] || grouped.contains(op)) {
        continue;
      }
      FailureOr<RebaseGroup> group =
          buildRebaseGroup(op, candidateOf, symbolUsers, symbolTables);
      if (failed(group)) {
        candidate.rebase = {false, false};
        continue;
      }
      grouped.insert_range(group->readers);
      groups.push_back(std::move(*group));
    }
  }
  DenseMap<Operation *, RebaseGroup *> groupOf;
  for (RebaseGroup &group : groups) {
    for (DequantizeAffineOp reader : group.readers) {
      groupOf[reader] = &group;
    }
  }

  // Drop the groups whose contraction operands do not all agree and the
  // contractions that no longer fit i32, until nothing changes.
  bool changed = true;
  while (changed) {
    changed = false;
    for (RebaseGroup &group : groups) {
      if (group.isDropped || isAgreed(group)) {
        continue;
      }
      group.isDropped = true;
      changed = true;
      for (auto &operand : group.operands) {
        operand.first->rebase = {false, false};
      }
    }
    for (Candidate &candidate : candidates) {
      if ((candidate.rebase[0] || candidate.rebase[1]) &&
          !fitsAccumulator(candidate, groupOf)) {
        candidate.rebase = {false, false};
        changed = true;
      }
    }
  }

  IRRewriter rewriter(context);
  ShiftedValues shiftedValues;
  SmallVector<Operation *> rebasedOps;
  for (const RebaseGroup &group : groups) {
    if (!group.isDropped) {
      rebaseGroup(rewriter, group, shiftedValues, symbolUsers, symbolTables,
                  rebasedOps);
    }
  }
  if (rebasedOps.empty()) {
    return markAllAnalysesPreserved();
  }

  // Drop zero points that became zero, such as a u8 zero point of 128.
  RewritePatternSet patterns(context);
  QuantizeAffineOp::getCanonicalizationPatterns(patterns, context);
  DequantizeAffineOp::getCanonicalizationPatterns(patterns, context);
  if (failed(applyOpPatternsGreedily(rebasedOps, std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
