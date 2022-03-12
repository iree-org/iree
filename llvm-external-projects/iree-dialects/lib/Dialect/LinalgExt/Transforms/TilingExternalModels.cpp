//===- TilingExternalModels.cpp - External models for TilingInterface -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "linalg-ext-tiling"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::iree_compiler::IREE::LinalgExt;

static Value getAsValue(OpBuilder &b, Location loc, OpFoldResult ofr) {
  if (auto v = ofr.dyn_cast<Value>()) return v;
  return b.create<arith::ConstantIndexOp>(
      loc, ofr.get<Attribute>().cast<IntegerAttr>().getInt());
}
static SmallVector<Value> getAsValues(OpBuilder &b, Location loc,
                                      ArrayRef<OpFoldResult> ofrs) {
  SmallVector<Value> vals;
  vals.reserve(ofrs.size());
  for (auto ofr : ofrs) vals.push_back(getAsValue(b, loc, ofr));
  return vals;
}

static SmallVector<Value, 4> makeTiledInputShapes(OpBuilder &b, Location loc,
                                                  LinalgOp linalgOp,
                                                  ArrayRef<Value> valuesToTile,
                                                  ArrayRef<Value> ivsRef,
                                                  ArrayRef<Value> tileSizesRef,
                                                  ArrayRef<Value> sizeBounds) {
  assert(static_cast<int64_t>(valuesToTile.size()) == linalgOp.getNumInputs() &&
         "expected one value to tile for every operand");

  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> tileSizes{tileSizesRef.begin(), tileSizesRef.end()};
  tileSizes.append(sizeBounds.size() - tileSizes.size(), zero);

  // Construct (potentially temporary) mins and maxes on which to apply maps
  // that define tile subshapes.
  SmallVector<Value> lbs = computeTileOffsets(b, loc, ivsRef, tileSizes);
  SmallVector<Value> subShapeSizes =
      computeTileSizes(b, loc, ivsRef, tileSizes, sizeBounds);

  SmallVector<Value, 4> tiledShapes;
  tiledShapes.reserve(valuesToTile.size());
  for (OpOperand *opOperand : linalgOp.getInputOperands()) {
    Value shapedOp = valuesToTile[opOperand->getOperandNumber()];
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShapes: for operand " << shapedOp);
    AffineMap map = linalgOp.getTiedIndexingMap(opOperand);
    LLVM_DEBUG(llvm::dbgs() << ": tiled: figure out subshape...\n");
    tiledShapes.push_back(makeTiledShape(b, loc, shapedOp, tileSizes, map, lbs,
                                         sizeBounds, subShapeSizes));
  }

  return tiledShapes;
}

namespace {

/// External model implementation of TilingInterface for LinalgOps. This is
/// templated on the actual Linalg named op for now since the registration of
/// the external model requires the original operation.
template <typename LinalgOpTy>
struct LinalgOpTilingInterface
    : public TilingInterface::ExternalModel<LinalgOpTilingInterface<LinalgOpTy>,
                                            LinalgOpTy> {
  SmallVector<Value> getDestinationOperands(Operation *op, OpBuilder &b) const {
    LinalgOp linalgOp = cast<LinalgOp>(op);
    return linalgOp.getOutputOperands();
  }

  SmallVector<StringRef> getLoopIteratorTypes(Operation *op) const {
    LinalgOp linalgOp = cast<LinalgOp>(op);
    SmallVector<StringRef> iteratorTypes;
    iteratorTypes.reserve(linalgOp.iterator_types().size());
    for (Attribute iteratorAttr : linalgOp.iterator_types()) {
      iteratorTypes.push_back(iteratorAttr.cast<StringAttr>().getValue());
    }
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    LinalgOp linalgOp = cast<LinalgOp>(op);
    return linalgOp.createLoopRanges(b, op->getLoc());
  }

  SmallVector<Operation *> getTiledImplementation(
      Operation *op, OpBuilder &b, ValueRange tiledDest,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      bool tileDestOperands) const {
    LinalgOp linalgOp = cast<LinalgOp>(op);
    if (op->getNumResults() != 1) {
      // TODO: Need a failure message here, but `notifyMatchFailure` is only a
      // method on `PatternRewriter`.
      return {};
    }
    Location loc = op->getLoc();
    AffineMap shapeSizesToLoopsMap = linalgOp.getShapesToLoopsMap();
    auto allShapeSizes = linalgOp.createFlatListOfOperandDims(b, loc);
    if (!shapeSizesToLoopsMap) return {};

    OpOperand *outOperand = linalgOp.getOutputOperand(0);
    AffineMap indexingMap = linalgOp.getTiedIndexingMap(outOperand);
    if (!indexingMap.isProjectedPermutation()) return {};

    SmallVector<Value> offsetsVals = getAsValues(b, loc, offsets);
    SmallVector<Value> sizeVals = getAsValues(b, loc, sizes);
    SmallVector<Value> sizeBounds =
        applyMapToValues(b, loc, shapeSizesToLoopsMap, allShapeSizes);

    // The offsets and sizes form the slice operation only give you the tile
    // size of the output. Use that compute the tile sizes and offsets of the
    // loops. For loops not used to access the output, set the tile sizes to
    // loop bounds and set the offset to 0.
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> tileOffsets(sizeBounds.size(), zero);
    SmallVector<Value> tileSizes = sizeBounds;
    for (auto result : enumerate(indexingMap.getResults())) {
      unsigned position = result.value().cast<AffineDimExpr>().getPosition();
      tileOffsets[position] = offsetsVals[result.index()];
      tileSizes[position] = sizeVals[result.index()];
    }

    SmallVector<Value> valuesToTile = linalgOp.getInputOperands();
    SmallVector<Value> tiledOperands;
    if (tileDestOperands) {
      // Append the outputs then tile both the inputs and outputs.
      valuesToTile.append(tiledDest.begin(), tiledDest.end());
      tiledOperands = makeTiledShapes(b, loc, linalgOp, valuesToTile,
                                      tileOffsets, tileSizes, sizeBounds);
    } else {
      // Only tile the inputs, then apped the outputs.
      int64_t dim = offsets.size();
      ArrayRef<Value> tileOffsetsRef{tileOffsets.begin(), tileOffsets.end()};
      ArrayRef<Value> tileSizesRef{tileSizes.begin(), tileSizes.end()};
      tiledOperands = makeTiledInputShapes(
          b, loc, linalgOp, valuesToTile, tileOffsetsRef.take_front(dim + 1),
          tileSizesRef.take_front(dim + 1), sizeBounds);
      tiledOperands.append(tiledDest.begin(), tiledDest.end());
    }
    return {linalgOp.clone(b, loc, tiledDest.getTypes(), tiledOperands)};
  }
};
}  // namespace

template <typename OpType>
void registerOne(DialectRegistry &registry) {
  registry.addOpInterface<OpType, LinalgOpTilingInterface<OpType>>();
}

/// Variadic helper function.
template <typename... OpTypes>
void registerAll(DialectRegistry &registry) {
  // FIXME: In c++17 this can be simplified by using 'fold expressions'.
  (void)std::initializer_list<int>{0, (registerOne<OpTypes>(registry), 0)...};
}

#define GET_OP_LIST

void mlir::iree_compiler::IREE::LinalgExt::
    registerTilingInterfaceExternalModels(DialectRegistry &registry) {
  registerOne<linalg::GenericOp>(registry);
  registerAll<
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
      >(registry);
}
