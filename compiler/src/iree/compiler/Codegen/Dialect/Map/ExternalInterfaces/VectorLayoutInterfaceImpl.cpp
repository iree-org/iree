// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Map/ExternalInterfaces/VectorLayoutInterfaceImpl.h"

#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapAttrs.h"
#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapDialect.h"
#include "iree/compiler/Codegen/Dialect/Map/IR/IntTuple.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtInterfaces.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler::IREE::Map {

using VectorExt::VectorLayoutInterface;

namespace {

struct PackLayoutModel final
    : public VectorLayoutInterface::ExternalModel<PackLayoutModel,
                                                  PackLayoutAttr> {
  int64_t getRank(Attribute attr) const {
    return cast<PackLayoutAttr>(attr).getRank();
  }

  SmallVector<int64_t> getUndistributedShape(Attribute attr) const {
    auto layout = cast<PackLayoutAttr>(attr);
    SmallVector<int64_t> shape;
    int32_t rank = layout.getRank();
    shape.reserve(rank);
    for (int32_t i = 0; i < rank; ++i) {
      shape.push_back(getSize(layout.getShapeMode(i)));
    }
    return shape;
  }

  SmallVector<int64_t> getDistributedShape(Attribute attr) const {
    auto layout = cast<PackLayoutAttr>(attr);
    SmallVector<int64_t> shape;
    int32_t rank = layout.getRank();
    shape.reserve(rank);
    for (int32_t i = 0; i < rank; ++i) {
      // Stride-0 leaves represent per-thread value dimensions (broadcast).
      SmallVector<LeafInfo> valLeaves =
          filterLeafInfos(layout.getShapeMode(i), layout.getStrideMode(i),
                          [](const LeafInfo &l) { return l.stride == 0; });
      if (valLeaves.empty()) {
        shape.push_back(1);
      } else {
        for (auto &leaf : valLeaves) {
          shape.push_back(static_cast<int64_t>(leaf.size));
        }
      }
    }
    return shape;
  }

  LogicalResult isValidLayout(Attribute attr, ShapedType shapeTy,
                              Location loc) const {
    auto layout = cast<PackLayoutAttr>(attr);
    int64_t rank = layout.getRank();
    ArrayRef<int64_t> vecShape = shapeTy.getShape();
    if (static_cast<int64_t>(vecShape.size()) != rank) {
      return emitError(loc, "Rank of vector (")
             << vecShape.size() << ") does not match rank of layout (" << rank
             << ").";
    }
    SmallVector<int64_t> expected = getUndistributedShape(attr);
    for (int64_t i = 0; i < rank; ++i) {
      if (ShapedType::isStatic(vecShape[i]) && expected[i] != vecShape[i]) {
        return emitError(loc, "Vector shape mismatch at dim ")
               << i << ": expected " << expected[i] << ", got " << vecShape[i];
      }
    }
    return success();
  }

  VectorLayoutInterface permute(Attribute attr, ArrayRef<int64_t> perm) const {
    return VectorLayoutInterface(cast<PackLayoutAttr>(attr).permute(perm));
  }

  VectorLayoutInterface project(Attribute attr,
                                ArrayRef<bool> droppedDims) const {
    return VectorLayoutInterface(
        cast<PackLayoutAttr>(attr).project(droppedDims));
  }

  VectorLayoutInterface apply(Attribute attr, AffineMap map) const {
    auto layout = cast<PackLayoutAttr>(attr);
    MLIRContext *ctx = attr.getContext();
    int64_t numResults = map.getNumResults();

    SmallVector<Attribute> modeShapes(numResults);
    SmallVector<Attribute> modeStrides(numResults);

    for (auto [idx, expr] : llvm::enumerate(map.getResults())) {
      if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
        int64_t pos = dim.getPosition();
        modeShapes[idx] = layout.getShapeMode(pos);
        modeStrides[idx] = layout.getStrideMode(pos);
      } else {
        // Non-dim expressions (constants, adds, etc.) lose layout info.
        modeShapes[idx] = makeLeaf(ctx, 1);
        modeStrides[idx] = makeLeaf(ctx, 0);
      }
    }

    return VectorLayoutInterface(PackLayoutAttr::get(
        ctx, makeTuple(ctx, modeShapes), makeTuple(ctx, modeStrides)));
  }

  VectorLayoutInterface reshape(Attribute attr,
                                ArrayRef<int64_t> newShape) const {
    auto layout = cast<PackLayoutAttr>(attr);
    auto newShapeId =
        PackMapAttr::makeIdentity(layout.getMap().getContext(), newShape);
    return VectorLayoutInterface(PackLayoutAttr::get(
        attr.getContext(), layout.getMap().compose(newShapeId)));
  }

  bool
  needsSharedMemoryForConversion(Attribute attr,
                                 VectorLayoutInterface targetLayout) const {
    auto targetLayoutAttr = dyn_cast_if_present<PackLayoutAttr>(targetLayout);
    if (!targetLayoutAttr) {
      return true;
    }
    auto srcLayoutAttr = cast<PackLayoutAttr>(attr);
    return srcLayoutAttr.coalesce() != targetLayoutAttr.coalesce();
  }

  static VectorLayoutInterface
  getRecombinedLayout(ArrayRef<VectorLayoutInterface> layouts,
                      ArrayRef<AffineMap> maps, AffineMap resultMap) {
    if (!llvm::all_of(layouts, llvm::IsaPred<PackLayoutAttr>)) {
      return VectorLayoutInterface();
    }
    MLIRContext *ctx = resultMap.getContext();

    SmallVector<PackLayoutAttr> packLayouts;
    llvm::transform(layouts, std::back_inserter(packLayouts),
                    [](VectorLayoutInterface layout) {
                      return cast<PackLayoutAttr>(layout);
                    });

    int64_t resRank = resultMap.getNumResults();

    // Null Attribute serves as "not yet assigned" sentinel.
    Attribute unset;
    SmallVector<Attribute> modeShapes(resRank, unset);
    SmallVector<Attribute> modeStrides(resRank, unset);

    for (auto [layout, indexingMap] : llvm::zip(packLayouts, maps)) {
      for (int64_t resultIdx = 0;
           resultIdx < static_cast<int64_t>(indexingMap.getNumResults());
           ++resultIdx) {
        auto dimExpr =
            dyn_cast<AffineDimExpr>(indexingMap.getResult(resultIdx));
        if (!dimExpr) {
          continue;
        }
        int64_t iterPos = dimExpr.getPosition();
        auto maybeResPos =
            resultMap.getResultPosition(getAffineDimExpr(iterPos, ctx));
        if (!maybeResPos.has_value()) {
          continue;
        }
        int64_t resPos = maybeResPos.value();

        Attribute ms = layout.getShapeMode(resultIdx);
        Attribute md = layout.getStrideMode(resultIdx);

        if (modeShapes[resPos] && modeShapes[resPos] != ms) {
          return VectorLayoutInterface();
        }
        modeShapes[resPos] = ms;
        modeStrides[resPos] = md;
      }
    }

    for (int64_t i = 0; i < resRank; ++i) {
      if (!modeShapes[i]) {
        modeShapes[i] = makeLeaf(ctx, 1);
        modeStrides[i] = makeLeaf(ctx, 0);
      }
    }

    return VectorLayoutInterface(PackLayoutAttr::get(
        ctx, makeTuple(ctx, modeShapes), makeTuple(ctx, modeStrides)));
  }
};

} // namespace

void registerVectorLayoutInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, IREEMapDialect *dialect) {
    PackLayoutAttr::attachInterface<PackLayoutModel>(*context);
  });
}

} // namespace mlir::iree_compiler::IREE::Map
