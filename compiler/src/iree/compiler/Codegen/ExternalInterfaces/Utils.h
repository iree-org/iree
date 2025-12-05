// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_UTILS_H_
#define IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_UTILS_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/Utils/EncodingUtils.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"

namespace mlir::iree_compiler::IREE {

static const char kEncodingInfoAttrName[] = "encoding_info";

// This class is the base class for the external model of different packed
// encoding layout attributes. It provides a public method, `getEncodingInfo` to
// reduce the duplicated implementations before. To inherit it, it requires the
// derived class to implement the `getConfiguration` method and the
// `getEncodingInfoImpl` method.
template <typename EncodingPackedLayoutMaterializerAttr,
          typename EncodingLayoutAttr>
struct PackedLayoutMaterializerAttrExternalModelBase
    : public IREE::Codegen::PackedLayoutMaterializerAttr::ExternalModel<
          EncodingPackedLayoutMaterializerAttr, EncodingLayoutAttr> {
public:
  IREE::Codegen::MaterializeEncodingInfo
  getEncodingInfo(Attribute attr, RankedTensorType type) const {
    const EncodingPackedLayoutMaterializerAttr *impl =
        static_cast<const EncodingPackedLayoutMaterializerAttr *>(this);
    // If the layout is already resolved, use it directly.
    if (auto config = impl->getConfiguration(attr)) {
      if (auto namedAttr = config.getNamed(kEncodingInfoAttrName)) {
        std::optional<IREE::Codegen::MaterializeEncodingInfo> info =
            IREE::Codegen::deserializeEncodingInfo(
                cast<DictionaryAttr>(namedAttr->getValue()));
        assert(info && "encoding_info is invalid");
        return info.value();
      }
    }
    return impl->getEncodingInfoImpl(attr, type);
  }

  LogicalResult verifyPackedLayoutWithType(
      Attribute attr, ArrayRef<int64_t> shape, Type elementType,
      function_ref<InFlightDiagnostic()> emitError) const {
    const EncodingPackedLayoutMaterializerAttr *impl =
        static_cast<const EncodingPackedLayoutMaterializerAttr *>(this);
    DictionaryAttr config = impl->getConfiguration(attr);
    if (!config) {
      return success();
    }

    auto encodingInfoAttr = config.getNamed(kEncodingInfoAttrName);
    if (!encodingInfoAttr) {
      return success();
    }

    std::optional<IREE::Codegen::MaterializeEncodingInfo> info =
        IREE::Codegen::deserializeEncodingInfo(
            cast<DictionaryAttr>(encodingInfoAttr->getValue()));
    if (!info) {
      return emitError() << "invalid encoding_info in configuration";
    }

    // Identity layouts are always valid.
    if (IREE::Codegen::isIdentityLayout(info.value())) {
      return success();
    }

    if (info->innerDimsPos.size() != info->innerTileSizes.size()) {
      return emitError() << "innerDimsPos size (" << info->innerDimsPos.size()
                         << ") does not match innerTileSizes size ("
                         << info->innerTileSizes.size() << ")";
    }

    int64_t rank = shape.size();

    // Utility to check that indices are valid (in bounds) and unique.
    auto verifyIndices = [&](ArrayRef<int64_t> indices,
                             StringRef fieldName) -> LogicalResult {
      llvm::SmallDenseSet<int64_t> seenIndices;
      for (int64_t pos : indices) {
        if (pos < 0 || pos >= rank) {
          return emitError() << fieldName << " index " << pos
                             << " is out of bounds for tensor rank " << rank;
        }
        if (!seenIndices.insert(pos).second) {
          return emitError()
                 << fieldName << " contains duplicate index " << pos;
        }
      }
      return success();
    };

    if (failed(verifyIndices(info->innerDimsPos, "innerDimsPos"))) {
      return failure();
    }
    if (failed(verifyIndices(info->outerDimsPerm, "outerDimsPerm"))) {
      return failure();
    }

    return success();
  }
};

template <typename EncodingLayoutMaterializerAttr, typename EncodingLayoutAttr>
struct EncodingLayoutMaterializerAttrExternalModelBase
    : public IREE::Encoding::LayoutMaterializerAttr::ExternalModel<
          EncodingLayoutMaterializerAttr, EncodingLayoutAttr> {
public:
  IREE::Codegen::MaterializeEncodingInfo
  getEncodingInfo(EncodingLayoutAttr layoutAttr, RankedTensorType type) const {
    return getEncodingInfoFromLayout(
        type, cast<IREE::Encoding::LayoutMaterializerAttr>(layoutAttr));
  }

  Type convertType(Attribute attr, Type type) const {
    EncodingLayoutAttr layoutAttr = cast<EncodingLayoutAttr>(attr);
    return TypeSwitch<Type, Type>(type)
        .Case([&](RankedTensorType type) {
          // For a given tensor type with an encoding, return the materialized
          // type to use for it. If no encoding is set, then return the tensor
          // type itself.
          IREE::Codegen::MaterializeEncodingInfo encodingInfo =
              getEncodingInfo(layoutAttr, type);
          if (IREE::Codegen::isIdentityLayout(encodingInfo)) {
            return type.dropEncoding();
          }
          // Mark scalable tiles as dynamic sizes for the shape inference. Note,
          // scalable tiles that are represented with static inner tile sizes.
          SmallVector<int64_t> innerTileSizesVector =
              llvm::to_vector(encodingInfo.innerTileSizes);
          if (encodingInfo.scalableTiles.has_value()) {
            for (auto [index, value] :
                 llvm::enumerate(encodingInfo.scalableTiles.value())) {
              if (value) {
                innerTileSizesVector[index] = ShapedType::kDynamic;
              }
            }
          }
          auto packedType =
              cast<RankedTensorType>(linalg::PackOp::inferPackedType(
                  type, innerTileSizesVector, encodingInfo.innerDimsPos,
                  encodingInfo.outerDimsPerm));

          // There is no swizzle, we are already done. Typically the case on
          // CPU.
          if (!encodingInfo.swizzle) {
            return packedType;
          }

          // There is a swizzle, we need to handle it. Typically the case on
          // GPU.
          auto swizzle = *encodingInfo.swizzle;
          SmallVector<int64_t> newShape(packedType.getShape().drop_back(
              encodingInfo.innerTileSizes.size()));
          SmallVector<int64_t> swizzledTileShape =
              IREE::Codegen::getExpandedTileShape(swizzle.expandShape);
          applyPermutationToVector(swizzledTileShape, swizzle.permutation);
          newShape.append(swizzledTileShape);
          return RankedTensorType::get(newShape, packedType.getElementType());
        })
        .Case([&](IREE::TensorExt::DispatchTensorType dispatchTensorType) {
          Type boundType = dispatchTensorType.getBoundType();
          Type convertedBoundType = convertType(attr, boundType);
          if (convertedBoundType == boundType) {
            return dispatchTensorType;
          }
          return IREE::TensorExt::DispatchTensorType::get(
              dispatchTensorType.getAccess(), convertedBoundType);
        })
        .Default([&](auto concreteType) { return concreteType; });
  }

  LogicalResult getOffsetsSizesStrides(
      Attribute attr, OpBuilder &builder, Location loc,
      IREE::TensorExt::DispatchTensorType type, ValueRange dynamicDims,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      ArrayRef<OpFoldResult> strides, SmallVectorImpl<OpFoldResult> &newOffsets,
      SmallVectorImpl<OpFoldResult> &newSizes,
      SmallVectorImpl<OpFoldResult> &newStrides) const {
    auto boundType = dyn_cast<RankedTensorType>(type.getBoundType());
    if (!boundType || !boundType.getEncoding()) {
      return failure();
    }

    // Only handle cases where the slice spans the whole
    // `!iree_tensor_ext.dispatch.tensor` type.
    // TODO(jornt): Enable partial slices.
    if (!type.doesSliceSpanWholeTensor(dynamicDims, offsets, sizes, strides)) {
      return failure();
    }

    auto layoutAttr = cast<IREE::Encoding::LayoutMaterializerAttr>(attr);
    IREE::Codegen::MaterializeEncodingInfo encodingInfo =
        getEncodingInfoFromLayout(boundType, layoutAttr);
    newSizes = getMixedValues(boundType.getShape(), dynamicDims, builder);
    FailureOr<SmallVector<OpFoldResult>> convertedMixedSizes =
        getPackedDimsForDispatchTensorImpl(builder, loc, type, dynamicDims,
                                           layoutAttr, encodingInfo);
    if (succeeded(convertedMixedSizes)) {
      newSizes = convertedMixedSizes.value();
    }
    newOffsets.resize(newSizes.size(), builder.getIndexAttr(0));
    newStrides.resize(newSizes.size(), builder.getIndexAttr(1));
    return success();
  }
};

/// Calculates the storage size in bytes for the given `type` with a packed
/// layout encoding `attr`. Requirement: `attr` must implement
/// IREE::Codegen::PackedLayoutMaterializerAttr.
Value calculatePackedStorageSizeInBytesImpl(Attribute attr, Location loc,
                                            OpBuilder &builder,
                                            RankedTensorType type,
                                            ValueRange dynamicDims);

/// Returns a dictionary attribute that contains the materialized encoding info,
/// i.e., serialized MaterializeEncodingInfo struct. The EncodingAttr attribute
/// is attached to the dictionary, if it is present in `type` and
/// `addEncodingAttr` is true.
/// TODO(hanchung): only attach needed information to the configuration. The
/// `addEncodingAttr` is mainly for VMVX ukernel path because the ukernel ops
/// lowering requires all the information. There are no direct mappings from
/// layouts to ukernels.
/// Requirement: `attr` must implement
/// IREE::Codegen::PackedLayoutMaterializerAttr.
DictionaryAttr getPackedLayoutImpl(Attribute attr, RankedTensorType type,
                                   bool addEncodingAttr = false);

/// Appends the NamedAttribute into `config` if there is a `name` NamedAttribute
/// in the `dictAttr`.
void storeNamedAttrIfPresent(SmallVectorImpl<NamedAttribute> &config,
                             DictionaryAttr dictAttr, StringRef name);

/// Returns a `linalg.fill` operation with provided operands, which are assumed
/// to have types without encodings.
Operation *lowerFillOpWithResolvedLayouts(OpBuilder &builder,
                                          linalg::FillOp fillOp,
                                          TypeRange convertedResTypes,
                                          ValueRange convertedOperands);

/// Converts a linalg::GenericOp with encoded inputs into the packed domain,
/// with an optional swizzle expansion and permutation if applicable. The
/// `genericOp` must have all parallel iterator types and a single output with
/// an identity indexing map.
Operation *lowerGenericOpWithResolvedLayouts(
    OpBuilder &builder, linalg::GenericOp genericOp,
    TypeRange convertedResTypes, ValueRange convertedOperands,
    IREE::Encoding::LayoutMaterializerAttr layoutAttr);

} // namespace mlir::iree_compiler::IREE

#endif // IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_UTILSS_H_
