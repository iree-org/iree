// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef IREE_GLOBALOPTIMIZATION_DATALAYOUTUTILS_H_
#define IREE_GLOBALOPTIMIZATION_DATALAYOUTUTILS_H_

#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
class Location;
class OpBuilder;
class Operation;
class RewriterBase;
class Value;
} // namespace mlir

namespace mlir::iree_compiler::GlobalOptimization {

/// UNINITIALIZED: This node has not been initialized.
/// INTERMEDIATE: It is possible to propagate a layout through this node.
/// BARRIER: It is not possible to propagate a layout through this node.
enum class DataLayoutNodeType {
  UNINITIALIZED,
  INTERMEDIATE,
  BARRIER,
};

/// TODO: Abstractify DataLayoutTransformation to decouple from specific types
/// of transformations.
class DataLayoutTransformation {
public:
  DataLayoutTransformation(ShapedType orig, ShapedType transformed)
      : originalType(orig), transformedType(transformed){};
  DataLayoutTransformation(ShapedType orig) : originalType(orig){};
  DataLayoutTransformation(){};

  ShapedType getOriginalType() const { return originalType; };
  ShapedType getTransformedType() const { return transformedType; };
  SmallVector<int64_t> getInnerDimsPos() const { return innerDimsPos; };
  SmallVector<int64_t> getInnerTileSizes() const { return innerTileSizes; };
  SmallVector<int64_t> getOuterDimsPerm() const { return outerDimsPerm; };
  std::optional<TypedAttr> getConstantPadValue() const {
    return constantPadValue;
  };
  SmallVector<int64_t> getCorrespondingTransformedIndices() const {
    return correspondingTransformedIndices;
  };
  void setOriginalType(ShapedType type) { originalType = type; };
  void setTransformedType(ShapedType type) { transformedType = type; };
  void setInnerDimsPos(SmallVector<int64_t> pos) { innerDimsPos = pos; };
  void setInnerTileSizes(SmallVector<int64_t> tiles) {
    innerTileSizes = tiles;
  };
  void setOuterDimsPerm(SmallVector<int64_t> perm) { outerDimsPerm = perm; };
  void setConstantPadValue(std::optional<TypedAttr> attr) {
    constantPadValue = attr;
  };
  void setCorrespondingTransformedIndices(SmallVector<int64_t> inds) {
    correspondingTransformedIndices = inds;
  };

  /// Get the transformed layout at the `newValue`, given the `currentValue`
  /// with `this` layout. Return true for a successful transformation, and
  /// return false if the transformation is not supported.
  bool transformLayout(Value currentValue, Value newValue);

  /// Combine the information from this transform with another transform, and
  /// return whether or not information was gained.
  bool combineLayout(DataLayoutTransformation &other);

  /// Return whether this transform is valid. For now, only check that there is
  /// an originalType and a transformedType.
  const bool hasValidTransform();

  /// Return true if the transformed indices in this transformation overlap with
  /// the transformed indices of the other transformation.
  bool isIntersecting(DataLayoutTransformation &other);

  /// Return true if this transform is an identity transformation.
  bool isIdentity();

  /// Create an ArrayAttr containing transformation information for debugging.
  ArrayAttr makeTransformArrayAttr(MLIRContext *ctx);

  /// Return a new identity transformation.
  static DataLayoutTransformation *getIdentityTransformation(ShapedType type) {
    auto *tf = new DataLayoutTransformation(type, type);
    tf->setCorrespondingTransformedIndices(
        llvm::to_vector(llvm::seq<int64_t>(0, type.getRank())));
    return tf;
  }

private:
  /// Transform the layout as if propagating through an operation, from the
  /// `currentValue` to `newValue`, and return the new layout.
  bool transform(Operation *op, Value currentValue, Value newValue);

  /// The original type corresponding to the source of this layout.
  ShapedType originalType;
  /// The type of the layout source after the transformation is applied.
  ShapedType transformedType;
  /// Transformation metadate from `originalType`->`transformedType, represented
  /// as pack metadata (innerDimsPos, innerTileSizes, outerDimsPerm) for now.
  SmallVector<int64_t> innerDimsPos;
  SmallVector<int64_t> innerTileSizes;
  SmallVector<int64_t> outerDimsPerm;
  /// Optional padding value for packed layouts.
  std::optional<TypedAttr> constantPadValue = std::nullopt;
  /// Indices in the `originalType` corresponding to each index of the value
  /// associated with this DataLayoutTransformation.
  SmallVector<int64_t> correspondingTransformedIndices;
};

//===----------------------------------------------------------------------===//
// Analysis helpers
//===----------------------------------------------------------------------===//

/// Analyze the producer and users of this value, and return the node type for
/// the given value.
DataLayoutNodeType getNodeTypeForValue(Value value);

/// If the node is a terminal node (e.g., the source of a layout, like a value
/// next to a global load or store), then return the layoutIDs corresponding to
/// that terminal node.
SmallVector<StringRef> getTerminalNodeIDs(Value value);

//===----------------------------------------------------------------------===//
// Pass helpers
//===----------------------------------------------------------------------===//

/// Rewrite a Util::GlobalOp into the new layout indicated by the given
/// DataLayoutTransformation, and rewrite all GlobalLoadOps or GlobalStoreOps
/// of the `global`, passed in `edgeNodes` as the results or inputs to the
/// load/store ops. Also, create an initializer to fill the new packed global
/// with the padding value of the pack in the `transform`.
LogicalResult
transformGlobalsToNewLayout(IRRewriter &rewriter, SmallVector<Value> edgeNodes,
                            DataLayoutTransformation *transform,
                            const Explorer::GlobalInfo *globalInfo,
                            SymbolTable moduleSymbols);

/// Given metadata for extract_slice or insert_slice ops and its rank-reducing
/// mask, compute the new metadata for a slice in the packed domain defined by
/// the passed `mixedTiles`, `innerDimsPos`, and `outerDimsPerm`. This will
/// overwrite the passed slice metadata with the packed domain metadata.
LogicalResult
getPackedSliceMetadata(PatternRewriter &rewriter, Location loc,
                       SmallVector<OpFoldResult> mixedTiles,
                       SmallVector<int64_t> innerDimsPos,
                       SmallVector<int64_t> outerDimsPerm,
                       llvm::SmallDenseSet<unsigned> rankReducingMask,
                       SmallVector<OpFoldResult> &mixedOffsets,
                       SmallVector<OpFoldResult> &mixedSizes,
                       SmallVector<OpFoldResult> &mixedStrides);

/// Given a slice of a tensor, the corresponding slice sizes, and the metadata
/// for rank reduction and a pack of the full tensor, return the rank-reduced,
/// packed type of the slice. Unit dims are inserted for packed rank-reduced
/// dimensions that become inner dimensions after the pack.
/// For example, the packed slice type for the following insert_slice would be
/// `tensor<32x1x64x1x2xf32>`, corresponding to the packed `%0`:
///
/// %2 = tensor.insert_slice %0 into %1[32, 0, 0] [1, 32, 128] [1, 1, 1]
///     : tensor<32x128xf32> into tensor<4095x32x128xf32>
/// %4 = tensor.pack %0 outer_dims_perm = [1, 0, 2] inner_dims_pos = [0, 2]
///     inner_tiles = [16, 2] into %3 : tensor<4095x32x128xf32>
///     -> tensor<32x256x64x16x2xf32>
RankedTensorType getPackedSliceType(
    RankedTensorType packedSourceType, SmallVector<OpFoldResult> sliceSizes,
    llvm::SmallDenseSet<unsigned> rankReductionMask,
    ArrayRef<int64_t> outerDimsPerm, ArrayRef<int64_t> innerDimsPos);

///
FailureOr<Value>
packSliceOfTensor(PatternRewriter &rewriter, Value slice,
                  SmallVector<OpFoldResult> sliceSizes,
                  llvm::SmallDenseSet<unsigned> rankReductionMask,
                  tensor::PackOp packOp);

/// Given a packed slice of a tensor, some slicing and packing metadata, and
/// the original type of the slice before packing, unpack the packed slice, and
/// return the unpacked slice. The unpacked slice should be the same type as
/// `originalSliceType`.
FailureOr<Value>
unPackSliceOfTensor(PatternRewriter &rewriter, Value packedSlice,
                    SmallVector<OpFoldResult> sliceInnerTiles,
                    llvm::SmallDenseSet<unsigned> rankReductionMask,
                    tensor::UnPackOp unpackOp,
                    SmallVector<OpFoldResult> unpackedSliceSizes);

//===----------------------------------------------------------------------===//
// Attribute helpers
//===----------------------------------------------------------------------===//

/// Padding values can get in the way of unpack(pack(x)) foldings, so this sets
/// a `__foldable_pack_unpack__` attribute, which indicates that the op came
/// from a given data layout, and can be folded with the inverse of the op as
/// long as it also has the same attribute.
void setFoldablePackUnPackAttribute(Operation *op);

/// Return whether the op has the `__foldable_pack_unpack__` attribute.
bool hasFoldablePackUnPackAttribute(Operation *op);

/// Get the DataLayoutNodeType of an annotated op.
std::optional<DataLayoutNodeType> getNodeTypeFromAttr(Operation *op);

/// Set the `__node_type__` attribute for the op.
void setNodeTypeAttribute(Operation *op, DataLayoutNodeType nodeType);

/// Annotate an op for debugging with the layoutID and original + transformed
/// type for a transformation.
void setDataLayoutTransformationAttributes(Operation *op,
                                           DataLayoutTransformation *transform,
                                           StringRef transformID);

} // namespace mlir::iree_compiler::GlobalOptimization

#endif // IREE_GLOBALOPTIMIZATION_DATALAYOUTUTILS_H_
