// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/DataLayoutUtils.h"
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-global-opt-propagate-data-layout"

static const char kDataLayoutNodeTypeAttr[] = "__node_type__";
static const char kFoldablePackUnPack[] = "__foldable_pack_unpack__";

namespace mlir::iree_compiler::GlobalOptimization {
using iree_compiler::IREE::Util::GlobalOp;

template <typename T>
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const llvm::SmallVectorImpl<T> &vector) {
  os << "[ ";
  for (T element : vector) {
    os << element << " ";
  }
  os << "]";

  return os;
}

template <typename T>
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const llvm::ArrayRef<T> &vector) {
  os << "[ ";
  for (T element : vector) {
    os << element << " ";
  }
  os << "]";

  return os;
}

static llvm::raw_ostream &
operator<<(llvm::raw_ostream &os, const DataLayoutTransformation &transform) {
  os << "originalType: " << transform.getOriginalType() << "\n";
  os << "transformedType: " << transform.getTransformedType() << "\n";
  os << "innerDimsPos: " << transform.getInnerDimsPos() << "\n";
  os << "innerTileSizes: " << transform.getInnerTileSizes() << "\n";
  os << "outerDimsPerm: " << transform.getOuterDimsPerm() << "\n";
  os << "constantPadValue: " << transform.getConstantPadValue() << "\n";
  os << "correspondingTransformedIndices: "
     << transform.getCorrespondingTransformedIndices() << "\n";
  return os;
}

//===----------------------------------------------------------------------===//
// DataLayoutTransformation
//===----------------------------------------------------------------------===//

/// TODO: Replace this with a more meaningful check for transform validity.
const bool DataLayoutTransformation::hasValidTransform() {
  return llvm::detail::isPresent(originalType) &&
         llvm::detail::isPresent(transformedType);
}

bool DataLayoutTransformation::transformLayout(Value currentValue,
                                               Value newValue) {
  auto curDefiningOp = currentValue.getDefiningOp();
  auto newDefiningOp = newValue.getDefiningOp();
  auto isInitOperand = [](Operation *op, Value val) -> bool {
    if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op)) {
      return llvm::any_of(dpsOp.getDpsInits(),
                          [&](Value v) { return v == val; });
    }
    return false;
  };
  auto valueConsumedByOp = [](Operation *op, Value val) -> bool {
    return op &&
           llvm::any_of(op->getOperands(), [&](Value v) { return v == val; });
  };
  // currentValue is a producer of newValue
  if (valueConsumedByOp(newDefiningOp, currentValue)) {
    // If currentValue is an init operand, then no transform needed.
    if (isInitOperand(newDefiningOp, currentValue)) {
      return true;
    }
    // Otherwise, perform transformation down through `newDefiningOp`.
    if (auto newType = dyn_cast<ShapedType>(newValue.getType())) {
      return transform(newDefiningOp, currentValue, newValue);
    }
    // currentValue is a consumer of newValue
  } else if (valueConsumedByOp(curDefiningOp, newValue)) {
    // If newValue is an init operand, then no transform needed.
    if (isInitOperand(curDefiningOp, newValue)) {
      return true;
    }
    // Otherwise, perform transformation up through `curDefiningOp`.
    if (auto newType = dyn_cast<ShapedType>(newValue.getType())) {
      return transform(curDefiningOp, currentValue, newValue);
    }
  }
  // Fail if no connecting op
  return false;
}

ArrayAttr DataLayoutTransformation::makeTransformArrayAttr(MLIRContext *ctx) {
  SmallVector<Attribute> attrs;
  attrs.push_back(TypeAttr::get(originalType));
  attrs.push_back(TypeAttr::get(transformedType));
  return ArrayAttr::get(ctx, attrs);
}

/// TODO: Replace this trivially conservative placeholder implementation.
bool DataLayoutTransformation::isIntersecting(DataLayoutTransformation other) {
  return true;
}

//===----------------------------------------------------------------------===//
// DataLayoutTransformation transform implementations
//===----------------------------------------------------------------------===//

/// Compute the transformation through a tensor::PackOp. Currently the only
/// supported case is for `currentValue` as the source and `newValue` as the
/// result of the pack. The `transform` must not have any innerTileSizes,
/// innerDimsPos, or outerDimsPerm.
/// TODO: Support cases where transform has outerDimsPerm.
bool transformThroughOperation(tensor::PackOp packOp,
                               DataLayoutTransformation &transform,
                               Value currentValue, Value newValue) {
  if (!transform.getInnerTileSizes().empty() ||
      !transform.getInnerDimsPos().empty() ||
      !transform.getOuterDimsPerm().empty()) {
    return false;
  }
  // Only supporting source->dest transforms for now.
  if (currentValue != packOp.getSource() || newValue != packOp.getResult()) {
    return false;
  }
  auto padValue = packOp.getPaddingValue();
  std::optional<TypedAttr> constPadVal = std::nullopt;
  // Padding values must be the same accross all connected transforms.
  if (padValue) {
    auto padConst = padValue.getDefiningOp<arith::ConstantOp>();
    if (!padConst) {
      return false;
    }
    auto tfConstPadVal = transform.getConstantPadValue();
    if (tfConstPadVal.has_value() &&
        tfConstPadVal.value() != padConst.getValue()) {
      return false;
    }
    constPadVal = padConst.getValue();
  }
  auto origType = dyn_cast<RankedTensorType>(transform.getOriginalType());
  if (!origType) {
    return false;
  }
  // If any correspondingTransformedIndices are `-1`, then there is not enough
  // information to do the packing transformation, so bail and wait to try again
  // after the transform holds more information.
  auto cTfInds = transform.getCorrespondingTransformedIndices();
  if (llvm::any_of(cTfInds, [](int64_t ind) { return ind == -1; })) {
    return false;
  }

  auto getPermOrIdentity = [origType](SmallVector<int64_t> perm) {
    if (perm.empty()) {
      return llvm::to_vector(llvm::seq<int64_t>(0, origType.getRank()));
    }
    return perm;
  };
  SmallVector<int64_t> currentOuterDimsPerm =
      getPermOrIdentity(transform.getOuterDimsPerm());
  auto packOuterDimsPerm =
      getPermOrIdentity(SmallVector<int64_t>(packOp.getOuterDimsPerm()));
  SmallVector<int64_t> newOuterDimsPerm(currentOuterDimsPerm);
  for (auto [idx, permIdx] : llvm::enumerate(packOuterDimsPerm)) {
    newOuterDimsPerm[cTfInds[idx]] = cTfInds[permIdx];
  }

  // The transform does not have any permutations yet, so just correct for the
  // correspondingTransformedIndices.
  SmallVector<int64_t> packInnerDimsPos(packOp.getInnerDimsPos());
  SmallVector<int64_t> newInnerDimsPos;
  auto inverseOuterDimsPos = invertPermutationVector(currentOuterDimsPerm);
  for (auto innerPos : packInnerDimsPos) {
    newInnerDimsPos.push_back(inverseOuterDimsPos[cTfInds[innerPos]]);
  }

  transform.setOuterDimsPerm(newOuterDimsPerm);
  transform.setInnerDimsPos(newInnerDimsPos);
  SmallVector<int64_t> innerTiles(packOp.getStaticInnerTiles());
  transform.setInnerTileSizes(innerTiles);

  auto newTransformedType = tensor::PackOp::inferPackedType(
      origType, transform.getInnerTileSizes(), transform.getInnerDimsPos(),
      transform.getOuterDimsPerm());
  transform.setTransformedType(newTransformedType);
  transform.setConstantPadValue(constPadVal);
  return true;
}

/// Switch case function for different op types.
bool DataLayoutTransformation::transform(Operation *op, Value currentValue,
                                         Value newValue) {
  return TypeSwitch<Operation *, bool>(op)
      .Case<tensor::PackOp>([&](tensor::PackOp packOp) {
        return transformThroughOperation(packOp, *this, currentValue, newValue);
      })
      .Default([](Operation *op) { return false; });
}

/// The only information to combine for now is correspondingTransformedIndices.
/// TODO: Check that the transformations are compatible and fail if they aren't.
bool DataLayoutTransformation::combineLayout(DataLayoutTransformation other) {
  assert(correspondingTransformedIndices.size() ==
         other.correspondingTransformedIndices.size());
  bool changed = false;
  for (auto [idx, cTfInd] : llvm::enumerate(correspondingTransformedIndices)) {
    if (cTfInd == -1 && other.correspondingTransformedIndices[idx] != -1) {
      cTfInd = other.correspondingTransformedIndices[idx];
      changed = true;
    }
  }
  return changed;
}

//===----------------------------------------------------------------------===//
// Analysis helpers
//===----------------------------------------------------------------------===//

/// Terminal nodes are just GlobalLoadOp and GlobalStoreOp for now.
SmallVector<StringRef> getTerminalNodeIDs(Value value) {
  SmallVector<StringRef> IDs;
  DataLayoutTransformation newLayout(cast<ShapedType>(value.getType()));
  if (auto loadOp = value.getDefiningOp<IREE::Util::GlobalLoadOp>()) {
    IDs.push_back(loadOp.getGlobal());
  }
  for (Operation *op : value.getUsers()) {
    if (auto storeOp = dyn_cast<IREE::Util::GlobalStoreOp>(op)) {
      IDs.push_back(storeOp.getGlobal());
    }
  }
  return IDs;
}

/// Return true if the op can assume any possible DataLayoutTransformation.
static bool isLayoutFlexibleOp(Operation *op) {
  // return isa<tensor::InsertSliceOp, tensor::ExtractSliceOp,
  return isa<IREE::Util::GlobalStoreOp, IREE::Util::GlobalLoadOp>(op);
}

/// Return true if the op defines a DataLayoutTransformation.
static bool isLayoutDefiningOp(Operation *op) {
  return isa<tensor::PackOp>(op);
}

/// Return true if a value is an intermediate node. Intermediate nodes can be
/// propagated through by some layout, and a node is intermediate if:
///  - The node has a defining op that can assume any layout and does not define
///    a layout.
///  - All of the node's users can assume any layout or define a layout.
static bool isIntermediateNode(Value value) {
  if (auto definingOp = value.getDefiningOp()) {
    if (!isLayoutFlexibleOp(definingOp) || isLayoutDefiningOp(definingOp)) {
      return false;
    }
  } else {
    return false;
  }
  for (auto op : value.getUsers()) {
    if (!isLayoutFlexibleOp(op) && !isLayoutDefiningOp(op))
      return false;
  }
  return true;
}

DataLayoutNodeType getNodeTypeForValue(Value value) {
  if (isIntermediateNode(value))
    return DataLayoutNodeType::INTERMEDIATE;
  return DataLayoutNodeType::BARRIER;
}

//===----------------------------------------------------------------------===//
// Pass helpers
//===----------------------------------------------------------------------===//

LogicalResult transformGlobalsToNewLayout(IRRewriter &rewriter,
                                          SmallVector<Value> edgeNodes,
                                          DataLayoutTransformation *transform,
                                          GlobalOp global,
                                          SymbolTable moduleSymbols) {
  // Create a new transformed GlobalOp.
  std::string newGlobalName(global.getGlobalName().str());
  newGlobalName.append(".packed");
  auto transformedType = transform->getTransformedType();
  auto originalType = transform->getOriginalType();
  std::optional<TypedAttr> transformedInitialValue = std::nullopt;
  if (auto initialValue = global.getInitialValue()) {
    if (auto uninitializedAttr =
            dyn_cast<IREE::Util::UninitializedAttr>(initialValue.value())) {
      transformedInitialValue = IREE::Util::UninitializedAttr::get(
          rewriter.getContext(), transformedType);
    } else {
      return failure();
    }
  }
  rewriter.setInsertionPoint(global);
  auto newGlobal = rewriter.create<GlobalOp>(
      global->getLoc(), StringRef(newGlobalName), global.getIsMutable(),
      transformedType, transformedInitialValue);
  newGlobal.setInliningPolicyAttr(global.getInliningPolicyAttr());
  moduleSymbols.insert(newGlobal);
  SymbolTable::setSymbolVisibility(newGlobal,
                                   SymbolTable::getSymbolVisibility(global));

  // Create an initializer to initialize the new global to the padding
  // value of the pack. This is necessary because we have folded the pack
  // op into the new global. Additional analysis could tell us whether the
  // padding is actually needed, but for now we always pad.
  Location globalLoc = newGlobal->getLoc();
  auto initializerOp = rewriter.create<IREE::Util::InitializerOp>(globalLoc);
  auto initializerBuilder =
      OpBuilder::atBlockBegin(initializerOp.addEntryBlock());
  // If the transform has no padding value, then use a zero padding value, since
  // the global may need a pad value even if the layout found in the analysis
  // did not have a padding value.
  auto transformPadVal = transform->getConstantPadValue();
  TypedAttr constPadAttr =
      transformPadVal.has_value()
          ? transformPadVal.value()
          : rewriter.getZeroAttr(transformedType.getElementType());
  Value padValue =
      initializerBuilder.create<arith::ConstantOp>(globalLoc, constPadAttr);
  auto splatOp = initializerBuilder.create<IREE::Flow::TensorSplatOp>(
      globalLoc, transformedType, padValue, /*result_dims=*/ValueRange{});
  initializerBuilder.create<IREE::Util::GlobalStoreOp>(
      globalLoc, splatOp.getResult(), newGlobal.getName());
  initializerBuilder.create<IREE::Util::ReturnOp>(globalLoc);

  // Rewrite loads and stores to use the new global.
  SmallVector<OpFoldResult> innerTilesOfr;
  for (auto tile : transform->getInnerTileSizes()) {
    innerTilesOfr.push_back(rewriter.getIndexAttr(tile));
  }
  for (auto node : edgeNodes) {
    if (llvm::any_of(node.getUsers(), [](Operation *user) {
          return isa<IREE::Util::GlobalStoreOp>(user);
        })) {
      rewriter.setInsertionPointAfterValue(node);
      auto dest = rewriter.create<tensor::EmptyOp>(
          node.getLoc(), transformedType.getShape(),
          transformedType.getElementType());
      Value nodePadValue =
          rewriter.create<arith::ConstantOp>(node.getLoc(), constPadAttr);
      auto pack = rewriter.create<tensor::PackOp>(
          node.getLoc(), node, dest, transform->getInnerDimsPos(),
          innerTilesOfr, /*padding_value=*/nodePadValue,
          transform->getOuterDimsPerm());
      setFoldablePackUnPackAttribute(pack);
      for (auto user : node.getUsers()) {
        if (isa<IREE::Util::GlobalStoreOp>(user)) {
          rewriter.create<IREE::Util::GlobalStoreOp>(
              user->getLoc(), pack.getResult(), newGlobal);
          rewriter.eraseOp(user);
        }
      }
    }
    if (auto loadOp = node.getDefiningOp<IREE::Util::GlobalLoadOp>()) {
      rewriter.setInsertionPoint(loadOp);
      auto newLoad = rewriter.create<IREE::Util::GlobalLoadOp>(loadOp->getLoc(),
                                                               newGlobal);
      auto dest = rewriter.create<tensor::EmptyOp>(
          loadOp->getLoc(), originalType.getShape(),
          originalType.getElementType());
      auto unpack = rewriter.create<tensor::UnPackOp>(
          loadOp.getLoc(), newLoad, dest, transform->getInnerDimsPos(),
          innerTilesOfr, transform->getOuterDimsPerm());
      setFoldablePackUnPackAttribute(unpack);
      rewriter.replaceOp(loadOp, unpack.getResult());
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Attribute helpers
//===----------------------------------------------------------------------===//

static StringAttr getNodeTypeStringAttr(MLIRContext *ctx,
                                        DataLayoutNodeType type) {
  switch (type) {
  case DataLayoutNodeType::UNINITIALIZED:
    return StringAttr::get(ctx, "UNINITIALIZED");
  case DataLayoutNodeType::INTERMEDIATE:
    return StringAttr::get(ctx, "INTERMEDIATE");
  case DataLayoutNodeType::BARRIER:
    return StringAttr::get(ctx, "BARRIER");
  default:
    assert(false && "invalid DataLayoutNodeType");
  }
}

static DataLayoutNodeType getNodeTypeFromStringAttr(StringAttr attr) {
  if (attr.getValue().equals(StringRef("UNINITIALIZED")))
    return DataLayoutNodeType::UNINITIALIZED;
  if (attr.getValue().equals(StringRef("INTERMEDIATE")))
    return DataLayoutNodeType::INTERMEDIATE;
  return DataLayoutNodeType::BARRIER;
}

void setNodeTypeAttribute(Operation *op, DataLayoutNodeType nodeType) {
  op->setAttr(kDataLayoutNodeTypeAttr,
              getNodeTypeStringAttr(op->getContext(), nodeType));
  return;
}

void setFoldablePackUnPackAttribute(Operation *op) {
  op->setAttr(kFoldablePackUnPack, UnitAttr::get(op->getContext()));
  return;
}

bool hasFoldablePackUnPackAttribute(Operation *op) {
  return static_cast<bool>(op->getAttrOfType<UnitAttr>(kFoldablePackUnPack));
}

std::optional<DataLayoutNodeType> getNodeTypeFromAttr(Operation *op) {
  if (auto attr = op->getAttrOfType<StringAttr>(kDataLayoutNodeTypeAttr)) {
    return getNodeTypeFromStringAttr(attr);
  }
  return std::nullopt;
}

void setDataLayoutTransformationAttributes(Operation *op,
                                           DataLayoutTransformation *transform,
                                           StringRef transformID) {
  op->setAttr(transformID, transform->makeTransformArrayAttr(op->getContext()));
  return;
}

} // namespace mlir::iree_compiler::GlobalOptimization
