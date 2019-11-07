// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- IndexComputationAttribute.cpp ---------------------------*- C++//-*-===//
//
// Defines utility methods used to update information of attribute that stores
// the result of the IndexComputation Analysis
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Translation/SPIRV/IndexComputationAttribute.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace iree_compiler {
namespace index_computation_attribute {

namespace {
/// Adds a new ArrayAttr containing elements of `newEntryElements` to `currAttr`
/// which is an index computation attribute. Size of the `newEntryElements` must
/// match the size of existing ArrayAttr in `currAttr`.
ArrayAttr updateIndexComputationAttr(MLIRContext *context, ArrayAttr currAttr,
                                     ArrayRef<Attribute> newEntryElements) {
  auto newEntry = ArrayAttr::get(newEntryElements, context);
  if (currAttr) {
    SmallVector<Attribute, 2> updatedList;
    auto currAttrValue = currAttr.getValue();
    updatedList.reserve(currAttrValue.size() + 1);
    updatedList.append(currAttrValue.begin(), currAttrValue.end());
    if (!updatedList.empty()) {
      assert(currAttrValue.front().isa<ArrayAttr>() &&
             currAttrValue.front().cast<ArrayAttr>().size() ==
                 newEntryElements.size() &&
             "All elements of the index information attribute must have the "
             "same size");
    }
    updatedList.push_back(newEntry);
    return ArrayAttr::get(updatedList, context);
  }
  return ArrayAttr::get(newEntry, context);
}

/// Records the `resultIndexMap` representing an access of an element of a
/// tensor to `currAttr`, which is an index computation attribute.
ArrayAttr updateIndexComputationAttrWithResultIndex(MLIRContext *context,
                                                    ArrayAttr currAttr,
                                                    AffineMap resultIndexMap,
                                                    unsigned numOperands) {
  if (currAttr) {
    // Check for a duplicate entry of resultIndexMap.
    for (auto entry : currAttr.getValue()) {
      auto affineMapArrayAttr = entry.cast<ArrayAttr>().getValue();
      assert(affineMapArrayAttr.size() == numOperands + 1);
      if (affineMapArrayAttr[0].cast<AffineMapAttr>().getValue() ==
          resultIndexMap) {
        return currAttr;
      }
    }
  }
  SmallVector<Attribute, 4> newEntryElements(numOperands + 1);
  newEntryElements[0] = AffineMapAttr::get(resultIndexMap);
  return updateIndexComputationAttr(context, currAttr, newEntryElements);
}

/// Records the 'operandIndices` representing the indices of elements of the
/// tensor operands needed to compute the value of the result tensor at position
/// represented by `resultIndexMap`.
ArrayAttr updateIndexComputationAttrWithOperandIndices(
    MLIRContext *context, ArrayAttr currAttr, AffineMap resultIndexMap,
    ArrayRef<AffineMap> operandIndices) {
  if (!currAttr) {
    return currAttr;
  }
  SmallVector<Attribute, 4> updatedList(currAttr.size());
  for (auto list : enumerate(currAttr.getValue())) {
    auto indices = list.value().cast<ArrayAttr>().getValue();
    // Check if the result index matches. If so update the operand indices.
    if (resultIndexMap == indices[0].cast<AffineMapAttr>().getValue()) {
      SmallVector<Attribute, 4> updatedIndices(indices.size());
      assert(indices.size() == operandIndices.size() + 1);
      updatedIndices[0] = indices[0];
      for (auto operandIndex : enumerate(operandIndices)) {
        updatedIndices[operandIndex.index() + 1] =
            AffineMapAttr::get(operandIndex.value());
      }
      updatedList[list.index()] = ArrayAttr::get(updatedIndices, context);
    } else {
      updatedList[list.index()] = list.value();
    }
  }
  return ArrayAttr::get(updatedList, context);
}

/// Records the `resultIndexMap` representing an access of an element of the
/// `blockArg` to the index computation attribute.
LogicalResult addBlockArgIndexMap(BlockArgument *blockArg,
                                  AffineMap resultIndexMap) {
  auto block = blockArg->getOwner();
  auto funcOp = dyn_cast<FuncOp>(block->getParentOp());
  if (!funcOp) {
    return emitError(blockArg->getLoc(),
                     "unimplemented index computation for block argument when "
                     "block is not in a function");
  }
  auto attrName = getIndexComputationAttrName();
  auto currAttr =
      funcOp.getArgAttrOfType<ArrayAttr>(blockArg->getArgNumber(), attrName);
  auto updatedAttr = updateIndexComputationAttrWithResultIndex(
      blockArg->getContext(), currAttr, resultIndexMap, /*numOperands=*/0);
  if (currAttr != updatedAttr) {
    funcOp.setArgAttr(blockArg->getArgNumber(), attrName, updatedAttr);
  }
  return success();
}

/// Records the `resultIndexMap` representing an access to the result of the
/// `op` to the index computation attribute.
LogicalResult addOpResultIndexMap(Operation *op, AffineMap resultIndexMap) {
  if (op->getNumResults() > 1) {
    return emitError(
        op->getLoc(),
        "unimplemented index propagation for op with multiple results");
  }
  auto numOperands = op->getNumOperands();
  auto attrName = getIndexComputationAttrName();
  auto currAttr = op->getAttrOfType<ArrayAttr>(attrName);
  auto updatedAttr = updateIndexComputationAttrWithResultIndex(
      op->getContext(), currAttr, resultIndexMap, numOperands);
  if (currAttr != updatedAttr) {
    op->setAttr(attrName, updatedAttr);
  }
  return success();
}
}  // namespace

StringRef getIndexComputationAttrName() {
  return "iree.index_computation_info";
}

LogicalResult addNewIndexMapForValue(Value *value, AffineMap resultIndexMap) {
  // Check if the Value is a block argument or has a defining operation.
  auto valueKind = value->getKind();
  if (valueKind == Value::Kind::BlockArgument) {
    return addBlockArgIndexMap(cast<BlockArgument>(value), resultIndexMap);
  }
  return addOpResultIndexMap(value->getDefiningOp(), resultIndexMap);
}

LogicalResult addOperandIndexMap(Operation *op, AffineMap resultIndexMap,
                                 ArrayRef<AffineMap> operandIndices) {
  auto attrName = getIndexComputationAttrName();
  auto currAttr = op->getAttrOfType<ArrayAttr>(attrName);
  auto updatedAttr = updateIndexComputationAttrWithOperandIndices(
      op->getContext(), currAttr, resultIndexMap, operandIndices);
  if (currAttr != updatedAttr) {
    op->setAttr(attrName, updatedAttr);
  }
  return success();
}

}  // namespace index_computation_attribute
}  // namespace iree_compiler
}  // namespace mlir
