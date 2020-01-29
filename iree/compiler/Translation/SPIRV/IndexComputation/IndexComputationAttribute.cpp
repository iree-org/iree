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
#include "iree/compiler/Translation/SPIRV/IndexComputation/IndexComputationAttribute.h"

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace iree_compiler {

#include "iree/compiler/Translation/SPIRV/IndexComputation/IndexComputationAttr.cpp.inc"

//===----------------------------------------------------------------------===//
// Attribute build method
//===----------------------------------------------------------------------===//

IREE::IndexAttr getIndexAttr(
    MLIRContext *context, ArrayRef<AffineMap> resultIndexMap,
    ArrayRef<SmallVector<AffineMap, 1>> operandIndices) {
  auto getArrayAttrFn = [&](ArrayRef<AffineMap> maps) -> ArrayAttr {
    SmallVector<Attribute, 2> affineMapAttrs;
    affineMapAttrs.reserve(maps.size());
    for (auto map : maps) {
      affineMapAttrs.push_back(AffineMapAttr::get(map));
    }
    return ArrayAttr::get(affineMapAttrs, context);
  };
  SmallVector<Attribute, 2> operandIndexAttrs;
  operandIndexAttrs.reserve(operandIndices.size());
  for (auto operandIndex : operandIndices) {
    operandIndexAttrs.push_back(getArrayAttrFn(operandIndex));
  }
  return IREE::IndexAttr::get(getArrayAttrFn(resultIndexMap),
                              ArrayAttr::get(operandIndexAttrs, context),
                              context);
}

namespace {

/// Helper method to check if the result_indices of `indexAttr` matches the
/// indices passed in `resultIndices'
bool matchResultIndices(IREE::IndexAttr indexAttr,
                        ArrayRef<AffineMap> resultIndices) {
  if (indexAttr.result_index().size() != resultIndices.size()) {
    return false;
  }
  for (auto index : llvm::enumerate(indexAttr.result_index())) {
    if (index.value().cast<AffineMapAttr>().getValue() !=
        resultIndices[index.index()]) {
      return false;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Attribute updaters
//===----------------------------------------------------------------------===//

/// Adds a new ArrayAttr containing elements of `newEntryElements` to `currAttr`
/// which is an index computation attribute. Size of the `newEntryElements` must
/// match the size of existing ArrayAttr in `currAttr`.
ArrayAttr updateIndexComputationAttr(MLIRContext *context, ArrayAttr currAttr,
                                     IREE::IndexAttr newEntry) {
  if (currAttr) {
    SmallVector<Attribute, 2> updatedList;
    auto currAttrValue = currAttr.getValue();
    updatedList.append(currAttrValue.begin(), currAttrValue.end());
    updatedList.push_back(newEntry);
    return ArrayAttr::get(updatedList, context);
  }
  return ArrayAttr::get(newEntry, context);
}

/// Records the `resultIndexMap` representing an access of an element of a
/// tensor to `currAttr`, which is an index computation attribute.
ArrayAttr updateIndexComputationAttrWithResultIndex(
    MLIRContext *context, ArrayAttr currAttr,
    ArrayRef<AffineMap> resultIndexMap) {
  if (currAttr) {
    // Check for a duplicate entry of resultIndexMap.
    for (auto entry : currAttr.getValue()) {
      if (matchResultIndices(entry.cast<IREE::IndexAttr>(), resultIndexMap)) {
        return currAttr;
      }
    }
  }
  return updateIndexComputationAttr(context, currAttr,
                                    getIndexAttr(context, resultIndexMap));
}

/// Records the 'operandIndices` representing the indices of elements of the
/// tensor operands needed to compute the value of the result tensor at position
/// represented by `resultIndexMap`.
ArrayAttr updateIndexComputationAttrWithOperandIndices(
    MLIRContext *context, ArrayAttr currAttr,
    ArrayRef<AffineMap> resultIndexMap,
    ArrayRef<SmallVector<AffineMap, 1>> operandIndices) {
  assert(currAttr);
  SmallVector<Attribute, 4> updatedList(currAttr.size());
  for (auto list : enumerate(currAttr.getValue())) {
    auto indexAttr = list.value().cast<IREE::IndexAttr>();
    // Check if the result index matches. If so update the operand indices.
    if (matchResultIndices(indexAttr, resultIndexMap)) {
      assert(indexAttr.operand_indices().getValue().empty() &&
             "resetting operand_indices for an IndexAttr");
      updatedList[list.index()] =
          getIndexAttr(context, resultIndexMap, operandIndices);
    } else {
      updatedList[list.index()] = list.value();
    }
  }
  return ArrayAttr::get(updatedList, context);
}

/// Gets the next symbol number.
IntegerAttr updateMaxSymbolNumberAttr(MLIRContext *context,
                                      IntegerAttr currAttr) {
  if (currAttr) {
    auto maxSymbolNum = currAttr.getInt();
    return IntegerAttr::get(currAttr.getType(), maxSymbolNum + 1);
  }
  return IntegerAttr::get(IntegerType::get(32, context), 0);
}

/// Records the symbol number associated with the element of a tensor accessed
/// within a workitem.
ArrayAttr updateTensorIndexToSymbolNumberAttr(MLIRContext *context,
                                              ArrayAttr currAttr,
                                              AffineMap index,
                                              unsigned symbolNum) {
  SmallVector<Attribute, 1> elements;
  if (currAttr) {
    elements.reserve(currAttr.size() + 1);
    elements.append(currAttr.begin(), currAttr.end());
  }
  SmallVector<Attribute, 2> newElement = {
      AffineMapAttr::get(index),
      IntegerAttr::get(IntegerType::get(32, context), symbolNum)};
  auto newEntry = ArrayAttr::get(newElement, context);
  elements.push_back(newEntry);
  return ArrayAttr::get(elements, context);
}

//===----------------------------------------------------------------------===//
// Attribute Names
//===----------------------------------------------------------------------===//

/// Returns the name of the attribute that tracks index computation.
StringRef getIndexComputationAttrName() {
  return "iree.index_computation_info";
}

/// Returns the name of the attribute for storing the number of dims used for
/// all affine expressions in the dispatch region.
StringRef getNumDimsAttrName() { return "iree.num_dims"; }

/// Returns the name of the attribute that tracks the total number of symbols
/// for the dispatch function.
StringRef getMaxSymbolNumAttrName() { return "iree.max_symbol_num"; }

/// Returns the name of the attribute that tracks symbol numbers associated with
/// the scalar values produced by these ops.
StringRef getSymbolNumberAttrName() { return "iree.symbol_number_info"; }

//===----------------------------------------------------------------------===//
// Attribute Getter/Setters
//===----------------------------------------------------------------------===//

/// Gets an attribute associated with a block argument.
template <typename T>
T getBlockArgumentAttr(BlockArgument blockArg, StringRef attrName) {
  auto block = blockArg.getOwner();
  auto funcOp = dyn_cast<FuncOp>(block->getParentOp());
  if (!funcOp) {
    emitError(blockArg.getLoc(),
              "unimplemented index computation for block argument when "
              "block is not in a function");
    return nullptr;
  }
  return funcOp.getArgAttrOfType<T>(blockArg.getArgNumber(), attrName);
}

/// Updates an attribute associated with a block argument
template <typename T>
LogicalResult setBlockArgumentAttr(BlockArgument blockArg, T updatedAttr,
                                   StringRef attrName) {
  auto block = blockArg.getOwner();
  auto funcOp = dyn_cast<FuncOp>(block->getParentOp());
  if (!funcOp) {
    return emitError(blockArg.getLoc(),
                     "unimplemented index computation for block argument when "
                     "block is not in a function");
  }
  auto currAttr =
      funcOp.getArgAttrOfType<ArrayAttr>(blockArg.getArgNumber(), attrName);
  if (currAttr != updatedAttr) {
    funcOp.setArgAttr(blockArg.getArgNumber(), attrName, updatedAttr);
  }
  return success();
}

/// Sets the attribute that records the index maps association with an operation
LogicalResult setOpAttr(Operation *op, ArrayAttr updatedAttr,
                        StringRef attrName) {
  auto currAttr = op->getAttrOfType<ArrayAttr>(attrName);
  if (currAttr != updatedAttr) {
    op->setAttr(attrName, updatedAttr);
  }
  return success();
}

/// Records the `resultIndexMap` representing an access of an element of the
/// `blockArg` to the index computation attribute.
LogicalResult addBlockArgIndexMap(BlockArgument blockArg,
                                  AffineMap resultIndexMap) {
  auto attrName = getIndexComputationAttrName();
  auto currAttr = getBlockArgumentAttr<ArrayAttr>(blockArg, attrName);
  auto updatedAttr = updateIndexComputationAttrWithResultIndex(
      blockArg.getContext(), currAttr, resultIndexMap);
  return setBlockArgumentAttr(blockArg, updatedAttr, attrName);
}

/// Records the `resultIndexMap` representing an access to the result of the
/// `op` to the index computation attribute.
LogicalResult addOpResultIndexMap(Operation *op,
                                  ArrayRef<AffineMap> resultIndexMap) {
  // TODO(ravishankarm): This check can probably be removed, but need to do it
  // after an example of this is worked through.
  if (op->getNumResults() > 1) {
    return emitError(
        op->getLoc(),
        "unimplemented index propagation for op with multiple results");
  }
  auto attrName = getIndexComputationAttrName();
  auto currAttr = op->getAttrOfType<ArrayAttr>(attrName);
  auto updatedAttr = updateIndexComputationAttrWithResultIndex(
      op->getContext(), currAttr, resultIndexMap);
  return setOpAttr(op, updatedAttr, attrName);
}

}  // namespace

//===----------------------------------------------------------------------===//
// Interface functions
//===----------------------------------------------------------------------===//

/// Records an index map for a tensor value.
LogicalResult addNewIndexMapForValue(Value value, AffineMap resultIndexMap) {
  // Check if the Value is a block argument or has a defining operation.
  if (value.isa<BlockArgument>()) {
    return addBlockArgIndexMap(value.cast<BlockArgument>(), resultIndexMap);
  }
  return addOpResultIndexMap(value.getDefiningOp(), resultIndexMap);
}

Optional<int64_t> addNewSymbolNumberForTensorIndex(Value value,
                                                   AffineMap index) {
  if (value.isa<BlockArgument>() ||
      !isa<IREE::LoadInputOp>(value.getDefiningOp())) {
    emitError(value.getLoc(),
              "only result of a iree.load_input can be associated with "
              "an symbol number");
    return {};
  }
  auto loadInputOp = cast<IREE::LoadInputOp>(value.getDefiningOp());
  auto context = value.getContext();
  auto funcOp = loadInputOp.getOperation()->getParentOfType<FuncOp>();

  // Find the symbol number to use. It is recorded as an attribute on the
  // dispatch function.
  auto maxSymbolNumAttrName = getMaxSymbolNumAttrName();
  auto currNumSymbolsAttr =
      funcOp.getAttrOfType<IntegerAttr>(maxSymbolNumAttrName);
  auto updatedNumSymbolsAttr =
      updateMaxSymbolNumberAttr(context, currNumSymbolsAttr);
  funcOp.setAttr(maxSymbolNumAttrName, updatedNumSymbolsAttr);
  unsigned symbolNumber = static_cast<unsigned>(updatedNumSymbolsAttr.getInt());

  // Record the mapping from element at tensor index to the symbol.
  auto srcArg = loadInputOp.src().cast<BlockArgument>();
  auto attrName = getSymbolNumberAttrName();
  auto currAttr = getBlockArgumentAttr<ArrayAttr>(srcArg, attrName);
  auto updatedAttr = updateTensorIndexToSymbolNumberAttr(
      value.getContext(), currAttr, index, symbolNumber);
  setBlockArgumentAttr(srcArg, updatedAttr, attrName);
  return symbolNumber;
}

LogicalResult addOperandsIndexMap(
    Operation *op, AffineMap resultIndexMap,
    ArrayRef<SmallVector<AffineMap, 1>> operandIndices) {
  auto attrName = getIndexComputationAttrName();
  auto currAttr = op->getAttrOfType<ArrayAttr>(attrName);
  auto updatedAttr = updateIndexComputationAttrWithOperandIndices(
      op->getContext(), currAttr, resultIndexMap, operandIndices);
  return setOpAttr(op, updatedAttr, attrName);
}

AffineMap getAffineMap(FuncOp funcOp, ArrayRef<AffineExpr> exprs) {
  auto numDimsAttr = funcOp.getAttrOfType<IntegerAttr>(getNumDimsAttrName());
  auto maxSymbolNumAttr =
      funcOp.getAttrOfType<IntegerAttr>(getMaxSymbolNumAttrName());
  return AffineMap::get(numDimsAttr.getInt(),
                        (maxSymbolNumAttr ? maxSymbolNumAttr.getInt() + 1 : 0),
                        exprs);
}

void getIndexMapsForValue(Value value, SmallVectorImpl<AffineMap> &indices) {
  auto attrName = getIndexComputationAttrName();
  ArrayAttr allIndices =
      (value.isa<BlockArgument>()
           ? getBlockArgumentAttr<ArrayAttr>(value.cast<BlockArgument>(),
                                             attrName)
           : value.getDefiningOp()->getAttrOfType<ArrayAttr>(attrName));
  if (!allIndices) {
    return;
  }
  // TODO(ravishankarm): If the value is coming from an operation, then the
  // result values need to be checked to see which result matches the `value`
  // passed. For now just handle single result operations.
  for (auto affineMapList : allIndices) {
    auto indexList = affineMapList.cast<IREE::IndexAttr>();
    assert(indexList.result_index().size() == 1 &&
           "unhandled multiple result operation");
    indices.push_back(indexList.result_index()
                          .getValue()[0]
                          .cast<AffineMapAttr>()
                          .getValue());
  }
}

void getIndexMapsForOperands(
    Operation *op, ArrayRef<AffineMap> resultIndices,
    SmallVectorImpl<SmallVector<AffineMap, 1>> &operandIndices) {
  auto allIndices = op->getAttrOfType<ArrayAttr>(getIndexComputationAttrName());
  if (!allIndices) {
    return;
  }
  if (op->getNumResults() == 0) {
    return;
  }
  assert(op->getNumResults() == 1);
  for (auto affineMapList : allIndices) {
    auto indexList = affineMapList.cast<IREE::IndexAttr>();
    if (matchResultIndices(indexList, resultIndices)) {
      operandIndices.clear();
      operandIndices.resize(op->getNumOperands());
      for (auto indices : llvm::enumerate(indexList.operand_indices())) {
        auto &operandIndexList = operandIndices[indices.index()];
        operandIndexList.resize(indices.value().cast<ArrayAttr>().size());
        for (auto mapAttr :
             llvm::enumerate(indices.value().cast<ArrayAttr>())) {
          operandIndexList[mapAttr.index()] =
              mapAttr.value().cast<AffineMapAttr>().getValue();
        }
      }
    }
  }
}

void getSymbolNumberForTensorIndex(
    BlockArgument arg,
    SmallVectorImpl<std::pair<AffineMap, unsigned>> &symbolInfo) {
  auto attrName = getSymbolNumberAttrName();
  auto attr = getBlockArgumentAttr<ArrayAttr>(arg, attrName);
  if (!attr) {
    return;
  }
  for (auto elements : attr) {
    auto pairAttr = elements.cast<ArrayAttr>().getValue();
    symbolInfo.emplace_back(
        pairAttr[0].cast<AffineMapAttr>().getValue(),
        static_cast<unsigned>(pairAttr[1].cast<IntegerAttr>().getInt()));
  }
}

void setNumLaunchDims(FuncOp funcOp, unsigned numLaunchDims) {
  funcOp.setAttr(getNumDimsAttrName(),
                 IntegerAttr::get(IntegerType::get(32, funcOp.getContext()),
                                  numLaunchDims));
}

}  // namespace iree_compiler
}  // namespace mlir
