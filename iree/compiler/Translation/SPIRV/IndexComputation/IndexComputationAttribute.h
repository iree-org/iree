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

//===- IndexComputationAttribute.h ------------------------------*- C++ -*-===//
//
// Declares utility methods used to update information of attribute that stores
// the result of the IndexComputation Analysis
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_TRANSLATION_SPIRV_INDEXCOMPUTATION_COMPUTATIONATTR_H
#define IREE_COMPILER_TRANSLATION_SPIRV_INDEXCOMPUTATION_COMPUTATIONATTR_H

#include "mlir/IR/Attributes.h"

namespace mlir {

class AffineExpr;
class AffineMap;
class BlockArgument;
class FuncOp;
class Operation;
class Value;

namespace iree_compiler {
namespace index_computation_attribute {

/// The attribute used to store the result of the index computation analysis is
/// logically ArrayAttr<ArrayAttr<AffineMapAttr>>, which is implemented as just
/// an ArrayAttr. This attribute is attached to every operation within the
/// dispatch function which produces a result and function arguments of the
/// dispatch function.
///
/// Each element of the outer ArrayAttr has the same size, equal to
/// (1+numOperands) of the operation the attribute is attached to (For function
/// arguments the numOperands is treated as 0). The [0]-th element of every
/// inner ArrayAttr is the index map that records the mapping from global
/// invocation ID of the workitem to the element of the tensor accessed within
/// the dispatch function. The rest of the elements are the index maps of the
/// operands computed based on the semantics of the operation for a given index
/// map at the [0]-the position.

/// Records an index map for a tensor value.
LogicalResult addNewIndexMapForValue(Value value, AffineMap indexMap);

/// Records the symbol number that is used to refer to an element of a tensor
/// and is needed to express the index maps for all tensors within the dispatch
/// function. The tensor `value` has to be the result of an iree.load_input
/// operation.
Optional<int64_t> addNewSymbolNumberForTensorIndex(Value value,
                                                   AffineMap index);

/// Records the operand index maps for a given result index map,
/// computed based on the `op` semantics.
LogicalResult addOperandsIndexMap(Operation *op, AffineMap resultIndexMap,
                                  ArrayRef<AffineMap> operandIndices);

/// Builds the AffineMap that represents the index of a tensor accessed within
/// the dispatch function.
AffineMap getAffineMap(FuncOp funcOp, ArrayRef<AffineExpr> exprs);

/// Gets the index map associated with the value.
void getIndexMapsForValue(Value value, SmallVectorImpl<AffineMap> &indices);

/// Gets the index map for the operands given the index map of the result.
void getIndexMapsForOperands(Operation *op, AffineMap resultIndex,
                             SmallVectorImpl<AffineMap> &operandIndices);

/// Gets the symbol numbers that map to values read from the tensor. For now the
/// tensors are restricted to be block arguments (which are really of memref
/// type).
void getSymbolNumberForTensorIndex(
    BlockArgument arg,
    SmallVectorImpl<std::pair<AffineMap, unsigned>> &symbolInfo);

/// Sets the number of launch dimensions for the dispatch function.
void setNumLaunchDims(FuncOp funcOp, unsigned numLaunchDims);

}  // namespace index_computation_attribute
}  // namespace iree_compiler
}  // namespace mlir
#endif  // IREE_COMPILER_TRANSLATION_SPIRV_INDEXCOMPUTATION_COMPUTATIONATTR_H
