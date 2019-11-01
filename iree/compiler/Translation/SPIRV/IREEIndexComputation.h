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

//===- IREEIndexComputation.h ----------------------------------*- C++//-*-===//
//
// Index Propagation for IREE statements that are used in dispatch functions.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_TRANSLATION_SPIRV_H
#define IREE_COMPILER_TRANSLATION_SPIRV_H

#include "iree/compiler/Translation/SPIRV/IREECodegenUtils.h"
#include "iree/compiler/Translation/SPIRV/XLAIndexPropagation.h"
#include "mlir/IR/Function.h"

namespace mlir {
namespace iree_compiler {

/// Index propagation for iree.load_input operation. This operation is
/// essentially a copy from a memref to a tensor. So just copy the index map to
/// the memref operand from the result tensor.
class IREELoadIndexPropagation final
    : public IndexPropagationOp<IREE::LoadInputOp> {
 public:
  using IndexPropagationOp<IREE::LoadInputOp>::IndexPropagationOp;

  LogicalResult propagateIndexMap(
      Operation *operation, IndexComputationCache &indexMap) const override;
};

/// Index propagation for iree.store_output operation. The launch size is
/// assumed to match the shape of the tensor that is being stored. This
/// operation acts as a seed for the index propogation. Each workitem is assumed
/// to compute a single element of this tensor. The range of the index map is
/// the reverse of the launch dimension.
class IREEStoreIndexPropagation final
    : public IndexPropagationOp<IREE::StoreOutputOp> {
 public:
  using IndexPropagationOp<IREE::StoreOutputOp>::IndexPropagationOp;

  LogicalResult propagateIndexMap(
      Operation *operation, IndexComputationCache &indexMap) const override;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_SPIRV_H
