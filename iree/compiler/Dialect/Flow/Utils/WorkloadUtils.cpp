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

#include "iree/compiler/Dialect/Flow/Utils/WorkloadUtils.h"

#include <array>
#include <limits>

#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LLVM.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace iree_compiler {

using Shape::buildOrFindRankedShapeForValue;
using Shape::RankedDimOp;

namespace IREE {
namespace Flow {

Value calculateWorkload(Operation *op, Value baseOperand) {
  OpBuilder builder(op->getContext());
  auto baseOperandType = baseOperand.getType().cast<ShapedType>();
  if (baseOperandType.hasRank() && baseOperandType.hasStaticShape()) {
    // Just a constant (note this also covers rank0).
    int64_t numElements = baseOperandType.getNumElements();
    if (numElements > std::numeric_limits<int32_t>::max()) {
      return (op->emitOpError()
              << "total element count > 32bit integer capacity"),
             nullptr;
    }
    builder.setInsertionPointToStart(op->getBlock());
    return builder.create<ConstantOp>(
        op->getLoc(), builder.getIndexType(),
        builder.getIntegerAttr(builder.getIndexType(), numElements));
  } else if (baseOperandType.hasRank()) {
    // Materialize a ranked shape and compute.
    auto rankedShape = buildOrFindRankedShapeForValue(
        op->getLoc(), baseOperand, builder.getIndexType(), builder);
    if (!rankedShape) return nullptr;

    // Set the insertion point to the earliest feasible (either to just after
    // the input ranked shape op or the start of the block we are emitting
    // into).
    // TODO(laurenzo): Need to overhaul insertion points generally in
    // dispatch region formation as there are dominance hazards here.
    if (rankedShape.getDefiningOp()) {
      builder.setInsertionPointAfter(rankedShape.getDefiningOp());
    } else {
      builder.setInsertionPointToStart(op->getBlock());
    }

    Value numElements;
    for (int64_t i = 0, e = baseOperandType.getRank(); i < e; ++i) {
      auto dim = builder.create<RankedDimOp>(op->getLoc(), rankedShape, i);
      if (!numElements) {
        numElements = dim;
        continue;
      }
      numElements = builder.create<MulIOp>(op->getLoc(), numElements, dim);
    }
    return numElements;
  } else {
    op->emitOpError()
        << "unranked shapes not supported for workload calculation";
    return nullptr;
  }
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
