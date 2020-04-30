// Copyright 2020 Google LLC
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

#include "iree/compiler/Dialect/Shape/Plugins/VMLA/VMLAShapeBuilder.h"

#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeInterface.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLAOps.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Optional.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"

using namespace mlir::iree_compiler::Shape;

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VMLA {
namespace {

Value rewriteBatchMatMulPseudoOp(RankedShapeType resultShape,
                                 BatchMatMulPseudoOp op, OpBuilder &builder) {
  auto lhsShape = builder.create<GetRankedShapeOp>(op.getLoc(), op.lhs());
  auto rhsShape = builder.create<GetRankedShapeOp>(op.getLoc(), op.rhs());
  SmallVector<Value, 6> extents;
  // Batch dimension (already been established to match between both operands,
  // so arbitrarily use the LHS).
  extents.push_back(builder.create<RankedDimOp>(op.getLoc(), lhsShape, 0));
  // RHS free dimension.
  extents.push_back(builder.create<RankedDimOp>(op.getLoc(), rhsShape, 1));
  // LHS free dimension.
  extents.push_back(builder.create<RankedDimOp>(op.getLoc(), lhsShape, 1));
  // Due to a quirk of MakeRankedShapeOp, we only pass in the dynamic dims.
  // So prune them down here.
  SmallVector<Value, 6> onlyDynamicExtents;
  for (int i = 0; i < 3; i++) {
    if (resultShape.isDimDynamic(i)) {
      onlyDynamicExtents.push_back(extents[i]);
    }
  }
  return builder.create<MakeRankedShapeOp>(op.getLoc(), resultShape,
                                           onlyDynamicExtents);
}

}  // namespace

void populateVMLACustomOpShapeBuilder(CustomOpShapeBuilderList &builders) {
  auto &b = builders.make<CallbackCustomOpShapeBuilder>();
  b.insertOpRankedShapeBuilder<BatchMatMulPseudoOp>(rewriteBatchMatMulPseudoOp);
}

}  // namespace VMLA
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
