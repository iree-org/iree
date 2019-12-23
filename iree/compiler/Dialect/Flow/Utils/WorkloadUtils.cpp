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

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LLVM.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

ValuePtr calculateWorkload(Operation *op, ShapedType baseOperandType) {
  OpBuilder builder(op);

  std::array<int32_t, 3> workload = {1, 1, 1};

  // TODO(b/139353314): lookup/calculate based on type/etc.
  if (!baseOperandType.hasStaticShape()) {
    op->emitOpError() << "Dynamic shapes not yet supported";
    return nullptr;
  }
  auto shape = baseOperandType.getShape();
  if (auto conv = dyn_cast_or_null<xla_hlo::ConvOp>(op)) {
    workload[2] =
        shape[conv.dimension_numbers().output_batch_dimension().getInt()];
    int i = 0;
    for (const auto &dim :
         conv.dimension_numbers().output_spatial_dimensions().getIntValues()) {
      if (i > 1) {
        break;
      }
      workload[1 - i++] = shape[dim.getSExtValue()];
    }
  } else {
    // Drop the trailing ones from the shape.
    while (shape.size() > 1 && shape.back() == 1) {
      shape = shape.drop_back();
    }
    if (shape.size() <= 3) {
      // Maps to XYZ (possibly with 1's for unused dimensions).
      for (auto dim : enumerate(shape)) {
        workload[shape.size() - 1 - dim.index()] = dim.value();
      }
    } else {
      // Need to flatten the shape to fit XYZ. For now we just squash from LHS.
      workload[2] = 1;
      for (int i = 0; i < shape.size() - 2; ++i) {
        workload[2] *= shape[i];
      }
      workload[1] = shape[shape.size() - 2];
      workload[0] = shape.back();
    }
  }

  // TODO(b/139353314): optimize workload layout.

  auto constantType = VectorType::get({3}, builder.getIntegerType(32));
  return builder.create<ConstantOp>(
      op->getLoc(), constantType,
      DenseIntElementsAttr::get(constantType, workload));
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
