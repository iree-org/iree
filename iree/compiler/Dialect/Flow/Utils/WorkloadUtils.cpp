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

std::array<int32_t, 3> convWorkload(xla_hlo::ConvOp conv) {
  std::array<int32_t, 3> workload = {1, 1, 1};
  auto lhs = conv.lhs()->getType().cast<ShapedType>();
  auto rhs = conv.rhs()->getType().cast<ShapedType>();
  std::array<int32_t, 3> lhs_hw = {1, 1};
  int i = 0;
  for (const auto &spatial :
       conv.dimension_numbers().input_spatial_dimensions()) {
    if (i > 1) {
      break;
    }
    lhs_hw[i++] = lhs.getDimSize(spatial.getSExtValue());
  }
  std::array<int32_t, 2> rhs_hw = {1, 1};
  i = 0;
  for (const auto &spatial :
       conv.dimension_numbers().kernel_spatial_dimensions()) {
    if (i > 1) {
      break;
    }
    rhs_hw[i++] = rhs.getDimSize(spatial.getSExtValue());
  }
  std::array<int32_t, 2> padding = {0, 0};
  i = 0;
  for (const auto &pad : conv.padding().getValue().getIntValues()) {
    if (i > 3) {
      break;
    }
    padding[i++ / 2] += pad.getSExtValue();
  }
  // TODO(namiller): Generalize for other ranks and strides once supported.
  workload[2] =
      lhs.getDimSize(conv.dimension_numbers().input_batch_dimension().getInt());
  workload[1] = lhs_hw[0] - rhs_hw[0] + padding[0] + 1;
  workload[0] = lhs_hw[1] - rhs_hw[1] + padding[1] + 1;
  return workload;
}

Value *calculateWorkload(Operation *op, ShapedType baseOperandType) {
  OpBuilder builder(op);

  std::array<int32_t, 3> workload = {1, 1, 1};

  // TODO(b/139353314): lookup/calculate based on type/etc.
  if (!baseOperandType.hasStaticShape()) {
    op->emitOpError() << "Dynamic shapes not yet supported";
    return nullptr;
  }
  if (auto conv = llvm::dyn_cast_or_null<xla_hlo::ConvOp>(op)) {
    workload = convWorkload(conv);
  } else {
    auto shape = baseOperandType.getShape();
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
