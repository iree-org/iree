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

#include "iree/compiler/Utils/DispatchUtils.h"

#include <numeric>

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {

void calculateWorkload(ArrayRef<int64_t> shape,
                       std::array<int32_t, 3> &workload) {
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
    // Need to flatten the shape to fit XYZ. For now we just squash from
    // LHS.
    auto zRange = shape.drop_back(2);
    workload[2] = std::accumulate(zRange.begin(), zRange.end(), 1,
                                  std::multiplies<int32_t>());
    workload[1] = shape[shape.size() - 2];
    workload[0] = shape.back();
  }
}

Value calculateWorkload(Operation *op, Value baseOperand) {
  OpBuilder builder(op);

  std::array<int32_t, 3> workload = {1, 1, 1};

  // TODO(b/139353314): lookup/calculate based on type/etc.
  auto resultType = baseOperand.getType();
  if (auto shapedType = resultType.dyn_cast<ShapedType>()) {
    if (!shapedType.hasStaticShape()) {
      op->emitOpError() << "Dynamic shapes not yet supported";
      return nullptr;
    }
    auto shape = shapedType.getShape();
    if (auto conv = dyn_cast_or_null<xla_hlo::ConvOp>(op)) {
      workload[2] =
          shape[conv.dimension_numbers().output_batch_dimension().getInt()];
      int i = 0;
      for (const auto &dim : conv.dimension_numbers()
                                 .output_spatial_dimensions()
                                 .getIntValues()) {
        if (i > 1) {
          break;
        }
        workload[1 - i++] = shape[dim.getSExtValue()];
      }
    } else {
      calculateWorkload(shape, workload);
    }
  }

  // TODO(b/139353314): optimize workload layout.

  auto constantType = RankedTensorType::get({3}, builder.getIntegerType(32));
  return builder.create<ConstantOp>(
      op->getLoc(), constantType,
      DenseIntElementsAttr::get(constantType, workload));
}

}  // namespace iree_compiler
}  // namespace mlir
