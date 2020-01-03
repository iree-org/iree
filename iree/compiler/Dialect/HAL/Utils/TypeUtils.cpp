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

#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

int32_t getRoundedElementByteWidth(Type type) {
  return (type.getIntOrFloatBitWidth() + 8 - 1) / 8;
}

SmallVector<Value, 4> getStaticShapeDims(Location loc, ShapedType shapedType,
                                         ConversionPatternRewriter &rewriter) {
  SmallVector<Value, 4> shape;
  if (shapedType.getRank() >= 1) {
    for (auto dim : shapedType.getShape()) {
      shape.push_back(rewriter.createOrFold<mlir::ConstantOp>(
          loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(dim))));
    }
  }
  return shape;
}

SmallVector<Value, 4> getShapeDims(Value shapedValue,
                                   ConversionPatternRewriter &rewriter) {
  // TODO(benvanik): dynamic shape support.
  return getStaticShapeDims(shapedValue.getLoc(),
                            shapedValue.getType().cast<ShapedType>(), rewriter);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
