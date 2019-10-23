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

#include "iree/compiler/Utils/OpCreationUtils.h"

#include <cstdint>

#include "iree/compiler/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"

namespace mlir {
namespace iree_compiler {

namespace {

ElementsAttr elementsAttrFromArray(OpBuilder &builder,
                                   ArrayRef<int64_t> elements) {
  return DenseIntElementsAttr::get(
      RankedTensorType::get(elements.size(), builder.getIntegerType(64)),
      elements);
}

}  // namespace

IREE::ConstantOp createArrayConstant(OpBuilder &builder, Location loc,
                                     llvm::ArrayRef<int64_t> elements) {
  auto elementsAttr = elementsAttrFromArray(builder, elements);
  return builder.create<IREE::ConstantOp>(loc, elementsAttr);
}

}  // namespace iree_compiler
}  // namespace mlir
