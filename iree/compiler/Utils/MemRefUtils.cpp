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

#include "iree/compiler/Utils/MemRefUtils.h"

#include <cassert>

#include "iree/compiler/IR/Ops.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace iree_compiler {
Value *resolveValueToSourceMemRef(Value *value, Operation *useOp) {
  // TODO(benvanik): implement this for real; this is naive but enough for our
  // simple load patterns.
  auto *defInstr = value->getDefiningOp();
  if (auto loadOp = dyn_cast_or_null<LoadOp>(defInstr)) {
    // TODO(benvanik): support views.
    return loadOp.getMemRef();
  }
  return nullptr;
}

Value *wrapAsTensor(Value *value, Operation *srcOp, OpBuilder &builder) {
  if (srcOp->getResult(0)->getType().isa<TensorType>()) {
    if (isa_and_nonnull<IREE::TensorToMemRefOp>(value->getDefiningOp())) {
      return value->getDefiningOp()->getOperand(0);
    }
    auto newOp = builder.create<IREE::MemRefToTensorOp>(srcOp->getLoc(), value);
    value = newOp.getResult();
  }
  return value;
}

Value *wrapAsMemRef(Value *value, Operation *srcOp, OpBuilder &builder) {
  if (value->getType().isa<TensorType>()) {
    if (isa_and_nonnull<IREE::MemRefToTensorOp>(value->getDefiningOp())) {
      return value->getDefiningOp()->getOperand(0);
    }
    auto newOp = builder.create<IREE::TensorToMemRefOp>(srcOp->getLoc(), value);
    value = newOp.getResult();
  }
  return value;
}

Value *loadAccessValue(Location location, Value *operand, OpBuilder &builder) {
  if (operand->getType().isa<MemRefType>() ||
      operand->getType().isa<TensorType>()) {
    return operand;
  }

  auto memRefType = builder.getMemRefType({}, operand->getType());
  if (auto loadOp = dyn_cast_or_null<LoadOp>(operand->getDefiningOp())) {
    // TODO(benvanik): handle creating views.
    if (loadOp.getMemRefType() == memRefType) {
      return loadOp.getMemRef();
    }
  }

  auto allocOp = builder.create<AllocOp>(location, memRefType);
  builder.create<StoreOp>(location, operand, allocOp.getResult(),
                          ArrayRef<Value *>{});
  return allocOp.getResult();
}

Value *loadResultValue(Location location, const Type &originalType,
                       Value *result, OpBuilder &builder) {
  if (originalType.isa<MemRefType>()) {
    return result;
  } else if (auto tensorType = originalType.dyn_cast<TensorType>()) {
    return result;
  }

  auto loadOp = builder.create<LoadOp>(location, result, ArrayRef<Value *>{});
  return loadOp.getResult();
}

}  // namespace iree_compiler
}  // namespace mlir
