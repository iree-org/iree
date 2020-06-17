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

#ifndef IREE_COMPILER_DIALECT_VM_IR_VMTRAITS_H_
#define IREE_COMPILER_DIALECT_VM_IR_VMTRAITS_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {
namespace IREE {
namespace VM {

template <typename ConcreteType>
class DebugOnly : public OpTrait::TraitBase<ConcreteType, DebugOnly> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    // TODO(benvanik): verify debug-only.
    return success();
  }
};

template <typename ConcreteType>
class FullBarrier : public OpTrait::TraitBase<ConcreteType, FullBarrier> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    // TODO(benvanik): verify full barrier.
    return success();
  }
};

template <typename ConcreteType>
class PseudoOp : public OpTrait::TraitBase<ConcreteType, PseudoOp> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    // TODO(benvanik): verify pseudo op (not serializable?).
    return success();
  }
};

}  // namespace VM
}  // namespace IREE
}  // namespace OpTrait
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_IR_VMTRAITS_H_
