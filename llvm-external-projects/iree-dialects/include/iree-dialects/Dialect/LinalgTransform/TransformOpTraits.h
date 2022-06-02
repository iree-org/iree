// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_LINALGTRANSFORM_TRANSFORMOPTRAITS_H
#define IREE_DIALECTS_DIALECT_LINALGTRANSFORM_TRANSFORMOPTRAITS_H

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace transform {

template <typename OpTy>
class FunctionalStyleMultiOperandMultiResultTransformOpTrait
    : public OpTrait::TraitBase<
          OpTy, FunctionalStyleMultiOperandMultiResultTransformOpTrait> {
public:
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    Operation *op = this->getOperation();
    auto *transformMappingResource = TransformMappingResource::get();
    for (Value operand : op->getOperands()) {
      effects.emplace_back(MemoryEffects::Read::get(), operand,
                           transformMappingResource);
      effects.emplace_back(MemoryEffects::Free::get(), operand,
                           transformMappingResource);
    }
    for (Value result : op->getResults()) {
      effects.emplace_back(MemoryEffects::Allocate::get(), result,
                           transformMappingResource);
      effects.emplace_back(MemoryEffects::Write::get(), result,
                           transformMappingResource);
    }
    effects.emplace_back(MemoryEffects::Read::get(), PayloadIRResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), PayloadIRResource::get());
  }

  static LogicalResult verifyTrait(Operation *) {
    static_assert(
        OpTy::template hasTrait<MemoryEffectOpInterface::Trait>(),
        "the op must have MemoryEffectOpInterface for this trait to apply");
    return success();
  }
};

} // namespace transform
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_LINALGTRANSFORM_TRANSFORMOPTRAITS_H
