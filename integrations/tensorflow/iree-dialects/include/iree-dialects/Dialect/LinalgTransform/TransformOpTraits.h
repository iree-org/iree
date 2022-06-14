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

namespace detail {
/// Appends `result` to the vector assuming it corresponds to the success state
/// in `FailureOr<convertible-to-Operation*>`. If `result` is just a
/// `LogicalResult`, does nothing.
template <typename Ty>
std::enable_if_t<std::is_same<Ty, LogicalResult>::value, LogicalResult>
appendTransformResultToVector(Ty result,
                              SmallVectorImpl<Operation *> &results) {
  return result;
}
template <typename Ty>
std::enable_if_t<!std::is_same<Ty, LogicalResult>::value, LogicalResult>
appendTransformResultToVector(Ty result,
                              SmallVectorImpl<Operation *> &results) {
  static_assert(
      std::is_convertible<typename Ty::value_type, Operation *>::value,
      "expected transform function to return operations");
  if (failed(result))
    return failure();

  results.push_back(*result);
  return success();
}

/// Applies a one-to-one transform to each of the given targets. Puts the
/// results of transforms, if any, in `results` in the same order. Fails if any
/// of the application fails. Individual transforms must be callable with
/// one of the following signatures:
///   - FailureOr<convertible-to-Operation*>(OpTy)
///   - LogicalResult(OpTy)
/// where OpTy is either
///   - Operation *, in which case the transform is always applied;
///   - a concrete Op class, in which case a check is performed whether
///   `targets` contains operations of the same class and a failure is reported
///   if it does not.
template <typename FnTy>
LogicalResult applyTransformToEach(ArrayRef<Operation *> targets,
                                   SmallVectorImpl<Operation *> &results,
                                   FnTy transform) {
  using OpTy = typename llvm::function_traits<FnTy>::template arg_t<0>;
  static_assert(std::is_convertible<OpTy, Operation *>::value,
                "expected transform function to take an operation");
  using RetTy = typename llvm::function_traits<FnTy>::result_t;
  static_assert(std::is_convertible<RetTy, LogicalResult>::value,
                "expected transform function to return LogicalResult or "
                "FailureOr<convertible-to-Operation*>");
  for (Operation *target : targets) {
    auto specificOp = dyn_cast<OpTy>(target);
    if (!specificOp)
      return failure();

    auto result = transform(specificOp);
    if (failed(appendTransformResultToVector(result, results)))
      return failure();
  }
  return success();
}
} // namespace detail

/// Trait implementing the TransformOpInterface for operations applying a
/// transformation to a single operation handle and producing a single operation
/// handle. The op must implement a method with one of the following signatures:
///   - FailureOr<convertible-to-Operation*> applyToOne(OpTy)
///   - LogicalResult applyToOne(OpTy)
/// to perform a transformation that is applied in turn to all payload IR
/// operations that correspond to the handle of the transform IR operation.
/// In the functions above, OpTy is either Operation * or a concrete payload IR
/// Op class that the transformation is applied to (NOT the class of the
/// transform IR op).
template <typename OpTy>
class TargetableSingleOperandOpTrait
    : public OpTrait::TraitBase<OpTy, TargetableSingleOperandOpTrait> {
public:
  /// Applies the transformation to each op from the only target and sets the
  /// only result to correspond to the list of individual results.
  LogicalResult apply(TransformResults &transformResults,
                      TransformState &state) {
    using TransformOpType = typename llvm::function_traits<decltype(
        &OpTy::applyToOne)>::template arg_t<0>;
    ArrayRef<Operation *> targets =
        state.getPayloadOps(this->getOperation()->getOperand(0));
    SmallVector<Operation *> results;
    if (failed(detail::applyTransformToEach(
            targets, results, [&](TransformOpType specificOp) {
              return static_cast<OpTy *>(this)->applyToOne(specificOp);
            })))
      return failure();
    if (OpTy::template hasTrait<OpTrait::OneResult>()) {
      transformResults.set(
          this->getOperation()->getResult(0).template cast<OpResult>(),
          results);
    }
    return success();
  }

  /// Verifies that the op satisfies the requirements for this trait.
  static LogicalResult verifyTrait(Operation *) {
    static_assert(OpTy::template hasTrait<OpTrait::OneOperand>(),
                  "expected single-operand op");
    static_assert(OpTy::template hasTrait<OpTrait::OneResult>() ||
                      OpTy::template hasTrait<OpTrait::ZeroResults>(),
                  "expected zero- or single-result op");
    return success();
  }
};

template <typename OpTy>
class FunctionalStyleTransformOpTrait
    : public OpTrait::TraitBase<OpTy, FunctionalStyleTransformOpTrait> {
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

template <typename OpTy>
class PayloadTransformOpTrait
    : public OpTrait::TraitBase<OpTy, PayloadTransformOpTrait> {
public:
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
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
