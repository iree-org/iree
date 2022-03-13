//===-- TransformOpInterface.h - Interface for transform ops ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORM_TRANSFORM_OP_INTERFACE_H
#define MLIR_DIALECT_LINALG_TRANSFORM_TRANSFORM_OP_INTERFACE_H

#include <mlir/IR/OpDefinition.h>

#include <type_traits>

#include "Transforms/Functional.h"
#include "iree-dialects/Dialect/LinalgTransform/TransformOpMapping.h"

namespace mlir {
namespace linalg {
namespace transform {

class TransformOpInterface;

/// The state maintained across applications of various ops implementing the
/// TransformOpInterface. The operations implementing this interface and the
/// surrounding structure are referred to as transform IR. The operations to
/// which transformations apply are referred to as payload IR. The state thus
/// contains the mapping between values defined transform IR ops and payload IR
/// ops. It assumes that each value in the transform IR can be used at most once
/// (since transformations are likely to change the payload IR ops the value
/// corresponds to). Checks that transform IR values correspond to disjoint sets
/// of payload IR ops throughout the transformation.
class TransformState {
 public:
  /// Creates a state for the transformation rooted at the given op.
  explicit TransformState(Operation *root);

  /// Returns the op at which the transformation state is rooted. This is
  /// typically helpful for transformations that apply globally.
  Operation *getTopLevel() const;

  /// Returns the list of ops that the given transform IR value corresponds to.
  /// This is helpful for transformations that apply to a particular handle.
  ArrayRef<Operation *> getPayloadOps(Value value) const;

  /// Applies the transformation specified by the given transform op and updates
  /// the state accordingly.
  LogicalResult applyTransform(TransformOpInterface transform);

  /// The extension mechanism for TransformState. Extensions are expected to
  /// derive this class and may use its methods to access the state. Extensions
  /// are identified by their type and a state can only have one extension of
  /// a particular type.
  class Extension {
    friend class TransformState;

   public:
    // Out-of-line implementation to ensure vtable and metadata are emitted in
    // a single .o file.
    virtual ~Extension();

   protected:
    Extension(TransformState &state) : state(state) {}

    /// Read-only access to the mapping between transform IR values and payload
    /// IR operations contained in the state.
    const TransformOpMapping &getMapping() const { return state.operations; }

    /// Notifies the extension that payload IR operations were associated with
    /// the given transform IR handle. Concrete extensions that are willing to
    /// be notified should override this method.
    virtual void notifySetPayload(Value handle,
                                  ArrayRef<Operation *> operations) {}
    /// Notifies the extension that the association between a transform IR
    /// handle and a list of payload IR operations is about to be removed.
    /// Concrete extensions that are willing to be notified should override this
    /// method.
    virtual void notifyRemovePayload(Value handle,
                                     ArrayRef<Operation *> operations) {}

    /// Notifies the extension that the ops associated with the transform IR
    /// handle changed. Concrete extensions that are willing to be notified
    /// should override this method.
    virtual void notifyUpdatePayload(Value handle, ArrayRef<Operation *> oldOps,
                                     ArrayRef<Operation *> newOps) {}

    /// Sets the payload IR ops associated with the given transform IR value.
    /// Fails if this would result in multiple transform IR values with uses
    /// corresponding to the same payload IR ops. This extension will NOT
    /// be notified about this event.
    LogicalResult setPayloadOps(Value handle,
                                ArrayRef<Operation *> operations) {
      propagatingSetPayload = true;
      LogicalResult result = state.setPayloadOps(handle, operations);
      propagatingSetPayload = false;
      return result;
    }

    /// Forgets the payload IR ops associated with the given transform IR value.
    /// This extension will NOT be notified about this event.
    void removePayloadOps(Value handle) {
      propagatingRemovePayload = true;
      state.removePayloadOps(handle);
      propagatingRemovePayload = false;
    }

    /// Updates the payload IR ops associated with the given transform IR value.
    /// The callback function is called once per associated operation and is
    /// expected to return the modified operation or nullptr. In the latter
    /// case, the corresponding operation is no longer associated with the
    /// transform IR value. This extension will NOT be notified about it.
    void updatePayloadOps(Value handle,
                          function_ref<Operation *(Operation *)> callback) {
      propagatingUpdatePayload = true;
      state.updatePayloadOps(handle, callback);
      propagatingUpdatePayload = false;
    }

   private:
    /// Flags indicating whether a notifiable event originates at this
    /// extension. If set, this extension is not notified about the event.
    bool propagatingSetPayload = false;
    bool propagatingRemovePayload = false;
    bool propagatingUpdatePayload = false;

    /// Sends notifications to about an event to the current extension. Expected
    /// to be called by the TransformState only.
    void sendNotifySetPayload(Value handle, ArrayRef<Operation *> operations) {
      if (!propagatingSetPayload) notifySetPayload(handle, operations);
    }
    void sendNotifyRemovePayload(Value handle,
                                 ArrayRef<Operation *> operations) {
      if (!propagatingRemovePayload) notifyRemovePayload(handle, operations);
    }
    void sendNotifyUpdatePayload(Value handle, ArrayRef<Operation *> oldOps,
                                 ArrayRef<Operation *> newOps) {
      if (!propagatingUpdatePayload)
        notifyUpdatePayload(handle, oldOps, newOps);
    }

    /// Back-reference to the state this is extending.
    TransformState &state;
  };

  /// Adds a new extension of the type specifeid as template parameter,
  /// constructing it with the arguments provided. The extension is owned by the
  /// TransformState. It is expected that the state does not already have an
  /// extension of the same type. Extension constructors are expected to take
  /// a reference to TransformState as first argument, automatically supplied
  /// by this call.
  template <typename Ty, typename... Args>
  Ty &addExtension(Args &&...args) {
    static_assert(
        std::is_base_of<Extension, Ty>::value,
        "only an class derived from TransformState::Extension is allowed here");
    auto ptr = std::make_unique<Ty>(*this, std::forward<Args>(args)...);
    auto result = extensions.try_emplace(TypeID::get<Ty>(), std::move(ptr));
    assert(result.second && "extension already added");
    return *static_cast<Ty *>(result.first->second.get());
  }

  /// Returns the extension of the specified type.
  template <typename Ty>
  Ty &getExtension() {
    static_assert(
        std::is_base_of<Extension, Ty>::value,
        "only an class derived from TransformState::Extension is allowed here");
    auto iter = extensions.find(TypeID::get<Ty>());
    assert(iter != extensions.end() && "extension not found");
    return *static_cast<Ty *>(iter->second.get());
  }

  /// Removes the extension of the specified type.
  template <typename Ty>
  void removeExtension() {
    static_assert(
        std::is_base_of<Extension, Ty>::value,
        "only an class derived from TransformState::Extension is allowed here");
    extensions.erase(TypeID::get<Ty>());
  }

 private:
  /// Identifier for storing top-level value in the `operations` mapping.
  constexpr const static Value kTopLevelValue = Value();

  /// Sets the payload IR ops associated with the given transform IR value.
  /// Fails if this would result in multiple transform IR values with uses
  /// corresponding to the same payload IR ops.
  LogicalResult setPayloadOps(Value value, ArrayRef<Operation *> targets);

  /// Forgets the payload IR ops associated with the given transform IR value.
  void removePayloadOps(Value value);

  /// Updates the payload IR ops associated with the given transform IR value.
  /// The callback function is called once per associated operation and is
  /// expected to return the modified operation or nullptr. In the latter case,
  /// the corresponding operation is no longer associated with the transform IR
  /// value.
  void updatePayloadOps(Value value,
                        function_ref<Operation *(Operation *)> callback);

  /// The mapping between payload IR values and transform IR ops.
  TransformOpMapping operations;

  /// Extensions attached to the TransformState, identified by the TypeID of
  /// their type. Only one extension of any given type is allowed.
  DenseMap<TypeID, std::unique_ptr<Extension>> extensions;
};

/// Local mapping between values defined by a specific op implementing the
/// TransformOpInterface and the payload IR ops they correspond to.
class TransformResults {
  friend class TransformState;

 public:
  /// Indicates that the result of the transform IR op at the given position
  /// corresponds to the given list of payload IR ops. Each result must be set
  /// by the transformation exactly once.
  void set(OpResult value, ArrayRef<Operation *> ops);

 private:
  /// Creates an instance of TransformResults that expects mappings for
  /// `numSegments` values.
  explicit TransformResults(unsigned numSegments);

  /// Gets the list of operations associated with the result at the given
  /// position.
  ArrayRef<Operation *> get(unsigned position) const;

  /// Storage for pointers to payload IR ops that are associated with results of
  /// a transform IR op. `segments` contains as many entries as the transform IR
  /// op has results. Each entry is a reference to a contiguous segment in
  /// the `operations` list that contains the pointers to operations. This
  /// allows for operations to be stored contiguously without nested vectors and
  /// for different segments to be set in any order.
  SmallVector<ArrayRef<Operation *>, 2> segments;
  SmallVector<Operation *> operations;
};

namespace detail {
/// Appends `result` to the vector assuming it corresponds to the success state
/// in `FailureOr<convertible-to-Operation*>`. If `result` is just a
/// `LogicalResult`, does nothing.
template <typename Ty>
std::enable_if_t<std::is_same<Ty, LogicalResult>::value>
appendTransformResultToVector(Ty result,
                              SmallVectorImpl<Operation *> &results) {}

template <typename Ty>
std::enable_if_t<!std::is_same<Ty, LogicalResult>::value>
appendTransformResultToVector(Ty result,
                              SmallVectorImpl<Operation *> &results) {
  static_assert(
      std::is_convertible<typename Ty::value_type, Operation *>::value,
      "Expected transform function to return operations");
  results.push_back(*result);
}
}  // namespace detail

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
  using TransformOpType =
      typename llvm::function_traits<FnTy>::template arg_t<0>;
  static_assert(std::is_convertible<TransformOpType, Operation *>::value,
                "Expected transform function to take an operation");
  for (Operation *target : targets) {
    auto specificOp =
        functional::detail::IsaOr<TransformOpType>::dyn_cast(target);
    if (!specificOp) return failure();

    auto result = transform(specificOp);
    if (failed(result)) return failure();

    detail::appendTransformResultToVector(result, results);
  }
  return success();
}

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
    using TransformOpType = typename llvm::function_traits<
        decltype(&OpTy::applyToOne)>::template arg_t<0>;
    ArrayRef<Operation *> targets =
        state.getPayloadOps(this->getOperation()->getOperand(0));
    SmallVector<Operation *> results;
    if (failed(applyTransformToEach(
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
                      OpTy::template hasTrait<OpTrait::ZeroResult>(),
                  "expected zero- or single-result op");
    return success();
  }
};

}  // namespace transform
}  // namespace linalg
}  // namespace mlir

#include "iree-dialects/Dialect/LinalgTransform/TransformOpInterface.h.inc"

#endif  // MLIR_DIALECT_LINALG_TRANSFORM_TRANSFORM_OP_INTERFACE_H
