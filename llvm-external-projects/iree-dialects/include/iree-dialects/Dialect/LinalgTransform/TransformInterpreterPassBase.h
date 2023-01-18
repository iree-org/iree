// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_LINALG_TRANSFORM_TRANSFORM_INTERPRETER_UTILS_H
#define IREE_DIALECTS_LINALG_TRANSFORM_TRANSFORM_INTERPRETER_UTILS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include <memory>

namespace mlir {
class LogicalResult;
class MLIRContext;
class Operation;
class Region;

namespace transform {
namespace detail {
/// Template-free implementation of TransformInterpreterPassBase::initialize.
LogicalResult interpreterBaseInitializeImpl(
    MLIRContext *context, StringRef transformFileName,
    std::shared_ptr<OwningOpRef<ModuleOp>> &module,
    function_ref<OwningOpRef<ModuleOp>(Location)> transformConstructor);

/// Template-free implementation of
/// TransformInterpreterPassBase::runOnOperation.
LogicalResult interpreterBaseRunOnOperationImpl(
    Operation *target, StringRef passName,
    const std::shared_ptr<OwningOpRef<ModuleOp>> &sharedTransformModule,
    const Pass::Option<std::string> &transformFileName,
    const Pass::Option<std::string> &debugPayloadRootTag,
    const Pass::Option<std::string> &debugTransformRootTag);

/// Template-free implementation of
/// TransformInterpreterPassBase::hasSharedTransformModule.
bool hasSharedTransformModuleImpl(
    const std::shared_ptr<OwningOpRef<ModuleOp>> &module);
} // namespace detail

/// Base class for transform dialect interpreter passes that can consume and
/// dump transform dialect scripts in separate files. The pass is controlled by
/// three string options:
///
///   - transformFileName: if non-empty, the name of the file containing the
///     transform script. If empty, `debugTransformRootTag` is considered or the
///     pass root operation must contain a single top-level transform op that
///     will be interpreted.
///   - debugPayloadRootTag: if non-empty, the value of the attribute named
///     `kTransformIreeTagAttrName` indicating the single op that is considered
///     the payload root of the transform interpreter; otherwise, the root
///     operation of the pass is used.
///   - debugTransformRootTag: if non-empty, the value of the attribute named
///     `kTransformIreeTagAttrName` indicating the single top-level transform
///     op contained in the payload root to be used as the entry point by the
///     transform interpreter; mutually exclusive with `transformFileName`.
///
/// The pass runs the transform dialect interpreter as directed by the options
/// and customization hooks described below. The transform IR consumed by this
/// pass may be present within the payload IR, e.g., as a nested module, or
/// constructed on-the-fly. In the latter case, the concrete implementation must
/// implement the corresponding hooks.
///
/// It also provides the mechanism to dump reproducers into stderr
/// (-debug-only=iree-transform-dialect-dump-repro) or into a temporary file
/// (-debug-only=iree-transform-dialect-save-repro) that can be used with this
/// pass in a standalone mode. The reproducer generation is *NOT* guaranteed
/// to be thread-safe. In particular, if multiple instances of the interpreter
/// pass run on multiple payload roots nested in the same container as part of a
/// longer pipeline, the generated payload IR may reflect changes made to other
/// payload roots by the passes following the interpreter pass in the pipeline.
/// This is a fundamental problem with parallelism on nested IR objects in
/// presence of verifiers: dumping *only* the operation the interpreter pass
/// runs on would be safe, but may not be parsable in isolation, which makes the
/// reproducer useless.
///
/// Reproducer generation works *ONLY* in single-threaded mode. If
/// multi-threading is enabled, requesting a reproducer will lead to an
/// immediate failure of the pass. Note that this also applies to the case where
/// only a single instance of the pass running in parallel.
///
/// Concrete passes must derive from this class instead of their generated base
/// class (or PassWrapper), and supply themselves and the generated base class
/// as template arguments. They are *not* expected to to implement `initialize`
/// or `runOnOperation`. They *are* expected to call the copy constructor of
/// this class in their copy constructors, short of which the file-based
/// transform dialect script injection facility will become nonoperational.
///
/// Concrete passes may customize the behavior by implementing the following
/// functions:
///
///   * `getPayloadRoots` sets up the list of operations that are used as
///   payload roots for invocations of the transform interpreter. By default,
///   the interpreter is invoked once on the root operation of the pass.
///
///   * `runBeforeInterpreter` is executed before each interpreter run and may
///   abort the pass if some preconditions are not met. By default, it does
///   nothing and always succeeds.
///
///   * `runAfterInterpreter` is executed after each interpreter run and may
///   abort the pass if some postconditions are not met. By default, it does
///   nothing and succeeds.
///
///   * `constructTransformModule` is executed during pass initialization and
///   may be used to build transform IR that will be fed to the interpreter if
///   it isn't already present in the IR the pass operates on. This is useful
///   when the transform IR cannot be injected into the IR processed by the
///   pass. Transform IR module is shared by all copies of the pass and must not
///   be modified during pass execution. Providing an external source of the
///   transform IR disables this generation.
template <typename Concrete, template <typename> typename GeneratedBase>
class TransformInterpreterPassBase : public GeneratedBase<Concrete> {
public:
  TransformInterpreterPassBase() = default;

  TransformInterpreterPassBase(const TransformInterpreterPassBase &pass) {
    // TODO: if we really don't like shared_ptr, we could also clone the
    // transformModule here.
    sharedTransformModule = pass.sharedTransformModule;
  }

  LogicalResult initialize(MLIRContext *context) final {

#define REQUIRE_PASS_OPTION(NAME)                                              \
  static_assert(                                                               \
      std::is_same_v<                                                          \
          std::remove_reference_t<decltype(std::declval<Concrete &>().NAME)>,  \
          Pass::Option<std::string>>,                                          \
      "required " #NAME " string pass option is missing")

    REQUIRE_PASS_OPTION(transformFileName);
    REQUIRE_PASS_OPTION(debugPayloadRootTag);
    REQUIRE_PASS_OPTION(debugTransformRootTag);

#undef REQUIRE_PASS_OPTION

    StringRef transformFileName =
        static_cast<Concrete *>(this)->transformFileName;
    auto constructorCallback = [this](Location initialLoc) {
      return static_cast<Concrete *>(this)->constructTransformModule(
          initialLoc);
    };
    return detail::interpreterBaseInitializeImpl(
        context, transformFileName, sharedTransformModule, constructorCallback);
  }

  /// Hook for passes to run additional logic in the pass before the
  /// interpreter. If failure is returned, the pass fails and the interpreter is
  /// not run.
  LogicalResult runBeforeInterpreter(Operation *) { return success(); }

  /// Hook for passes to run additional logic in the pass after the interpreter.
  /// Only runs if everything succeeded before. If failure is returned, the pass
  /// fails.
  LogicalResult runAfterInterpreter(Operation *) { return success(); }

  /// Hook for passes to indicate payload roots for successive invocations of
  /// the transform dialect interpreter.
  void getPayloadRoots(SmallVectorImpl<Operation *> &targets) {
    auto *pass = static_cast<Concrete *>(this);
    targets.push_back(pass->getOperation());
  }

  /// Hook for passes to construct transform IR separately from payload IR.
  OwningOpRef<ModuleOp> constructTransformModule(Location) { return nullptr; }

  void runOnOperation() final {
    auto *pass = static_cast<Concrete *>(this);
    SmallVector<Operation *> payloadRoots;
    pass->getPayloadRoots(payloadRoots);

    // The concrete pass may request running the interpreter with payload roots
    // different from the root of the pass, e.g., when it wants to control the
    // traversal of the op without delegating it to the transform dialect.
    for (Operation *root : payloadRoots) {
      if (failed(pass->runBeforeInterpreter(root)) ||
          failed(detail::interpreterBaseRunOnOperationImpl(
              root, pass->getArgument(), sharedTransformModule,
              pass->transformFileName, pass->debugPayloadRootTag,
              pass->debugTransformRootTag)) ||
          failed(pass->runAfterInterpreter(root))) {
        return pass->signalPassFailure();
      }
    }
  }

protected:
  /// Returns `true` if the separate transformation module is available as a
  /// result of parsing or construction.
  bool hasSeparateTransformModule() const {
    return detail::hasSharedTransformModuleImpl(sharedTransformModule);
  }

private:
  /// IR module containing the transform IR used for transformation.
  // TODO: Figure a better way to build a transform module and transport it in
  // the proper places in the IR as it is transformed by IREE so that it is
  // available with better ownership semantics.
  // Note: we wrap the OwningOpRef to get the desired destruction mechanism.
  // Note: shared_ptr is not great but we know the sharedTransformModule is
  // readonly.
  // Alternatives comprise:
  //   1. no shared_ptr but copying the module with every pass clone that the
  //      OpPassManager decides to perform.
  //   2. lifting ownership of the parsed transform module higher up in the
  //      IREE stack. This may be only shift the problem as we have passes
  //      building pass managers in IREE.
  //   3. build better support to embed the transformation module in the
  //      input IR and transport it to the place of use in IREE. This is deemed
  //      too intrusive atm.
  //   4. (future) config/resources mechanism that is being proposed in core?
  std::shared_ptr<OwningOpRef<ModuleOp>> sharedTransformModule = nullptr;
};

} // namespace transform
} // namespace mlir
#endif // IREE_DIALECTS_LINALG_TRANSFORM_TRANSFORM_INTERPRETER_UTILS_H
