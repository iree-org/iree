// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MLIR_DIALECT_LINALG_IR_LINALGTRANSFORMOPS_H
#define MLIR_DIALECT_LINALG_IR_LINALGTRANSFORMOPS_H

#include "TrackingListener.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgTransform/TrackingListener.h"
#include "iree-dialects/Dialect/LinalgTransform/TransformOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace scf {
class ForOp;
} // namespace scf
} // namespace mlir

#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformDialect.h.inc"

#define GET_OP_CLASSES
#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h.inc"

namespace mlir {
namespace linalg {
namespace transform {

/// Base class for extensions of the Transform dialect that supports injecting
/// operations into the Transform dialect at load time. Concrete extensions
/// are expected to derive this class and register operations in the
/// constructor. They can be registered with the DialectRegistry and
/// automatically applied to the Transform dialect when it is loaded.
using TransformDialect = LinalgTransformDialect;
template <typename DerivedTy, typename... ExtraDialects>
class TransformDialectExtension
    : public DialectExtension<DerivedTy, TransformDialect, ExtraDialects...> {
  using Initializer = std::function<void(TransformDialect *)>;
  using DialectLoader = std::function<void(MLIRContext *)>;

public:
  /// Extension application hook. Actually loads the dependent dialects and
  /// registers the additional operations. Not expected to be called directly.
  void apply(MLIRContext *context, TransformDialect *transformDialect,
             ExtraDialects *...) const final {
    for (const DialectLoader &loader : dialectLoaders)
      loader(context);
    for (const Initializer &init : opInitializers)
      init(transformDialect);
  }

protected:
  /// Injects the operation into the Transform dialect.
  template <typename OpTy>
  void registerTransformOp() {
    opInitializers.push_back([](TransformDialect *transformDialect) {
      RegisteredOperationName::insert<OpTy>(*transformDialect);
    });
  }

  /// Injects the operations into the Transform dialect.
  template <typename... OpTys>
  void registerTransformOps() {
    (void)std::initializer_list<int>{(registerTransformOp<OpTys>(), 0)...};
  }

  /// Declares that the Transform dialect depends on the dialect provided as
  /// template parameter. When the Transform dialect is loaded, dependent
  /// dialects will be loaded as well. This is intended for dialects that
  /// contain attributes and types used in creation and canonicalization of
  /// the injected operations.
  template <typename DialectTy>
  void declareDependentDialect() {
    dialectLoaders.push_back(
        [](MLIRContext *context) { context->loadDialect<DialectTy>(); });
  }

private:
  SmallVector<Initializer> opInitializers;
  SmallVector<DialectLoader> dialectLoaders;
};

} // namespace transform
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_IR_LINALGTRANSFORMOPS_H
