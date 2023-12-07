// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_COMMONEXTENSIONS_H_
#define IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_COMMONEXTENSIONS_H_

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"

namespace mlir {

class DialectRegistry;

namespace func {
class FuncOp;
} // namespace func

namespace scf {
class ForallOp;
} // namespace scf

namespace transform {
// Types needed for builders.
struct TileSizesSpec;
struct NumThreadsSpec;
class TransformTypeInterface;
} // namespace transform

} // namespace mlir

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensionsOps.h.inc"

namespace mlir::iree_compiler {

/// Registers common transformations that require IREE-specific information
/// into the transform dialect.
void registerTransformDialectCommonExtension(DialectRegistry &registry);

namespace IREE::transform_dialect {
/// Hook to register common transformations to the transform dialect.
class CommonExtensions
    : public transform::TransformDialectExtension<CommonExtensions> {
public:
  CommonExtensions();
};
} // namespace IREE::transform_dialect

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_COMMONEXTENSIONS_H_
