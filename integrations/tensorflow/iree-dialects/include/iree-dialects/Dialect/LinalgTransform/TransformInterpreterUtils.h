// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_LINALG_TRANSFORM_TRANSFORM_INTERPRETER_UTILS_H
#define IREE_DIALECTS_LINALG_TRANSFORM_TRANSFORM_INTERPRETER_UTILS_H

#include <memory>

namespace mlir {
class Operation;
class LogicalResult;
namespace transform {
class TransformOpInterface;

/// Utility to parse the content of a `transformFileName` mlir file containing
/// a transform dialect specification.
LogicalResult
parseTransformModuleFromFile(MLIRContext *context,
                             llvm::StringRef transformFileName,
                             OwningOpRef<ModuleOp> &transformModule);

/// Utility to extract the `TransformOpInterface` ops that have the trait
/// `PossibleTopLevelTransformOpTrait`. Such ops are
LogicalResult
extractTopLevelTransformOps(Region &r,
                            SmallVectorImpl<TransformOpInterface> &res);

/// Utility to run a transform dialect specification contained in a
/// `transformRegion`, on a `target` op.
/// Since the transform dialect may use PDL which may modify the IR, the
/// underlying implementation clones the transform dialect operations before
/// applying them.
LogicalResult applyTransformsInRegion(Region &transformRegion,
                                      Operation *target);
} // namespace transform
} // namespace mlir
#endif // IREE_DIALECTS_LINALG_TRANSFORM_TRANSFORM_INTERPRETER_UTILS_H
