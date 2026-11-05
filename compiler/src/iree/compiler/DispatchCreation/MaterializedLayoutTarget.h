// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DISPATCHCREATION_MATERIALIZEDLAYOUTTARGET_H_
#define IREE_COMPILER_DISPATCHCREATION_MATERIALIZEDLAYOUTTARGET_H_

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::DispatchCreation {

/// Discardable module attribute holding the `#hal.executable.target` that the
/// module's tensor layouts were materialized for before dispatch creation.
///
/// A module carrying it already has target-specific layouts, e.g. linalg.pack
/// and linalg.mmt4d, instead of encodings. The attribute changes dispatch
/// creation as follows:
///  - iree-dispatch-creation-assign-data-tiling-encodings does not assign
///    encodings again, which would re-encode materialized ops.
///  - iree-dispatch-creation-verify-materialized-layout-target rejects
///    compiling the module for any other executable target, for several
///    targets, or for none.
/// The attribute is preserved across serialization so that compilation can be
/// resumed. Its accessors live here rather than in the Encoding dialect because
/// its value is a HAL attribute, and HAL depends on the Encoding dialect.
///
/// TODO: Materializing layouts before dispatch creation makes these phases
/// target aware on purpose. The layout decision, however, is spread over the
/// passes that read this attribute, the attribute ties the module to a single
/// executable target, and it makes DispatchCreation depend on HAL. Move the
/// layout decision into its own component that records one decision per
/// affinity, and let the passes query that component instead.
constexpr char kMaterializedLayoutTargetAttrName[] =
    "iree.encoding.materialized_layout_target";

/// Returns the executable target that the tensor layouts of `op`, or of the
/// module containing it, were materialized for. Returns null if the layouts
/// were not materialized.
IREE::HAL::ExecutableTargetAttr getMaterializedLayoutTarget(Operation *op);

/// Records that the tensor layouts of `moduleOp` were materialized for
/// `target`.
void setMaterializedLayoutTarget(ModuleOp moduleOp,
                                 IREE::HAL::ExecutableTargetAttr target);

/// Verifies that a module with materialized layouts is compiled for exactly the
/// executable target that its layouts were materialized for. Succeeds for
/// modules without materialized layouts.
LogicalResult verifyMaterializedLayoutTarget(ModuleOp moduleOp);

} // namespace mlir::iree_compiler::DispatchCreation

#endif // IREE_COMPILER_DISPATCHCREATION_MATERIALIZEDLAYOUTTARGET_H_
