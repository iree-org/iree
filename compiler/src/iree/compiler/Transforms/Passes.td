// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_TRANSFORMS_PASSES
#define IREE_COMPILER_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def IREEAffineExpandIndexOpsPass :
    Pass<"iree-affine-expand-index-ops", ""> {
  let summary = "Expand affine index ops with region simplification config";
}

def IREELowerAffinePass :
    Pass<"iree-lower-affine", ""> {
  let summary = "Lower affine operations to standard with region simplification config";
}

#endif // IREE_COMPILER_TRANSFORMS_PASSES
