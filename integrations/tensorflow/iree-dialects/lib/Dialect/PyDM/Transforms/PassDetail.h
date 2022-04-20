// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_PASSDETAIL_H
#define IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {

namespace iree {
class IREEDialect;
}

namespace iree_compiler {
namespace IREE {
namespace PYDM {

class FuncOp;

#define GEN_PASS_CLASSES
#include "iree-dialects/Dialect/PyDM/Transforms/Passes.h.inc"

} // namespace PYDM
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_PASSDETAIL_H
