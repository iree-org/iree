//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_DIALECTS_DIALECT_TMTENSOR_IR_TMTENSORINTERFACES_H_
#define TORCH_MLIR_DIALECTS_DIALECT_TMTENSOR_IR_TMTENSORINTERFACES_H_

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace torch {
namespace TMTensor {
class TMTensorOp;

namespace detail {
LogicalResult verifyTMTensorOpInterface(Operation *op);
}

#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.h.inc" // IWYU pragma: export

/// Include the generated interface declarations.
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOpInterfaces.h.inc" // IWYU pragma: export

} // namespace TMTensor
} // namespace torch
} // namespace mlir

#endif // TORCH_MLIR_DIALECTS_DIALECT_TMTENSOR_IR_TMTENSORINTERFACES_H_
