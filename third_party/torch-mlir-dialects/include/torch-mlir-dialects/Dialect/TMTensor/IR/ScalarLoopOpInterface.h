//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_DIALECTS_DIALECT_TMTENSOR_IR_SCALARLOOPOPINTERFACE_H_
#define TORCH_MLIR_DIALECTS_DIALECT_TMTENSOR_IR_SCALARLOOPOPINTERFACE_H_

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"

/// Include the ODS generated interface header files.
#include "torch-mlir-dialects/Dialect/TMTensor/IR/ScalarLoopOpInterface.h.inc"

#endif // TORCH_MLIR_DIALECTS_DIALECT_TMTENSOR_IR_SCALARLOOPOPINTERFACE_H_
