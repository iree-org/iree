//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir-dialects/Dialect/TMTensor/IR/ScalarLoopOpInterface.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torch-mlir-tiled-op-interface"

using namespace mlir;
using namespace mlir::torch::TMTensor;

#include "torch-mlir-dialects/Dialect/TMTensor/IR/ScalarLoopOpInterface.cpp.inc"
