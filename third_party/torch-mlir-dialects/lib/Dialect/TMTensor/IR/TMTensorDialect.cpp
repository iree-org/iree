//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::torch::TMTensor;

void TMTensorDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.cpp.inc"
      >();
}

#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.cpp.inc"
