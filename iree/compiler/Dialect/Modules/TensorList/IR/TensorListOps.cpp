// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListOps.h"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListOps.cpp.inc"
