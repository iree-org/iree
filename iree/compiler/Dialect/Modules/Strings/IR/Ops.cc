// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Modules/Strings/IR/Ops.h"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/DialectConversion.h"

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Modules/Strings/IR/Ops.cc.inc"
