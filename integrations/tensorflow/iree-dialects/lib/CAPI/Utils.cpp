// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects-c/Utils.h"

#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

MlirOperation ireeLookupNearestSymbolFrom(MlirOperation fromOp,
                                          MlirAttribute symbolRefAttr) {
  auto symbolRefAttrCpp = unwrap(symbolRefAttr).cast<SymbolRefAttr>();
  return wrap(
      SymbolTable::lookupNearestSymbolFrom(unwrap(fromOp), symbolRefAttrCpp));
}
