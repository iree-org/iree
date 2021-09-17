// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_COMMON_PASSES
#define IREE_COMPILER_INPUTCONVERSION_COMMON_PASSES

include "mlir/Pass/PassBase.td"

def TopLevelSCFToCFG :
    Pass<"iree-top-level-scf-to-cfg", "FuncOp"> {
  let summary = "Converts non-nested SCF constructs to CFG (not traversing into opaque operations).";
  let constructor = "mlir::iree_compiler::createTopLevelSCFToCFGPass()";
}

def IREEImportPublic :
    Pass<"iree-import-public", "ModuleOp"> {
  let summary = "Imports IREE public dialect to internal implementation.";
  let constructor = "mlir::iree_compiler::createIREEImportPublicPass()";
}

def PromoteI1ToI8 :
    Pass<"iree-promote-i1-to-i8", "mlir::FuncOp"> {
  let summary = "Legalizes i1 tensor constants to i8s";
  let constructor = "mlir::iree_compiler::createPromoteI1ToI8Pass()";
}

#endif // IREE_COMPILER_INPUTCONVERSION_COMMON_PASSES
