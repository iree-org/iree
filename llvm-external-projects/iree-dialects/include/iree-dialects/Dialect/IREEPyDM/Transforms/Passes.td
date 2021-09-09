// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_CONVERSION_TO_IREE_PASSES_TD
#define IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_CONVERSION_TO_IREE_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertIREEPyDMToIREE : Pass<"convert-iree-pydm-to-iree", "ModuleOp"> {
  let summary = "Convert iree_pydm modules to the IREE+related dialects";
  let description = [{
    Converts whole modules from Python in the iree_pydm dialect to the IREE
    dialect + various standard dialects.
  }];
  let constructor = "mlir::iree_pydm::createConvertIREEPyDMToIREEPass()";
  let dependentDialects = ["iree::IREEDialect"];
}

#endif // IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_CONVERSION_TO_IREE_PASSES_TD
