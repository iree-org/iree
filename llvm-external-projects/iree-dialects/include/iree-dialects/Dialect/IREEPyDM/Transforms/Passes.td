// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_CONVERSION_TO_IREE_PASSES_TD
#define IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_CONVERSION_TO_IREE_PASSES_TD

include "mlir/Pass/PassBase.td"

def FixateWeakNumeric : Pass<"fixate-weak-numeric", ""> {
  let summary = "Fixates weak numeric (integer/real/complex) to specific types";
  let description = [{
    After all type inference is complete, it is necessary to fixate weak numeric
    types to a concrete type (either an explicit size or arbitrary-precision).
    This can also be a useful thing to do early in development in order to
    eliminate these weak types.
  }];
  let constructor = "mlir::iree_pydm::createFixateWeakNumericPass()";
}

def LowerIREEPyDMToRTL : Pass<"lower-iree-pydm-to-rtl", "ModuleOp"> {
  let summary = "Lowers PyDM ops with runtime implementations to calls";
  let description = [{
    Once type propagation and optimizations are done, most still-remaining
    generic (object-based) ops need to be lowered to runtime library calls.
    This pass makes such conversions and emits imports.
  }];
  let constructor = "mlir::iree_pydm::createLowerIREEPyDMToRTLPass()";
}

def LinkIREEPyDMRTL : Pass<"link-iree-pydm-rtl", "ModuleOp"> {
  let summary = "Links any unresolved symbols functions against a runtime library";
  let description = [{
    Given a runtime library consisting of module-of-modules containing
    exported functions, repeatedly inlines matching references until no
    further unresolved externals are present.
  }];
  let options = [
    Option<"rtlFile", "rtl-file", "std::string", /*default=*/"std::string()",
           "File containing the RTL to link. Required for non-API use.">,
  ];
  let constructor = "mlir::iree_pydm::createLinkIREEPyDMRTLPass()";
}

def ConvertIREEPyDMToIREE : Pass<"convert-iree-pydm-to-iree", "ModuleOp"> {
  let summary = "Convert iree_pydm modules to the IREE+related dialects";
  let description = [{
    Converts whole modules from Python in the iree_pydm dialect to the IREE
    dialect + various standard dialects.
  }];
  let constructor = "mlir::iree_pydm::createConvertIREEPyDMToIREEPass()";
}

#endif // IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_CONVERSION_TO_IREE_PASSES_TD
