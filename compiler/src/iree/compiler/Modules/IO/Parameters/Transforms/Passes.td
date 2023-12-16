// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_IO_PARAMETERS_PASSES
#define IREE_MODULES_IO_PARAMETERS_PASSES

include "mlir/Pass/PassBase.td"

def ParameterizeGlobals :
    Pass<"iree-io-parameterize-globals", "mlir::ModuleOp"> {
  let summary = "Moves all global initial values to a parameter archive";
  let constructor = "mlir::iree_compiler::IREE::IO::Parameters::createParameterizeGlobalsPass()";
  let dependentDialects = [
    "IREE::Stream::StreamDialect",
    "IREE::Util::UtilDialect",
  ];
  let options = [
    Option<"parameterNamespace", "parameter-namespace", "std::string",
           /*default=*/"",
           "Optional namespace to use for the exported parameters.">,
    Option<"archivePath", "archive-path", "std::string",
           /*default=*/"",
           "Path to write the parameter archive to.">
  ];
}

#endif  // IREE_MODULES_IO_PARAMETERS_PASSES
