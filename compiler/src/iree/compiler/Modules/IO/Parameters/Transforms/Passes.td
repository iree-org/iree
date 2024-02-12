// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_IO_PARAMETERS_PASSES
#define IREE_MODULES_IO_PARAMETERS_PASSES

include "mlir/Pass/PassBase.td"

def ExportParametersPass :
    Pass<"iree-io-export-parameters", "mlir::ModuleOp"> {
  let summary = "Moves all inline constants of a minimum size and constant "
                "initialized globals values to a parameter archive";
  let dependentDialects = [
    "IREE::Stream::StreamDialect",
    "IREE::Util::UtilDialect",
  ];
  let options = [
    Option<"parameterScope", "scope", "std::string",
           /*default=*/"",
           "Optional scope to use for the exported parameters.">,
    Option<"archivePath", "archive-path", "std::string",
           /*default=*/"",
           "Path to write the parameter archive to.">,
    Option<"minimumSize", "minimum-size", "int64_t",
           /*default=*/"256",
           "Minimum size of a serialized global to export.">
  ];
}

def GenerateSplatParameterArchivePass :
    Pass<"iree-io-generate-splat-parameter-archive", "mlir::ModuleOp"> {
  let summary = "Generates a .irpa file with splat entries for all parameters";
  let dependentDialects = [];
  let options = [
    Option<"archivePath", "archive-path", "std::string",
           /*default=*/"",
           "Path to write the parameter archive to.">
  ];
}

#endif  // IREE_MODULES_IO_PARAMETERS_PASSES
