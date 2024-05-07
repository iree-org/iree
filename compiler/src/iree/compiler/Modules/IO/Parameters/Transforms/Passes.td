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
  let summary = "Exports all global constants to an archive file when "
                "they are larger than the specified minimum size.";
  let dependentDialects = [
    "IREE::Flow::FlowDialect",
    "IREE::Util::UtilDialect",
  ];
  let options = [
    Option<"scopePath", "path", "std::string",
           /*default=*/"",
           "File path to an archive to export from with an optional "
           "`scope=` prefix.">,
    Option<"minimumSize", "minimum-size", "int64_t",
           /*default=*/"0",
           "Minimum size of a serialized global to export.">,
  ];
}

def GenerateSplatParameterArchivePass :
    Pass<"iree-io-generate-splat-parameter-archive", "mlir::ModuleOp"> {
  let summary = "Generates a .irpa file with splat entries for all parameters.";
  let options = [
    Option<"filePath", "file", "std::string",
           /*default=*/"",
           "Path to write the parameter archive to.">,
  ];
}

def ImportParametersPass :
    Pass<"iree-io-import-parameters", "mlir::ModuleOp"> {
  let summary = "Imports parameters from an archive file.";
  let dependentDialects = [
    "IREE::Flow::FlowDialect",
    "IREE::Util::UtilDialect",
  ];
  let options = [
    ListOption<"scopePaths", "paths", "std::string",
               "File paths to archives to import from with an optional "
               "`scope=` prefix.">,
    ListOption<"keys", "keys", "std::string",
               "List of parameter keys to import.">,
    Option<"maximumSize", "maximum-size", "int64_t",
           /*default=*/"9223372036854775807",
           "Maximum size of a serialized global to import.">,
  ];
}

#endif  // IREE_MODULES_IO_PARAMETERS_PASSES
