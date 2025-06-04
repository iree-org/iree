// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCMATTRS
#define IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCMATTRS

include "ROCMDialect.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// rocm.builtin.*
//===----------------------------------------------------------------------===//

def ROCM_BuiltinTuningModuleAttr :
    AttrDef<ROCM_Dialect, "BuiltinTuningModule", [
      DeclareAttrInterfaceMethods<Util_StoredModuleAttrInterface, [
        "getModule",
      ]>
    ]> {
  let mnemonic = "builtin.tuning_module";
  let summary = [{Represents a reference to a builtin tuning module.}];
  let description = [{
    This attribute carries a |builtin_filename| to mirror the files in
    "ROCM/builtins/tuning" that this attribute references. Those builtins are
    embedded directly in the compiler and parsed when this attribute is accessed
    via |getModule|. The parsed module is cached on the ROCM dialect to avoid
    duplicate parsing operations across multiple threads.
  }];

  let parameters = (ins
    StringRefParameter<>:$builtin_filename
  );
  let assemblyFormat = "`<` $builtin_filename `>`";
}

#endif // IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCMATTRS
