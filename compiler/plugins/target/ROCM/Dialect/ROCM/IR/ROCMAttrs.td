// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCMATTRS
#define IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCMATTRS

include "ROCMDialect.td"
include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.td"
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

def ROCM_UKernelProviderAttr  :
    AttrDef<ROCM_Dialect, "UKernelProvider", [
    DeclareAttrInterfaceMethods<IREECodegen_UKernelProviderInterface, [
        "createAndReplaceWithUkernelOp"
      ]>
    ]> {
  let mnemonic = "ukernel_provider";
  let summary = [{
    An attribute that provides context specific ukernel implementations for ROCM.
  }];
  let parameters = (ins);
  let assemblyFormat = [{}];
}

//===---------------------------------------------------------------------===//
// rocm.tensor_ukernel_provider
//===---------------------------------------------------------------------===//

def ROCM_TensorUKernelProviderAttr  :
    AttrDef<ROCM_Dialect, "TensorUKernelProvider", [
    DeclareAttrInterfaceMethods<IREECodegen_UKernelProviderInterface, [
      "getDataLayoutForUKernel",
      "getMLIRUKernel",
      ]>
    ]> {
  let mnemonic = "tensor_ukernel_provider";
  let summary = [{
    An attribute that provides context specific tensor ukernel implementations for ROCM.
  }];
  let parameters = (ins);
  let assemblyFormat = [{}];
}

//===---------------------------------------------------------------------===//
// rocm.ukernel_info
//===---------------------------------------------------------------------===//

def ROCM_UKernelInfoAttr : AttrDef<ROCM_Dialect, "UKernelInfo", []> {
  let mnemonic = "ukernel_info";
  let summary = [{
    An attribute that provides metadata around a ukernel, e.g. for matching.
  }];
  let parameters =
      (ins AttrParameter<
           "DictionaryAttr",
           "Parameters controlling matching this ukernel, in addition to ISA "
           "requirements from the mma attribute.">:$match,
          DefaultValuedParameter<"int64_t", "0",
                                 "Determines which ukernel is selected when "
                                 "multiple ones are applicable.">:$benefit,
          OptionalParameter<"Attribute",
                            "MMAAttr-like attribute if applicable, dictates "
                            "layout and/or ISA requirements">:$mma);
  let assemblyFormat = "`<` struct(params) `>`";
}

//===---------------------------------------------------------------------===//
// rocm.ukernel_interation_size_constraint
//===---------------------------------------------------------------------===//

def ROCM_UKernelIterationSizeConstraint
    : AttrDef<ROCM_Dialect, "UKernelIterationSizeConstraint", []> {
  let mnemonic = "ukernel_interation_size_constraint";
  let summary = [{
    An attribute that provides a matching constraint on an iteration size.
  }];
  let parameters = (ins OptionalParameter<
                        "IntegerAttr",
                        "Index of iteration dimension to match">:$index,
      OptionalParameter<"IntegerAttr",
                        "Minimum iteration dimension size to match">:$size_min,
      OptionalParameter<"IntegerAttr",
                        "Maximum iteration dimension size to match">:$size_max,
      OptionalParameter<"IntegerAttr", "Divisor of iteration dimension size to "
                                       "match">:$size_div);
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCMATTRS
